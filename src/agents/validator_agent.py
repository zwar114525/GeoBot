import json
from dataclasses import dataclass, field
from src.ingestion.pdf_processor import extract_text_from_pdf, extract_pages_with_numbers
from src.vectordb.qdrant_store import GeoVectorStore
from src.utils.llm_client import call_llm
from src.templates.report_structure import STANDARD_REPORT_TEMPLATE
from config.settings import REASONING_MODEL


@dataclass
class CheckItem:
    category: str
    item: str
    status: str
    details: str = ""
    reference: str = ""
    severity: str = "medium"


@dataclass
class ValidationReport:
    document_name: str
    checks: list[CheckItem] = field(default_factory=list)
    summary: str = ""
    overall_status: str = ""

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "warning")


class ValidatorAgent:
    def __init__(self, use_local_db: bool = False):
        self.store = GeoVectorStore(use_local=use_local_db)

    def validate_report(self, pdf_path: str) -> ValidationReport:
        document_name = pdf_path.split("/")[-1]
        report = ValidationReport(document_name=document_name)
        full_text = extract_text_from_pdf(pdf_path)
        _ = extract_pages_with_numbers(pdf_path)
        if not full_text or len(full_text.strip()) < 200:
            report.checks.append(
                CheckItem(
                    category="extraction",
                    item="Document readability",
                    status="fail",
                    details="Could not extract meaningful text from PDF.",
                    severity="critical",
                )
            )
            report.overall_status = "rejected"
            return report
        report.checks.extend(self._check_completeness(full_text))
        report.checks.extend(self._check_parameters(full_text))
        report.checks.extend(self._check_code_compliance(full_text))
        report.checks.extend(self._check_consistency(full_text))
        report.summary = self._generate_summary(report)
        critical_fails = sum(1 for c in report.checks if c.status == "fail" and c.severity == "critical")
        if critical_fails > 0:
            report.overall_status = "rejected"
        elif report.fail_count > 0:
            report.overall_status = "revisions_needed"
        else:
            report.overall_status = "acceptable"
        return report

    def _json_or_default(self, text: str, default):
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except Exception:
            return default

    def _check_completeness(self, text: str) -> list[CheckItem]:
        expected = [{"number": s.number, "title": s.title} for s in STANDARD_REPORT_TEMPLATE if s.required]
        prompt = (
            "Determine whether each section is present/absent/inadequate.\n"
            f"Expected sections: {json.dumps(expected)}\n"
            f"Report text: {text[:8000]}\n"
            "Return JSON array with section,status,notes."
        )
        result = self._json_or_default(call_llm(prompt, temperature=0.1, max_tokens=1800), [])
        checks = []
        for item in result:
            status_map = {"present": "pass", "inadequate": "warning", "absent": "fail"}
            checks.append(
                CheckItem(
                    category="completeness",
                    item=f"Section: {item.get('section', 'unknown')}",
                    status=status_map.get(item.get("status"), "not_checked"),
                    details=item.get("notes", ""),
                    severity="high" if item.get("status") == "absent" else "medium",
                )
            )
        return checks

    def _check_parameters(self, text: str) -> list[CheckItem]:
        prompt = (
            "Extract key parameters as JSON with factors_of_safety and issues_noticed.\n"
            f"Report text: {text[:10000]}"
        )
        params = self._json_or_default(call_llm(prompt, temperature=0.1, max_tokens=1800), {})
        checks = []
        fos_req = {"bearing": 3.0, "sliding": 1.5, "overturning": 2.0, "slope": 1.4}
        fos_found = params.get("factors_of_safety", {})
        for fos_type, min_value in fos_req.items():
            fos_value = fos_found.get(fos_type)
            if fos_value is None:
                continue
            if fos_value < min_value:
                checks.append(CheckItem(category="parameter", item=f"FoS ({fos_type})", status="fail", details=f"{fos_value} < {min_value}", severity="critical"))
            else:
                checks.append(CheckItem(category="parameter", item=f"FoS ({fos_type})", status="pass", details=f"{fos_value} >= {min_value}"))
        for issue in params.get("issues_noticed", []):
            checks.append(CheckItem(category="parameter", item="Noticed issue", status="warning", details=str(issue)))
        return checks

    def _check_code_compliance(self, text: str) -> list[CheckItem]:
        prompt = (
            "Identify design_methods and referenced_codes as JSON.\n"
            f"Report text: {text[:6000]}"
        )
        methods = self._json_or_default(call_llm(prompt, temperature=0.1, max_tokens=1000), {})
        checks = []
        for method in methods.get("design_methods", []):
            chunks = self.store.search(query=f"design method {method} requirements", top_k=3, document_type="code")
            if chunks:
                checks.append(CheckItem(category="compliance", item=f"Design method: {method}", status="pass", details="Method found in code references."))
            else:
                checks.append(CheckItem(category="compliance", item=f"Design method: {method}", status="warning", details="No direct code chunk retrieved."))
        return checks

    def _check_consistency(self, text: str) -> list[CheckItem]:
        prompt = (
            "Find internal inconsistencies and return JSON array of issue,severity,sections_involved.\n"
            f"Report text: {text[:12000]}"
        )
        issues = self._json_or_default(call_llm(prompt, temperature=0.2, max_tokens=1200, model=REASONING_MODEL), [])
        if not issues:
            return [CheckItem(category="consistency", item="Internal consistency", status="pass", details="No obvious inconsistencies detected")]
        checks = []
        for issue in issues:
            sev = issue.get("severity", "medium")
            checks.append(
                CheckItem(
                    category="consistency",
                    item=issue.get("issue", "Consistency issue"),
                    status="fail" if sev == "high" else "warning",
                    details=f"Sections involved: {issue.get('sections_involved', 'N/A')}",
                    severity=sev,
                )
            )
        return checks

    def _generate_summary(self, report: ValidationReport) -> str:
        prompt = (
            "Write a brief professional summary from this data:\n"
            f"{json.dumps({'total': len(report.checks), 'pass': report.pass_count, 'fail': report.fail_count, 'warnings': report.warning_count})}"
        )
        return call_llm(prompt, temperature=0.3, max_tokens=500)

    def format_validation_report(self, report: ValidationReport) -> str:
        lines = [
            "# SUBMISSION REVIEW REPORT",
            "",
            f"**Document Reviewed:** {report.document_name}",
            f"**Overall Status:** {report.overall_status.upper().replace('_', ' ')}",
            f"**Checks Performed:** {len(report.checks)} (✓ {report.pass_count} | ✗ {report.fail_count} | ⚠ {report.warning_count})",
            "",
            "---",
            "",
            "## Summary",
            "",
            report.summary,
            "",
            "---",
            "",
            "## Detailed Findings",
            "",
            "| Status | Category | Item | Details |",
            "|--------|----------|------|---------|",
        ]
        for c in report.checks:
            lines.append(f"| {c.status.upper()} | {c.category} | {c.item} | {c.details} |")
        lines.extend(["", "---", "", "## Recommended Actions", "", "Address all FAIL items before acceptance."])
        return "\n".join(lines)
