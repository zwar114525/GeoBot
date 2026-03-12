import json
from dataclasses import dataclass, field
from loguru import logger
from src.ingestion.pdf_processor import extract_text_from_pdf, extract_pages_with_numbers
from src.vectordb.qdrant_store import GeoVectorStore
from src.utils.llm_client import call_llm
from src.utils.json_validator import (
    extract_json_from_response,
    safe_parse_json,
)
from src.schemas.validator_schemas import (
    CompletenessCheckSchema,
    SectionCheckSchema,
    CheckStatus,
    CheckSeverity,
    ParameterCheckSchema,
    FactorOfSafetySchema,
    ParameterIssueSchema,
    CodeComplianceSchema,
    DesignMethodSchema,
    ReferencedCodeSchema,
    ConsistencyCheckSchema,
    ConsistencyIssueSchema,
    ValidationCheckItemSchema,
    CheckCategory,
)
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

    def _check_completeness(self, text: str) -> list[CheckItem]:
        expected = [{"number": s.number, "title": s.title} for s in STANDARD_REPORT_TEMPLATE if s.required]
        prompt = (
            "Determine whether each required section is present, absent, or inadequate in the report.\n"
            f"Expected sections: {json.dumps(expected)}\n"
            f"Report text (first 8000 chars): {text[:8000]}\n\n"
            "Return JSON array where each item has: section (string), status (one of: present/absent/inadequate), notes (string explaining the assessment)."
        )
        
        try:
            response = call_llm(prompt, temperature=0.1, max_tokens=1800)
            result = safe_parse_json(
                response=response,
                target_type=list[SectionCheckSchema],
                default_factory=list,
                context="completeness check",
            )
        except Exception as e:
            logger.error(f"Completeness check parsing failed: {e}")
            return [
                CheckItem(
                    category="completeness",
                    item="Section completeness check",
                    status="warning",
                    details=f"Could not parse LLM response: {e}",
                    severity="medium",
                )
            ]
        
        checks = []
        status_map = {"present": "pass", "inadequate": "warning", "absent": "fail"}
        
        for item in result:
            try:
                status = status_map.get(item.status, "not_checked")
                severity = "high" if item.status == "absent" else "medium"
                checks.append(
                    CheckItem(
                        category="completeness",
                        item=f"Section: {item.section}",
                        status=status,
                        details=item.notes or "",
                        severity=severity,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process completeness item: {e}")
                continue
        
        return checks

    def _check_parameters(self, text: str) -> list[CheckItem]:
        prompt = (
            "Extract key geotechnical parameters and factors of safety from the report.\n"
            f"Report text (first 10000 chars): {text[:10000]}\n\n"
            "Return JSON with: factors_of_safety (object with bearing/sliding/overturning/slope as numbers), "
            "issues_noticed (array of strings describing any parameter issues found)."
        )
        
        try:
            response = call_llm(prompt, temperature=0.1, max_tokens=1800)
            params = safe_parse_json(
                response=response,
                target_type=ParameterCheckSchema,
                default_factory=ParameterCheckSchema,
                context="parameter extraction",
            )
        except Exception as e:
            logger.error(f"Parameter check parsing failed: {e}")
            return [
                CheckItem(
                    category="parameter",
                    item="Parameter extraction",
                    status="warning",
                    details=f"Could not parse LLM response: {e}",
                    severity="medium",
                )
            ]
        
        checks = []
        fos_req = {"bearing": 3.0, "sliding": 1.5, "overturning": 2.0, "slope": 1.4}
        
        # Extract FoS values from the validated schema
        fos_found = params.factors_of_safety
        
        for fos_type, min_value in fos_req.items():
            fos_value = getattr(fos_found, fos_type, None)
            if fos_value is None:
                continue
            if fos_value < min_value:
                checks.append(
                    CheckItem(
                        category="parameter",
                        item=f"FoS ({fos_type})",
                        status="fail",
                        details=f"{fos_value} < {min_value} (minimum required)",
                        severity="critical",
                    )
                )
            else:
                checks.append(
                    CheckItem(
                        category="parameter",
                        item=f"FoS ({fos_type})",
                        status="pass",
                        details=f"{fos_value} >= {min_value} (minimum required)",
                    )
                )
        
        # Add noticed issues as warnings
        for issue in params.issues_noticed:
            checks.append(
                CheckItem(
                    category="parameter",
                    item="Noticed issue",
                    status="warning",
                    details=str(issue),
                )
            )
        
        return checks

    def _check_code_compliance(self, text: str) -> list[CheckItem]:
        prompt = (
            "Identify design methods and referenced codes/standards from the report.\n"
            f"Report text (first 6000 chars): {text[:6000]}\n\n"
            "Return JSON with: design_methods (array of objects with method_name, code_reference, description), "
            "referenced_codes (array of objects with code_name, clause, year if available)."
        )
        
        try:
            response = call_llm(prompt, temperature=0.1, max_tokens=1000)
            compliance = safe_parse_json(
                response=response,
                target_type=CodeComplianceSchema,
                default_factory=CodeComplianceSchema,
                context="code compliance check",
            )
        except Exception as e:
            logger.error(f"Code compliance parsing failed: {e}")
            return [
                CheckItem(
                    category="compliance",
                    item="Code compliance check",
                    status="warning",
                    details=f"Could not parse LLM response: {e}",
                    severity="medium",
                )
            ]
        
        checks = []
        
        for method in compliance.design_methods:
            try:
                chunks = self.store.search(
                    query=f"design method {method.method_name} requirements",
                    top_k=3,
                    document_type="code",
                )
                if chunks:
                    checks.append(
                        CheckItem(
                            category="compliance",
                            item=f"Design method: {method.method_name}",
                            status="pass",
                            details=f"Method found in code references. {method.description or ''}",
                            reference=method.code_reference or "",
                        )
                    )
                else:
                    checks.append(
                        CheckItem(
                            category="compliance",
                            item=f"Design method: {method.method_name}",
                            status="warning",
                            details="No direct code chunk retrieved for this method.",
                            reference=method.code_reference or "",
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to verify design method {method.method_name}: {e}")
                continue
        
        return checks

    def _check_consistency(self, text: str) -> list[CheckItem]:
        prompt = (
            "Find internal inconsistencies in the geotechnical report (conflicting values, contradictory statements, mismatched parameters).\n"
            f"Report text (first 12000 chars): {text[:12000]}\n\n"
            "Return JSON array where each item has: issue (string describing the inconsistency), "
            "severity (one of: low/medium/high/critical), sections_involved (array of section names/numbers)."
        )
        
        try:
            response = call_llm(
                prompt,
                temperature=0.2,
                max_tokens=1200,
                model=REASONING_MODEL,
            )
            consistency = safe_parse_json(
                response=response,
                target_type=list[ConsistencyIssueSchema],
                default_factory=list,
                context="consistency check",
            )
        except Exception as e:
            logger.error(f"Consistency check parsing failed: {e}")
            return [
                CheckItem(
                    category="consistency",
                    item="Internal consistency",
                    status="warning",
                    details=f"Could not parse LLM response: {e}",
                    severity="medium",
                )
            ]
        
        if not consistency:
            return [
                CheckItem(
                    category="consistency",
                    item="Internal consistency",
                    status="pass",
                    details="No obvious inconsistencies detected",
                )
            ]
        
        checks = []
        for issue in consistency:
            try:
                status = "fail" if issue.severity == CheckSeverity.HIGH else "warning"
                sections_str = ", ".join(issue.sections_involved) if issue.sections_involved else "N/A"
                checks.append(
                    CheckItem(
                        category="consistency",
                        item=issue.issue,
                        status=status,
                        details=f"Sections involved: {sections_str}",
                        severity=issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process consistency issue: {e}")
                continue
        
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
