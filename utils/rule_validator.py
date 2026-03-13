"""
Rule-based validation checks for geotechnical reports.
Provides deterministic checks that don't rely on LLM.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum


class CheckSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_CHECKED = "not_checked"


@dataclass
class RuleCheck:
    """Definition of a rule-based check."""
    id: str
    name: str
    description: str
    category: str
    severity: CheckSeverity
    check_fn: Callable[[str], tuple]  # Returns (passed: bool, details: str)
    reference: str = ""  # Code reference


@dataclass
class CheckResult:
    """Result of a rule check."""
    rule_id: str
    rule_name: str
    status: CheckStatus
    details: str
    severity: CheckSeverity
    reference: str = ""
    category: str = ""


class RuleBasedValidator:
    """
    Rule-based validator for geotechnical reports.
    Provides deterministic checks for common requirements.
    """
    
    def __init__(self):
        self.rules: List[RuleCheck] = []
        self._register_builtin_rules()
    
    def _register_builtin_rules(self):
        """Register built-in rule checks."""
        # Factor of Safety Checks
        self.rules.append(RuleCheck(
            id="fos_bearinging",
            name="Bearing Capacity Factor of Safety",
            description="Check minimum factor of safety for bearing capacity",
            category="parameter",
            severity=CheckSeverity.CRITICAL,
            check_fn=self._check_fos_bearing,
            reference="HK CoP 2017 Section 6.1.5",
        ))
        
        self.rules.append(RuleCheck(
            id="fos_sliding",
            name="Sliding Factor of Safety",
            description="Check minimum factor of safety against sliding",
            category="parameter",
            severity=CheckSeverity.CRITICAL,
            check_fn=self._check_fos_sliding,
            reference="HK CoP 2017 Section 6.4",
        ))
        
        self.rules.append(RuleCheck(
            id="fos_overturning",
            name="Overturning Factor of Safety",
            description="Check minimum factor of safety against overturning",
            category="parameter",
            severity=CheckSeverity.CRITICAL,
            check_fn=self._check_fos_overturning,
            reference="HK CoP 2017 Section 6.4",
        ))
        
        self.rules.append(RuleCheck(
            id="fos_slope",
            name="Slope Stability Factor of Safety",
            description="Check minimum factor of safety for slope stability",
            category="parameter",
            severity=CheckSeverity.CRITICAL,
            check_fn=self._check_fos_slope,
            reference="Geotechnical Manual for Slopes",
        ))
        
        # Required Section Checks
        self.rules.append(RuleCheck(
            id="section_intro",
            name="Introduction Section",
            description="Check for presence of introduction section",
            category="completeness",
            severity=CheckSeverity.MEDIUM,
            check_fn=self._check_section_intro,
            reference="",
        ))
        
        self.rules.append(RuleCheck(
            id="section_gi",
            name="Ground Investigation Section",
            description="Check for presence of ground investigation section",
            category="completeness",
            severity=CheckSeverity.HIGH,
            check_fn=self._check_section_gi,
            reference="",
        ))
        
        self.rules.append(RuleCheck(
            id="section_analysis",
            name="Analysis and Design Section",
            description="Check for presence of analysis and design section",
            category="completeness",
            severity=CheckSeverity.HIGH,
            check_fn=self._check_section_analysis,
            reference="",
        ))
        
        # Parameter Consistency Checks
        self.rules.append(RuleCheck(
            id="param_cohesion_range",
            name="Cohesion Value Range",
            description="Check cohesion values are within reasonable range",
            category="parameter",
            severity=CheckSeverity.MEDIUM,
            check_fn=self._check_cohesion_range,
            reference="",
        ))
        
        self.rules.append(RuleCheck(
            id="param_friction_range",
            name="Friction Angle Range",
            description="Check friction angle values are within reasonable range",
            category="parameter",
            severity=CheckSeverity.MEDIUM,
            check_fn=self._check_friction_range,
            reference="",
        ))
        
        # Required Information Checks
        self.rules.append(RuleCheck(
            id="info_project_name",
            name="Project Name",
            description="Check for project name identification",
            category="completeness",
            severity=CheckSeverity.LOW,
            check_fn=self._check_project_name,
            reference="",
        ))
        
        self.rules.append(RuleCheck(
            id="info_date",
            name="Report Date",
            description="Check for report date",
            category="completeness",
            severity=CheckSeverity.LOW,
            check_fn=self._check_report_date,
            reference="",
        ))
        
        self.rules.append(RuleCheck(
            id="info_author",
            name="Author/Company Information",
            description="Check for author or company identification",
            category="completeness",
            severity=CheckSeverity.LOW,
            check_fn=self._check_author_info,
            reference="",
        ))
    
    def validate(self, report_text: str) -> List[CheckResult]:
        """
        Run all rule-based checks on report text.
        
        Args:
            report_text: Full report text
            
        Returns:
            List of CheckResult objects
        """
        results = []
        
        for rule in self.rules:
            try:
                passed, details = rule.check_fn(report_text)
                
                if passed:
                    status = CheckStatus.PASS
                elif rule.severity == CheckSeverity.CRITICAL:
                    status = CheckStatus.FAIL
                else:
                    status = CheckStatus.WARNING
                
                results.append(CheckResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    status=status,
                    details=details,
                    severity=rule.severity,
                    reference=rule.reference,
                    category=rule.category,
                ))
            except Exception as e:
                results.append(CheckResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    status=CheckStatus.NOT_CHECKED,
                    details=f"Check failed: {str(e)}",
                    severity=rule.severity,
                    reference=rule.reference,
                    category=rule.category,
                ))
        
        return results
    
    # ============== Factor of Safety Checks ==============
    
    def _check_fos_bearing(self, text: str) -> tuple:
        """Check bearing capacity FoS >= 3.0"""
        # Look for bearing capacity FoS mentions
        patterns = [
            r"(?:bearing\s*(?:capacity)?)\s*(?:factor\s*of\s*safety|fos|f\.?o\.?s\.?)\s*(?:=|is|of)\s*([\d.]+)",
            r"(?:factor\s*of\s*safety|fos)\s*(?:for\s*)?(?:bearing)\s*(?:=|:)\s*([\d.]+)",
            r"FoS\s*(?:for\s*bearing|bearing)\s*=\s*([\d.]+)",
        ]
        
        found_values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    found_values.append(float(match))
                except ValueError:
                    continue
        
        if not found_values:
            return False, "No bearing capacity factor of safety found in report"
        
        min_fos = min(found_values)
        if min_fos >= 3.0:
            return True, f"Bearing capacity FoS = {min_fos:.2f} >= 3.0 (OK)"
        else:
            return False, f"Bearing capacity FoS = {min_fos:.2f} < 3.0 (FAIL)"
    
    def _check_fos_sliding(self, text: str) -> tuple:
        """Check sliding FoS >= 1.5"""
        patterns = [
            r"(?:sliding)\s*(?:factor\s*of\s*safety|fos)\s*(?:=|is|of)\s*([\d.]+)",
            r"(?:factor\s*of\s*safety|fos)\s*(?:against\s*)?(?:sliding)\s*(?:=|:)\s*([\d.]+)",
        ]
        
        found_values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    found_values.append(float(match))
                except ValueError:
                    continue
        
        if not found_values:
            return False, "No sliding factor of safety found"
        
        min_fos = min(found_values)
        if min_fos >= 1.5:
            return True, f"Sliding FoS = {min_fos:.2f} >= 1.5 (OK)"
        else:
            return False, f"Sliding FoS = {min_fos:.2f} < 1.5 (FAIL)"
    
    def _check_fos_overturning(self, text: str) -> tuple:
        """Check overturning FoS >= 2.0"""
        patterns = [
            r"(?:overturning)\s*(?:factor\s*of\s*safety|fos)\s*(?:=|is|of)\s*([\d.]+)",
            r"(?:factor\s*of\s*safety|fos)\s*(?:against\s*)?(?:overturning)\s*(?:=|:)\s*([\d.]+)",
        ]
        
        found_values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    found_values.append(float(match))
                except ValueError:
                    continue
        
        if not found_values:
            return False, "No overturning factor of safety found"
        
        min_fos = min(found_values)
        if min_fos >= 2.0:
            return True, f"Overturning FoS = {min_fos:.2f} >= 2.0 (OK)"
        else:
            return False, f"Overturning FoS = {min_fos:.2f} < 2.0 (FAIL)"
    
    def _check_fos_slope(self, text: str) -> tuple:
        """Check slope stability FoS >= 1.4"""
        patterns = [
            r"(?:slope)\s*(?:stability)?\s*(?:factor\s*of\s*safety|fos)\s*(?:=|is|of)\s*([\d.]+)",
            r"(?:factor\s*of\s*safety|fos)\s*(?:for\s*)?(?:slope)\s*(?:=|:)\s*([\d.]+)",
        ]
        
        found_values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    found_values.append(float(match))
                except ValueError:
                    continue
        
        if not found_values:
            return False, "No slope stability factor of safety found"
        
        min_fos = min(found_values)
        if min_fos >= 1.4:
            return True, f"Slope FoS = {min_fos:.2f} >= 1.4 (OK)"
        else:
            return False, f"Slope FoS = {min_fos:.2f} < 1.4 (FAIL)"
    
    # ============== Section Checks ==============
    
    def _check_section_intro(self, text: str) -> tuple:
        """Check for introduction section"""
        patterns = [
            r"#+\s*(?:1\.?)?\s*introduction",
            r"#+\s*project\s*(?:background|overview)",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "Introduction section found"
        
        return False, "No introduction section found"
    
    def _check_section_gi(self, text: str) -> tuple:
        """Check for ground investigation section"""
        patterns = [
            r"#+\s*(?:ground\s*investigation|site\s*investigation|GI)",
            r"#+\s*(?:borehole|field\s*work|laboratory\s*testing)",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "Ground investigation section found"
        
        return False, "No ground investigation section found"
    
    def _check_section_analysis(self, text: str) -> tuple:
        """Check for analysis and design section"""
        patterns = [
            r"#+\s*(?:analysis\s*(?:and)?\s*design|design\s*(?:analysis)?|calculations)",
            r"#+\s*(?:foundation\s*design|bearing\s*capacity|slope\s*stability)",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "Analysis and design section found"
        
        return False, "No analysis and design section found"
    
    # ============== Parameter Range Checks ==============
    
    def _check_cohesion_range(self, text: str) -> tuple:
        """Check cohesion values are reasonable (0-500 kPa)"""
        patterns = [
            r"cohesion\s*(?:=|:|of)\s*([\d.]+)\s*(?:kpa|kpa)?",
            r"c\s*(?:=|:)\s*([\d.]+)\s*(?:kpa|kpa)?",
            r"([\d.]+)\s*kpa\s*cohesion",
        ]
        
        found_values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    found_values.append(float(match))
                except ValueError:
                    continue
        
        if not found_values:
            return True, "No cohesion values to check"
        
        out_of_range = [v for v in found_values if v < 0 or v > 500]
        
        if out_of_range:
            return False, f"Cohesion values out of range (0-500 kPa): {out_of_range}"
        
        return True, f"All cohesion values in reasonable range: {found_values}"
    
    def _check_friction_range(self, text: str) -> tuple:
        """Check friction angle values are reasonable (0-50 degrees)"""
        patterns = [
            r"(?:friction\s*angle|phi)\s*(?:=|:|of)\s*([\d.]+)\s*(?:degrees|deg|°)?",
            r"φ\s*(?:=|:)\s*([\d.]+)",
            r"([\d.]+)\s*(?:degrees|deg|°)\s*(?:friction|friction\s*angle)",
        ]
        
        found_values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    found_values.append(float(match))
                except ValueError:
                    continue
        
        if not found_values:
            return True, "No friction angle values to check"
        
        out_of_range = [v for v in found_values if v < 0 or v > 50]
        
        if out_of_range:
            return False, f"Friction angles out of range (0-50°): {out_of_range}"
        
        return True, f"All friction angles in reasonable range: {found_values}"
    
    # ============== Required Information Checks ==============
    
    def _check_project_name(self, text: str) -> tuple:
        """Check for project name"""
        patterns = [
            r"#+\s*(?:project\s*name|project\s*title)\s*[:\s]*(.+)",
            r"project\s*(?:name|title)\s*[:\s]+([A-Za-z][^\n]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return True, f"Project name found: {match.group(1).strip()[:50]}"
        
        return False, "No project name found"
    
    def _check_report_date(self, text: str) -> tuple:
        """Check for report date"""
        patterns = [
            r"(?:date|dated)\s*[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?:date|dated)\s*[:\s]+(\w+\s+\d{1,2},?\s+\d{4})",
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return True, f"Date found: {match.group(1)}"
        
        return False, "No report date found"
    
    def _check_author_info(self, text: str) -> tuple:
        """Check for author or company information"""
        patterns = [
            r"(?:prepared\s*by|author|company|firm)\s*[:\s]+([^\n]+)",
            r"(?:limited|ltd|llc|engineers|consultants)",
            r"(?:P\.?E\.?|C\.?Eng|FICE|MICE)",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "Author/company information found"
        
        return False, "No author or company information found"


def run_rule_based_validation(report_text: str) -> List[CheckResult]:
    """
    Convenience function to run rule-based validation.
    
    Args:
        report_text: Report text to validate
        
    Returns:
        List of CheckResult objects
    """
    validator = RuleBasedValidator()
    return validator.validate(report_text)
