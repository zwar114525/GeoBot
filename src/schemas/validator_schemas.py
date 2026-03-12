"""
Pydantic schemas for validator agent checks and validation reports.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class CheckStatus(str, Enum):
    """Status of a validation check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_CHECKED = "not_checked"
    ERROR = "error"


class CheckSeverity(str, Enum):
    """Severity level of a validation issue."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CheckCategory(str, Enum):
    """Category of validation check."""
    COMPLETENESS = "completeness"
    PARAMETER = "parameter"
    COMPLIANCE = "compliance"
    CONSISTENCY = "consistency"
    EXTRACTION = "extraction"
    FORMAT = "format"


class SectionCheckSchema(BaseModel):
    """Schema for section completeness check."""
    section: str = Field(..., description="Section identifier")
    status: CheckStatus = Field(..., description="Check status")
    notes: Optional[str] = Field(default="", description="Additional notes")


class CompletenessCheckSchema(BaseModel):
    """Schema for completeness check results."""
    sections: List[SectionCheckSchema] = Field(default_factory=list, description="Section checks")
    overall_completeness: float = Field(default=0.0, description="Overall completeness percentage")
    missing_sections: List[str] = Field(default_factory=list, description="List of missing sections")
    inadequate_sections: List[str] = Field(default_factory=list, description="List of inadequate sections")


class FactorOfSafetySchema(BaseModel):
    """Schema for factor of safety values."""
    bearing: Optional[float] = Field(default=None, description="Bearing capacity FoS")
    sliding: Optional[float] = Field(default=None, description="Sliding FoS")
    overturning: Optional[float] = Field(default=None, description="Overturning FoS")
    slope: Optional[float] = Field(default=None, description="Slope stability FoS")
    overall: Optional[float] = Field(default=None, alias="global", description="Overall/global FoS")


class ParameterIssueSchema(BaseModel):
    """Schema for parameter-related issues."""
    parameter: str = Field(..., description="Parameter name")
    issue: str = Field(..., description="Description of the issue")
    expected: Optional[str] = Field(default=None, description="Expected value or range")
    actual: Optional[str] = Field(default=None, description="Actual value found")
    severity: CheckSeverity = Field(default=CheckSeverity.MEDIUM, description="Issue severity")


class ParameterCheckSchema(BaseModel):
    """Schema for parameter validation results."""
    factors_of_safety: FactorOfSafetySchema = Field(default_factory=FactorOfSafetySchema)
    issues_noticed: List[ParameterIssueSchema] = Field(default_factory=list)
    missing_parameters: List[str] = Field(default_factory=list)
    out_of_range_parameters: List[ParameterIssueSchema] = Field(default_factory=list)


class DesignMethodSchema(BaseModel):
    """Schema for design method information."""
    method_name: str = Field(..., description="Name of design method")
    code_reference: Optional[str] = Field(default=None, description="Referenced code/clause")
    description: Optional[str] = Field(default=None, description="Method description")
    assumptions: List[str] = Field(default_factory=list, description="Method assumptions")


class ReferencedCodeSchema(BaseModel):
    """Schema for code reference."""
    code_name: str = Field(..., description="Name of the code/standard")
    clause: Optional[str] = Field(default=None, description="Clause/section reference")
    year: Optional[str] = Field(default=None, description="Code year/version")
    relevance: str = Field(default="general", description="Relevance to design")


class CodeComplianceSchema(BaseModel):
    """Schema for code compliance check results."""
    design_methods: List[DesignMethodSchema] = Field(default_factory=list)
    referenced_codes: List[ReferencedCodeSchema] = Field(default_factory=list)
    compliance_issues: List[str] = Field(default_factory=list)
    missing_references: List[str] = Field(default_factory=list)


class ConsistencyIssueSchema(BaseModel):
    """Schema for consistency issues."""
    issue: str = Field(..., description="Description of the inconsistency")
    severity: CheckSeverity = Field(default=CheckSeverity.MEDIUM, description="Issue severity")
    sections_involved: List[str] = Field(default_factory=list, description="Sections involved")
    parameter_involved: Optional[str] = Field(default=None, description="Parameter involved if applicable")
    recommendation: Optional[str] = Field(default=None, description="Recommended action")


class ConsistencyCheckSchema(BaseModel):
    """Schema for consistency check results."""
    issues: List[ConsistencyIssueSchema] = Field(default_factory=list)
    overall_consistent: bool = Field(default=True, description="Whether document is overall consistent")
    critical_issues: List[ConsistencyIssueSchema] = Field(default_factory=list)
    warnings: List[ConsistencyIssueSchema] = Field(default_factory=list)


class ValidationCheckItemSchema(BaseModel):
    """Schema for individual validation check item."""
    category: CheckCategory = Field(..., description="Check category")
    item: str = Field(..., description="Check item description")
    status: CheckStatus = Field(..., description="Check status")
    details: Optional[str] = Field(default="", description="Detailed findings")
    reference: Optional[str] = Field(default="", description="Code/reference clause")
    severity: CheckSeverity = Field(default=CheckSeverity.MEDIUM, description="Issue severity")


class ValidationSummarySchema(BaseModel):
    """Schema for validation summary."""
    total_checks: int = Field(..., description="Total number of checks performed")
    pass_count: int = Field(..., description="Number of passed checks")
    fail_count: int = Field(..., description="Number of failed checks")
    warning_count: int = Field(..., description="Number of warnings")
    not_checked_count: int = Field(default=0, description="Number of not checked items")
    overall_status: str = Field(..., description="Overall validation status")
    critical_issues: int = Field(default=0, description="Number of critical issues")
    summary_text: str = Field(default="", description="Human-readable summary")


class FullValidationReportSchema(BaseModel):
    """Schema for complete validation report."""
    document_name: str = Field(..., description="Name of document reviewed")
    checks: List[ValidationCheckItemSchema] = Field(default_factory=list)
    summary: ValidationSummarySchema = Field(default_factory=lambda: ValidationSummarySchema(
        total_checks=0, pass_count=0, fail_count=0, warning_count=0, overall_status="unknown"
    ))
    overall_status: str = Field(default="unknown", description="Overall validation status")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
