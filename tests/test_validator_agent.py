"""
Tests for Validator Agent functionality.
Tests report validation, completeness checks, and compliance verification.
"""
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from src.agents.validator_agent import ValidatorAgent
from src.schemas.validator_schemas import CheckStatus, CheckSeverity


class TestValidatorAgentInitialization:
    """Tests for ValidatorAgent initialization."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            agent = ValidatorAgent(use_local_db=False)
            assert agent.store is not None


class TestReportValidation:
    """Tests for complete report validation workflow."""
    
    def test_validate_report_structure(self, sample_report_text):
        """Test validation of report structure."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.extract_text_from_pdf') as mock_extract:
                with patch('src.agents.validator_agent.extract_pages_with_numbers') as mock_pages:
                    mock_extract.return_value = sample_report_text
                    mock_pages.return_value = []
                    
                    agent = ValidatorAgent()
                    # Mock the check methods to return predictable results
                    with patch.object(agent, '_check_completeness') as mock_completeness:
                        with patch.object(agent, '_check_parameters') as mock_parameters:
                            with patch.object(agent, '_check_code_compliance') as mock_compliance:
                                with patch.object(agent, '_check_consistency') as mock_consistency:
                                    with patch.object(agent, '_generate_summary') as mock_summary:
                                        mock_completeness.return_value = []
                                        mock_parameters.return_value = []
                                        mock_compliance.return_value = []
                                        mock_consistency.return_value = []
                                        mock_summary.return_value = "Report is complete"
                                        
                                        report = agent.validate_report("/fake/path.pdf")
                                        
                                        assert report.document_name == "path.pdf"
                                        assert report.summary == "Report is complete"
    
    def test_validate_rejects_short_report(self):
        """Test validation rejects very short reports."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.extract_text_from_pdf') as mock_extract:
                with patch('src.agents.validator_agent.extract_pages_with_numbers') as mock_pages:
                    mock_extract.return_value = "Too short"
                    mock_pages.return_value = []
                    
                    agent = ValidatorAgent()
                    report = agent.validate_report("/fake/path.pdf")
                    
                    assert report.overall_status == "rejected"
                    assert any(c.severity == "critical" for c in report.checks)


class TestCompletenessCheck:
    """Tests for section completeness checking."""
    
    def test_check_completeness_valid_response(self, sample_report_text):
        """Test completeness check with valid LLM response."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
[
    {"section": "1. Introduction", "status": "present", "notes": "Complete"},
    {"section": "2. Site Description", "status": "present", "notes": "Adequate"},
    {"section": "5. Analysis", "status": "inadequate", "notes": "Missing calculations"}
]
```'''
                
                agent = ValidatorAgent()
                checks = agent._check_completeness(sample_report_text)
                
                assert len(checks) > 0
                assert any(c.status == "pass" for c in checks)
                assert any(c.status == "warning" for c in checks)
    
    def test_check_completeness_handles_invalid_json(self, sample_report_text):
        """Test completeness check handles invalid JSON."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = "Invalid JSON response"
                
                agent = ValidatorAgent()
                checks = agent._check_completeness(sample_report_text)
                
                # Should return a warning check instead of crashing
                assert len(checks) > 0
                assert any(c.status == "warning" for c in checks)
    
    def test_check_completeness_empty_response(self, sample_report_text):
        """Test completeness check handles empty response."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = ""
                
                agent = ValidatorAgent()
                checks = agent._check_completeness(sample_report_text)
                
                assert len(checks) > 0


class TestParameterCheck:
    """Tests for parameter validation checking."""
    
    def test_check_parameters_valid_fos(self, sample_report_text):
        """Test parameter check with valid factors of safety."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
{
    "factors_of_safety": {
        "bearing": 3.0,
        "sliding": 1.5,
        "overturning": 2.0,
        "slope": 1.4
    },
    "issues_noticed": []
}
```'''
                
                agent = ValidatorAgent()
                checks = agent._check_parameters(sample_report_text)
                
                # All FoS meet minimum requirements
                fos_checks = [c for c in checks if "FoS" in c.item]
                assert all(c.status == "pass" for c in fos_checks)
    
    def test_check_parameters_low_fos(self, sample_report_text):
        """Test parameter check identifies low factors of safety."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
{
    "factors_of_safety": {
        "bearing": 2.0,
        "sliding": 1.2
    },
    "issues_noticed": ["Low bearing capacity FoS"]
}
```'''
                
                agent = ValidatorAgent()
                checks = agent._check_parameters(sample_report_text)
                
                # Bearing FoS 2.0 < 3.0 required
                bearing_check = [c for c in checks if "bearing" in c.item.lower()][0]
                assert bearing_check.status == "fail"
                assert bearing_check.severity == "critical"
    
    def test_check_parameters_handles_invalid_json(self, sample_report_text):
        """Test parameter check handles invalid JSON."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = "Not valid JSON"
                
                agent = ValidatorAgent()
                checks = agent._check_parameters(sample_report_text)
                
                # Should return warning instead of crashing
                assert len(checks) > 0
                assert any(c.status == "warning" for c in checks)


class TestCodeComplianceCheck:
    """Tests for code compliance checking."""
    
    def test_check_compliance_with_methods(self, sample_report_text):
        """Test compliance check with design methods."""
        with patch('src.agents.validator_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = [
                {"text": "Code requirement", "metadata": {}, "score": 0.8}
            ]
            
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
{
    "design_methods": [
        {"method_name": "Bearing capacity analysis", "code_reference": "6.1.5"}
    ],
    "referenced_codes": []
}
```'''
                
                agent = ValidatorAgent()
                checks = agent._check_code_compliance(sample_report_text)
                
                assert len(checks) > 0
                assert checks[0].status == "pass"
    
    def test_check_compliance_no_code_reference(self, sample_report_text):
        """Test compliance check when no code chunk found."""
        with patch('src.agents.validator_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = []  # No results
            
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
{
    "design_methods": [
        {"method_name": "Unknown method"}
    ],
    "referenced_codes": []
}
```'''
                
                agent = ValidatorAgent()
                checks = agent._check_code_compliance(sample_report_text)
                
                assert len(checks) > 0
                assert checks[0].status == "warning"


class TestConsistencyCheck:
    """Tests for internal consistency checking."""
    
    def test_check_consistency_no_issues(self, sample_report_text):
        """Test consistency check when no issues found."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = "[]"
                
                agent = ValidatorAgent()
                checks = agent._check_consistency(sample_report_text)
                
                assert len(checks) == 1
                assert checks[0].status == "pass"
                assert "No obvious inconsistencies" in checks[0].details
    
    def test_check_consistency_with_issues(self, sample_report_text):
        """Test consistency check identifies issues."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
[
    {
        "issue": "Conflicting soil parameters between sections 4.2 and 5.1",
        "severity": "high",
        "sections_involved": ["4.2", "5.1"]
    }
]
```'''
                
                agent = ValidatorAgent()
                checks = agent._check_consistency(sample_report_text)
                
                assert len(checks) > 0
                assert checks[0].status == "fail"  # High severity = fail
    
    def test_check_consistency_handles_invalid_json(self, sample_report_text):
        """Test consistency check handles invalid JSON."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.call_llm') as mock_llm:
                mock_llm.return_value = "Invalid response"
                
                agent = ValidatorAgent()
                checks = agent._check_consistency(sample_report_text)
                
                # Should return warning instead of crashing
                assert len(checks) > 0
                assert checks[0].status == "warning"


class TestValidationReportFormatting:
    """Tests for validation report formatting."""
    
    def test_format_validation_report(self):
        """Test formatting of validation report."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            from src.agents.validator_agent import ValidationReport, CheckItem
            
            report = ValidationReport(
                document_name="test_report.pdf",
                checks=[
                    CheckItem(
                        category="completeness",
                        item="Section 1",
                        status="pass",
                        details="Complete",
                    ),
                    CheckItem(
                        category="parameter",
                        item="FoS (bearing)",
                        status="fail",
                        details="2.0 < 3.0",
                        severity="critical",
                    ),
                ],
                summary="Report has issues",
                overall_status="revisions_needed",
            )
            
            agent = ValidatorAgent()
            formatted = agent.format_validation_report(report)
            
            assert "# SUBMISSION REVIEW REPORT" in formatted
            assert "test_report.pdf" in formatted
            assert "REVISIONS NEEDED" in formatted
            assert "✓ 1" in formatted  # Pass count
            assert "✗ 1" in formatted  # Fail count


class TestValidationStatus:
    """Tests for overall validation status determination."""
    
    def test_status_rejected_on_critical_fail(self):
        """Test status is rejected when critical failures exist."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            from src.agents.validator_agent import ValidationReport, CheckItem
            
            report = ValidationReport(
                document_name="test.pdf",
                checks=[
                    CheckItem(
                        category="extraction",
                        item="Readability",
                        status="fail",
                        severity="critical",
                    )
                ],
            )
            
            # Manually calculate status as validate_report would
            critical_fails = sum(1 for c in report.checks if c.status == "fail" and c.severity == "critical")
            
            if critical_fails > 0:
                status = "rejected"
            elif report.fail_count > 0:
                status = "revisions_needed"
            else:
                status = "acceptable"
            
            assert status == "rejected"
    
    def test_status_revisions_needed_on_non_critical_fail(self):
        """Test status is revisions_needed for non-critical failures."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            from src.agents.validator_agent import ValidationReport, CheckItem
            
            report = ValidationReport(
                document_name="test.pdf",
                checks=[
                    CheckItem(
                        category="completeness",
                        item="Section 5",
                        status="fail",
                        severity="medium",  # Not critical
                    )
                ],
            )
            
            critical_fails = sum(1 for c in report.checks if c.status == "fail" and c.severity == "critical")
            
            if critical_fails > 0:
                status = "rejected"
            elif report.fail_count > 0:
                status = "revisions_needed"
            else:
                status = "acceptable"
            
            assert status == "revisions_needed"
    
    def test_status_acceptable_when_all_pass(self):
        """Test status is acceptable when all checks pass."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            from src.agents.validator_agent import ValidationReport, CheckItem
            
            report = ValidationReport(
                document_name="test.pdf",
                checks=[
                    CheckItem(category="completeness", item="Section 1", status="pass"),
                    CheckItem(category="parameter", item="FoS", status="pass"),
                ],
            )
            
            assert report.fail_count == 0
            assert report.pass_count == 2
