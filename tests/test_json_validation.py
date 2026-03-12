"""
Tests for JSON validation utilities and schema parsing.
"""
import pytest
from pydantic import ValidationError
from src.utils.json_validator import (
    extract_json_from_response,
    parse_json_with_retry,
    safe_parse_json,
)
from src.schemas.designer_schemas import (
    ProjectExtractionSchema,
    ParameterExtractionSchema,
    SoilLayerSchema,
    GeometrySchema,
)
from src.schemas.validator_schemas import (
    SectionCheckSchema,
    CheckStatus,
    CheckSeverity,
    ConsistencyIssueSchema,
)


class TestExtractJsonFromResponse:
    """Tests for JSON extraction from LLM responses."""
    
    def test_plain_json(self):
        """Test extraction from plain JSON string."""
        response = '{"key": "value", "number": 42}'
        result = extract_json_from_response(response)
        assert result == '{"key": "value", "number": 42}'
    
    def test_json_with_markdown_block(self):
        """Test extraction from markdown code block."""
        response = '```json\n{"key": "value", "number": 42}\n```'
        result = extract_json_from_response(response)
        assert result == '{"key": "value", "number": 42}'
    
    def test_json_without_language_hint(self):
        """Test extraction from markdown block without language."""
        response = '```\n{"key": "value"}\n```'
        result = extract_json_from_response(response)
        assert result == '{"key": "value"}'
    
    def test_empty_response(self):
        """Test handling of empty response."""
        assert extract_json_from_response("") == ""
        assert extract_json_from_response("   ") == ""
    
    def test_response_with_prefix_text(self):
        """Test extraction when there's prefix text."""
        response = 'Here is the JSON:\n\n```json\n{"key": "value"}\n```'
        result = extract_json_from_response(response)
        assert '{"key": "value"}' in result


class TestParameterExtractionSchema:
    """Tests for parameter extraction schema validation."""
    
    def test_valid_parameter_extraction(self):
        """Test valid parameter extraction."""
        data = {
            "cohesion_kpa": 5.0,
            "friction_angle_deg": 30,
            "unit_weight_kn_m3": 18.5,
            "gwl_depth_m": 2.5,
        }
        schema = ParameterExtractionSchema(**data)
        assert schema.cohesion_kpa == 5.0
        assert schema.friction_angle_deg == 30
        assert schema.gwl_depth_m == 2.5
    
    def test_partial_parameters(self):
        """Test extraction with only some parameters."""
        data = {"cohesion_kpa": 10.0, "friction_angle_deg": 25}
        schema = ParameterExtractionSchema(**data)
        assert schema.cohesion_kpa == 10.0
        assert schema.unit_weight_kn_m3 is None
    
    def test_empty_parameters(self):
        """Test extraction with no parameters."""
        schema = ParameterExtractionSchema()
        assert schema.cohesion_kpa is None
        assert schema.friction_angle_deg is None


class TestProjectExtractionSchema:
    """Tests for project extraction schema validation."""
    
    def test_valid_project_extraction(self):
        """Test valid project extraction."""
        data = {
            "project_name": "Test Project",
            "project_type": "building",
            "location": "Hong Kong",
            "applicable_codes": ["HK CoP 2017", "EC7"],
            "soil_layers": [
                {
                    "description": "Fill",
                    "cohesion_kpa": 5,
                    "friction_angle_deg": 28,
                    "unit_weight_kn_m3": 18,
                }
            ],
            "geometry": {"width_m": 10, "length_m": 15, "depth_m": 1.5},
            "missing_critical_info": ["groundwater level"],
        }
        schema = ProjectExtractionSchema(**data)
        assert schema.project_name == "Test Project"
        assert len(schema.soil_layers) == 1
        assert schema.geometry.width_m == 10
    
    def test_project_with_nested_objects(self):
        """Test extraction with nested geometry and loads."""
        data = {
            "project_type": "retaining_wall",
            "geometry": {
                "height_m": 5,
                "width_m": 3,
            },
            "loads": {
                "surcharge_kpa": 10,
            },
        }
        schema = ProjectExtractionSchema(**data)
        assert schema.geometry.height_m == 5
        assert schema.loads.surcharge_kpa == 10


class TestSafeParseJson:
    """Tests for safe JSON parsing with fallback."""
    
    def test_valid_json_parsing(self):
        """Test parsing valid JSON."""
        response = '{"project_name": "Test", "project_type": "building"}'
        result = safe_parse_json(
            response=response,
            target_type=ProjectExtractionSchema,
            default_factory=lambda: ProjectExtractionSchema(),
            context="test",
        )
        assert result.project_name == "Test"
        assert result.project_type == "building"
    
    def test_invalid_json_fallback(self):
        """Test fallback on invalid JSON."""
        response = "This is not JSON at all"
        result = safe_parse_json(
            response=response,
            target_type=ProjectExtractionSchema,
            default_factory=lambda: ProjectExtractionSchema(
                missing_critical_info=["fallback triggered"]
            ),
            context="test",
        )
        assert "fallback triggered" in result.missing_critical_info
    
    def test_malformed_json_fallback(self):
        """Test fallback on malformed JSON."""
        response = '{"project_name": "Test", invalid}'
        result = safe_parse_json(
            response=response,
            target_type=ProjectExtractionSchema,
            default_factory=ProjectExtractionSchema,
            context="test",
        )
        assert result.project_name == ""  # Default value
    
    def test_json_with_markdown_fallback(self):
        """Test parsing JSON with markdown blocks."""
        response = '```json\n{"project_name": "Test", "project_type": "slope"}\n```'
        result = safe_parse_json(
            response=response,
            target_type=ProjectExtractionSchema,
            default_factory=ProjectExtractionSchema,
            context="test",
        )
        assert result.project_name == "Test"
        assert result.project_type == "slope"


class TestValidatorSchemas:
    """Tests for validator agent schemas."""
    
    def test_section_check_schema(self):
        """Test section check schema with valid CheckStatus."""
        data = {"section": "3.1", "status": "pass", "notes": "Complete"}
        schema = SectionCheckSchema(**data)
        assert schema.status == CheckStatus.PASS
    
    def test_section_check_absent(self):
        """Test section check with fail status (maps from 'absent')."""
        data = {"section": "5.2", "status": "fail", "notes": "Missing"}
        schema = SectionCheckSchema(**data)
        assert schema.status == CheckStatus.FAIL
    
    def test_consistency_issue_schema(self):
        """Test consistency issue schema."""
        data = {
            "issue": "Conflicting soil parameters",
            "severity": "high",
            "sections_involved": ["3.1", "4.2"],
        }
        schema = ConsistencyIssueSchema(**data)
        assert schema.severity == CheckSeverity.HIGH
        assert len(schema.sections_involved) == 2


class TestSchemaValidationErrors:
    """Tests for validation error handling."""
    
    def test_invalid_status_value(self):
        """Test that invalid status values are rejected."""
        with pytest.raises(ValidationError):
            SectionCheckSchema(
                section="1.0",
                status="invalid_status",  # Not a valid CheckStatus
            )
    
    def test_invalid_severity_value(self):
        """Test that invalid severity values are rejected."""
        with pytest.raises(ValidationError):
            ConsistencyIssueSchema(
                issue="Test",
                severity="extreme",  # Not a valid CheckSeverity
            )


class TestRealWorldLLMResponses:
    """Tests simulating real LLM response patterns."""
    
    def test_llm_response_with_explanation(self):
        """Test LLM response that includes explanation."""
        response = """Here's the extracted project information:

```json
{
    "project_name": "Foundation Design",
    "project_type": "building",
    "location": "Kowloon",
    "missing_critical_info": ["soil parameters"]
}
```

Let me know if you need more details!"""
        result = safe_parse_json(
            response=response,
            target_type=ProjectExtractionSchema,
            default_factory=lambda: ProjectExtractionSchema(missing_critical_info=["parse failed"]),
            context="LLM extraction",
        )
        # The extraction should work because extract_json_from_response handles markdown blocks
        assert result.project_name == "Foundation Design"
        assert result.location == "Kowloon"
    
    def test_llm_response_with_partial_data(self):
        """Test LLM response with partial data."""
        response = """```json
{
    "project_type": "slope",
    "geometry": {"slope_angle_deg": 35}
}
```"""
        result = safe_parse_json(
            response=response,
            target_type=ProjectExtractionSchema,
            default_factory=ProjectExtractionSchema,
            context="partial extraction",
        )
        assert result.project_type == "slope"
        assert result.geometry.slope_angle_deg == 35
    
    def test_llm_response_with_soil_layers(self):
        """Test LLM response with soil layer data."""
        response = """```json
{
    "project_name": "Test",
    "soil_layers": [
        {"description": "Fill", "cohesion_kpa": 5, "friction_angle_deg": 28},
        {"description": "CDV", "cohesion_kpa": 10, "friction_angle_deg": 32}
    ]
}
```"""
        result = safe_parse_json(
            response=response,
            target_type=ProjectExtractionSchema,
            default_factory=ProjectExtractionSchema,
            context="soil layers",
        )
        assert len(result.soil_layers) == 2
        assert result.soil_layers[0].description == "Fill"
        assert result.soil_layers[1].friction_angle_deg == 32
