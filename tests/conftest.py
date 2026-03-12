"""
Pytest fixtures for GeoBot tests.
Provides reusable test data, mock objects, and test utilities.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

from src.schemas.designer_schemas import (
    ProjectExtractionSchema,
    ParameterExtractionSchema,
    SoilLayerSchema,
    GeometrySchema,
    LoadsSchema,
)
from src.schemas.validator_schemas import (
    SectionCheckSchema,
    CheckStatus,
    CheckSeverity,
    ConsistencyIssueSchema,
    ParameterCheckSchema,
    FactorOfSafetySchema,
)


# ============== Test Data Fixtures ==============

@pytest.fixture
def sample_project_data():
    """Sample project data for testing."""
    return {
        "project_name": "Test Foundation Project",
        "project_type": "building",
        "location": "Kowloon, Hong Kong",
        "description": "5-story residential building with basement",
        "applicable_codes": ["HK CoP 2017", "EC7"],
        "foundation_type": "shallow",
        "soil_layers": [
            {
                "description": "Fill",
                "cohesion_kpa": 5.0,
                "friction_angle_deg": 28,
                "unit_weight_kn_m3": 18.0,
                "depth_from_m": 0.0,
                "depth_to_m": 2.0,
            },
            {
                "description": "CDV (Completely Decomposed Volcanic Rock)",
                "cohesion_kpa": 10.0,
                "friction_angle_deg": 32,
                "unit_weight_kn_m3": 19.0,
                "depth_from_m": 2.0,
                "depth_to_m": 10.0,
            },
        ],
        "gwl_depth_m": 3.5,
        "geometry": {
            "width_m": 10.0,
            "length_m": 15.0,
            "depth_m": 1.5,
        },
        "loads": {
            "permanent_load_kn": 5000.0,
            "variable_load_kn": 1500.0,
        },
    }


@pytest.fixture
def sample_soil_parameters():
    """Sample soil parameters for calculation tests."""
    return {
        "cohesion_kpa": 10.0,
        "friction_angle_deg": 32,
        "unit_weight_kn_m3": 19.0,
    }


@pytest.fixture
def sample_foundation_geometry():
    """Sample foundation geometry for calculation tests."""
    return {
        "width_m": 2.0,
        "length_m": 2.0,
        "depth_m": 1.5,
    }


@pytest.fixture
def sample_report_text():
    """Sample geotechnical report text for validation tests."""
    return """
# GEOTECHNICAL DESIGN REPORT
## Test Project

## 1. Introduction
This report presents the geotechnical design for a 5-story residential building.

## 2. Site Description
The site is located in Kowloon, Hong Kong.

## 3. Ground Investigation
Three boreholes were drilled to depths of 15-20m.

## 4. Geotechnical Conditions
### 4.1 Stratigraphy
- Fill: 0-2m depth
- CDV: 2-10m depth
- HDV: 10m+

### 4.2 Soil Parameters
- Cohesion: 10 kPa
- Friction angle: 32 degrees
- Unit weight: 19 kN/m³

## 5. Analysis and Design
### 5.1 Foundation Design
Shallow foundations at 1.5m depth.
Allowable bearing capacity: 300 kPa.
Factor of safety for bearing: 3.0
Factor of safety for sliding: 1.5

### 5.2 Load Calculations
Permanent load: 5000 kN
Variable load: 1500 kN

## 6. Recommendations
- Excavation support required
- Groundwater control measures
- Settlement monitoring

## 7. Construction Considerations
- Temporary works design required
- Inspection during construction

## 8. References
- HK Code of Practice for Foundations 2017
- Geoguide 1
"""


@pytest.fixture
def sample_chunk_metadata():
    """Sample chunk metadata for retrieval tests."""
    return {
        "document_id": "hk_cop_2017",
        "document_name": "HK Code of Practice for Foundations 2017",
        "document_type": "code",
        "clause_id": "6.1.5",
        "clause_title": "Bearing Capacity",
        "section_title": "Foundation Design",
        "page_no": 45,
        "content_type": "clause_text",
        "regulatory_strength": "mandatory",
    }


@pytest.fixture
def sample_retrieved_chunks():
    """Sample retrieved chunks for RAG tests."""
    return [
        {
            "text": "The ultimate bearing capacity shall be calculated using appropriate methods. A minimum factor of safety of 3.0 shall be applied.",
            "metadata": {
                "document_id": "hk_cop_2017",
                "document_name": "HK Code of Practice for Foundations 2017",
                "clause_id": "6.1.5",
                "section_title": "Bearing Capacity",
                "page_no": 45,
            },
            "score": 0.85,
        },
        {
            "text": "For shallow foundations, the depth should not be less than 1.0m below finished ground level.",
            "metadata": {
                "document_id": "hk_cop_2017",
                "document_name": "HK Code of Practice for Foundations 2017",
                "clause_id": "6.2.1",
                "section_title": "Foundation Depth",
                "page_no": 48,
            },
            "score": 0.72,
        },
    ]


# ============== Mock Fixtures ==============

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return Mock(
        choices=[
            Mock(
                message=Mock(
                    content='{"project_name": "Test", "project_type": "building"}'
                )
            )
        ]
    )


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = MagicMock()
    store.search.return_value = [
        {
            "text": "Sample retrieved text",
            "metadata": {"document_name": "Test Doc", "clause_id": "1.0", "page_no": 1},
            "score": 0.8,
        }
    ]
    store.list_documents.return_value = [
        {
            "document_id": "test_doc",
            "document_name": "Test Document",
            "document_type": "code",
            "chunk_count": 10,
        }
    ]
    return store


@pytest.fixture
def mock_skill_catalog():
    """Mock skill catalog for testing."""
    catalog = MagicMock()
    skill = MagicMock()
    skill.skill_id = "shallow_bearing_capacity"
    skill.name = "Bearing Capacity Assessment"
    skill.category = "foundations"
    skill.inputs = [
        MagicMock(name="soil_cohesion_kpa", required=True, unit="kPa"),
        MagicMock(name="soil_friction_angle_deg", required=True, unit="degrees"),
        MagicMock(name="foundation_width", required=True, unit="m"),
    ]
    catalog.get_skill.return_value = skill
    catalog.get_missing_inputs.return_value = []
    return catalog


# ============== Temp File Fixtures ==============

@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test PDF Content\nGeotechnical Report")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            doc.save(tmp.name)
            tmp_path = tmp.name
        doc.close()
        
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.fixture
def temp_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp.write("Test content for processing")
        tmp_path = tmp.name
    
    yield tmp_path
    
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


# ============== Schema Fixtures ==============

@pytest.fixture
def valid_project_schema():
    """Valid project extraction schema instance."""
    return ProjectExtractionSchema(
        project_name="Test Project",
        project_type="building",
        location="Hong Kong",
        description="Test description",
        applicable_codes=["HK CoP 2017"],
        foundation_type="shallow",
        soil_layers=[
            SoilLayerSchema(
                description="Fill",
                cohesion_kpa=5.0,
                friction_angle_deg=28,
                unit_weight_kn_m3=18.0,
            )
        ],
        geometry=GeometrySchema(width_m=10, length_m=15, depth_m=1.5),
        loads=LoadsSchema(permanent_load_kn=5000, variable_load_kn=1500),
        missing_critical_info=[],
    )


@pytest.fixture
def valid_parameter_schema():
    """Valid parameter extraction schema instance."""
    return ParameterExtractionSchema(
        cohesion_kpa=10.0,
        friction_angle_deg=32,
        unit_weight_kn_m3=19.0,
        gwl_depth_m=3.5,
        foundation_width=2.0,
        foundation_depth=1.5,
    )


@pytest.fixture
def valid_section_check():
    """Valid section check schema instance."""
    return SectionCheckSchema(
        section="3.1",
        status=CheckStatus.PASS,
        notes="Section is complete and adequate",
    )


@pytest.fixture
def valid_consistency_issue():
    """Valid consistency issue schema instance."""
    return ConsistencyIssueSchema(
        issue="Conflicting soil parameters between sections",
        severity=CheckSeverity.MEDIUM,
        sections_involved=["4.2", "5.1"],
        recommendation="Verify parameters with GI data",
    )


@pytest.fixture
def valid_parameter_check():
    """Valid parameter check schema instance."""
    return ParameterCheckSchema(
        factors_of_safety=FactorOfSafetySchema(
            bearing=3.0,
            sliding=1.5,
            overturning=2.0,
            slope=1.4,
        ),
        issues_noticed=[],
        missing_parameters=[],
    )
