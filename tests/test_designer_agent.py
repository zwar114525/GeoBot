"""
Tests for Designer Agent functionality.
Tests project info extraction, parameter collection, and report generation.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.agents.designer_agent import DesignerAgent, AgentState
from src.schemas.designer_schemas import ProjectExtractionSchema


class TestDesignerAgentInitialization:
    """Tests for DesignerAgent initialization."""
    
    def test_agent_initialization(self):
        """Test agent initializes with correct state."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            agent = DesignerAgent(use_local_db=False)
            assert agent.state == AgentState.INIT
            assert agent.catalog is not None
            assert agent.executor is not None
    
    def test_agent_start(self):
        """Test agent start method."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            agent = DesignerAgent()
            result = agent.start()
            
            assert agent.state == AgentState.COLLECTING_PROJECT_INFO
            assert "message" in result
            assert result["questions"] == []


class TestProjectInfoExtraction:
    """Tests for project information extraction."""
    
    def test_extract_complete_project_info(self):
        """Test extraction of complete project information."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
{
    "project_name": "Test Building",
    "project_type": "building",
    "location": "Kowloon",
    "description": "5-story residential",
    "applicable_codes": ["HK CoP 2017"],
    "foundation_type": "shallow",
    "soil_layers": [
        {"description": "Fill", "cohesion_kpa": 5, "friction_angle_deg": 28}
    ],
    "geometry": {"width_m": 10, "length_m": 15, "depth_m": 1.5},
    "loads": {"permanent_load_kn": 5000},
    "missing_critical_info": []
}
```'''
                
                agent = DesignerAgent()
                result = agent._handle_project_info("5-story building in Kowloon")
                
                assert agent.state in [AgentState.IDENTIFYING_SKILLS, AgentState.COLLECTING_PARAMETERS]
                assert agent.project.project_name == "Test Building"
                assert agent.project.project_type == "building"
    
    def test_extract_with_missing_info(self):
        """Test extraction identifies missing information."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
{
    "project_name": "Test Slope",
    "project_type": "slope",
    "location": "HK",
    "missing_critical_info": ["slope angle", "soil parameters"]
}
```'''
                
                agent = DesignerAgent()
                result = agent._handle_project_info("Slope project in HK")
                
                assert agent.state == AgentState.COLLECTING_PARAMETERS
                assert len(result.get("questions", [])) > 0
    
    def test_extract_handles_invalid_json(self):
        """Test extraction handles invalid JSON gracefully."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = "This is not valid JSON at all"
                
                agent = DesignerAgent()
                result = agent._handle_project_info("Some input")
                
                # Should return gracefully with default message
                assert "message" in result
                assert agent.state == AgentState.COLLECTING_PROJECT_INFO
    
    def test_extract_handles_empty_response(self):
        """Test extraction handles empty LLM response."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = ""
                
                agent = DesignerAgent()
                result = agent._handle_project_info("Some input")
                
                assert "message" in result


class TestParameterExtraction:
    """Tests for parameter extraction from user responses."""
    
    def test_extract_soil_parameters(self):
        """Test extraction of soil parameters."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
{
    "cohesion_kpa": 10,
    "friction_angle_deg": 32,
    "unit_weight_kn_m3": 19
}
```'''
                
                agent = DesignerAgent()
                result = agent._handle_parameter_input("Soil has cohesion 10 kPa, friction angle 32 degrees")
                
                assert agent.project.design_parameters.get("cohesion_kpa") == 10
                assert agent.project.design_parameters.get("friction_angle_deg") == 32
    
    def test_extract_partial_parameters(self):
        """Test extraction when only some parameters are provided."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = '''```json
{
    "cohesion_kpa": 5,
    "friction_angle_deg": 28
}
```'''
                
                agent = DesignerAgent()
                result = agent._handle_parameter_input("Cohesion is 5 kPa")
                
                assert agent.project.design_parameters.get("cohesion_kpa") == 5
                assert agent.project.design_parameters.get("unit_weight_kn_m3") is None
    
    def test_extract_handles_invalid_response(self):
        """Test extraction handles invalid response - goes back to ask for more params."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = "Invalid response"
                
                agent = DesignerAgent()
                result = agent._handle_parameter_input("Some input")
                
                # Should not crash - goes back to COLLECTING_PARAMETERS since no params extracted
                assert agent.state == AgentState.COLLECTING_PARAMETERS
                assert "questions" in result


class TestSkillIdentification:
    """Tests for skill identification based on project type."""
    
    def test_identify_building_skills(self):
        """Test skill identification for building project."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = "Mock report content"
                
                agent = DesignerAgent()
                agent.project.project_type = "building"
                agent.project.design_parameters = {
                    "soil_cohesion_kpa": 10,
                    "soil_friction_angle_deg": 32,
                    "soil_unit_weight": 19,
                    "foundation_width": 2,
                    "foundation_length": 2,
                    "foundation_depth": 1.5,
                }
                
                result = agent._identify_required_skills()
                
                assert "shallow_bearing_capacity" in agent.selected_skills
    
    def test_identify_slope_skills(self):
        """Test skill identification for slope project."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = "Mock report content"
                
                agent = DesignerAgent()
                agent.project.project_type = "slope"
                agent.project.design_parameters = {
                    "slope_angle_deg": 30,
                    "friction_angle_deg": 35,
                    "cohesion_kpa": 5,
                    "unit_weight": 18,
                    "depth_to_slip_m": 2,
                }
                
                result = agent._identify_required_skills()
                
                assert "slope_stability_infinite" in agent.selected_skills


class TestClarifyingQuestions:
    """Tests for clarifying question generation."""
    
    def test_generate_questions_for_missing_info(self):
        """Test question generation for missing information."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            agent = DesignerAgent()
            questions = agent._generate_clarifying_questions(
                ["foundation width", "soil cohesion", "groundwater level"]
            )
            
            assert len(questions) == 3
            assert all("Please provide:" in q for q in questions)
    
    def test_generate_no_questions_when_complete(self):
        """Test no questions generated when info is complete."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            agent = DesignerAgent()
            questions = agent._generate_clarifying_questions([])
            
            assert questions == []


class TestParameterLookup:
    """Tests for parameter value lookup from project data."""
    
    def test_lookup_from_design_parameters(self):
        """Test lookup from design parameters."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            agent = DesignerAgent()
            agent.project.design_parameters = {"cohesion_kpa": 10}
            
            value = agent._find_parameter_value("cohesion_kpa")
            
            assert value == 10
    
    def test_lookup_from_geometry(self):
        """Test lookup from geometry."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            agent = DesignerAgent()
            agent.project.geometry = {"width_m": 5}
            
            value = agent._find_parameter_value("foundation_width")
            
            assert value == 5
    
    def test_lookup_from_soil_layers(self):
        """Test lookup from soil layers."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            agent = DesignerAgent()
            agent.project.soil_layers = [
                {"cohesion_kpa": 10, "friction_angle_deg": 32}
            ]
            
            value = agent._find_parameter_value("soil_cohesion_kpa")
            
            assert value == 10
    
    def test_lookup_returns_none_when_not_found(self):
        """Test lookup returns None when value not found."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            agent = DesignerAgent()
            agent.project.design_parameters = {}
            agent.project.geometry = {}
            agent.project.soil_layers = []
            
            value = agent._find_parameter_value("unknown_param")
            
            assert value is None
