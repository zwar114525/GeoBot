import json
from enum import Enum
from dataclasses import dataclass, field
from jinja2 import Template
from loguru import logger
from src.vectordb.qdrant_store import GeoVectorStore
from src.utils.llm_client import call_llm
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
    LoadsSchema,
)
from src.skills.catalog import SkillCatalog
from src.skills.executor import SkillExecutor
from src.templates.report_structure import STANDARD_REPORT_TEMPLATE
from config.settings import PRIMARY_MODEL


class AgentState(str, Enum):
    INIT = "init"
    COLLECTING_PROJECT_INFO = "collecting_project_info"
    IDENTIFYING_SKILLS = "identifying_skills"
    COLLECTING_PARAMETERS = "collecting_parameters"
    EXECUTING_CALCULATIONS = "executing_calculations"
    GENERATING_REPORT = "generating_report"
    COMPLETE = "complete"


@dataclass
class ProjectData:
    project_name: str = ""
    project_type: str = ""
    location: str = ""
    description: str = ""
    applicable_codes: list[str] = field(default_factory=list)
    soil_layers: list[dict] = field(default_factory=list)
    gwl_depth_m: float | None = None
    foundation_type: str = ""
    loads: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)
    design_parameters: dict = field(default_factory=dict)
    calculation_results: list[dict] = field(default_factory=list)
    retrieved_references: list[dict] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)


class DesignerAgent:
    def __init__(self, use_local_db: bool = False):
        self.store = GeoVectorStore(use_local=use_local_db)
        self.catalog = SkillCatalog()
        self.executor = SkillExecutor(self.catalog)
        self.state = AgentState.INIT
        self.project = ProjectData()
        self.selected_skills: list[str] = []
        self.pending_questions: list[str] = []

    def start(self, initial_description: str = "") -> dict:
        self.state = AgentState.COLLECTING_PROJECT_INFO
        self.project = ProjectData()
        if initial_description:
            return self.process_input(initial_description)
        return {
            "state": self.state.value,
            "message": "Describe your project type, location, geometry, loads, GI data, and design codes.",
            "questions": [],
        }

    def process_input(self, user_input: str) -> dict:
        self.project.conversation_history.append({"role": "user", "content": user_input})
        if self.state == AgentState.COLLECTING_PROJECT_INFO:
            return self._handle_project_info(user_input)
        if self.state == AgentState.COLLECTING_PARAMETERS:
            return self._handle_parameter_input(user_input)
        if self.state == AgentState.IDENTIFYING_SKILLS:
            return self._identify_required_skills()
        if self.state == AgentState.EXECUTING_CALCULATIONS:
            return self._execute_calculations()
        if self.state == AgentState.GENERATING_REPORT:
            return self._generate_report()
        return {"state": self.state.value, "message": "Session complete."}

    def _handle_project_info(self, user_input: str) -> dict:
        # Ensure we start in the correct state
        self.state = AgentState.COLLECTING_PROJECT_INFO
        
        extraction_prompt = (
            "Extract JSON with project_name, project_type, location, description, applicable_codes, "
            "foundation_type, soil_layers, gwl_depth_m, geometry, loads, missing_critical_info.\n"
            "For soil_layers, extract array of objects with: description, cohesion_kpa, friction_angle_deg, unit_weight_kn_m3.\n"
            "For geometry, extract object with: width_m, length_m, depth_m, height_m as applicable.\n"
            "For loads, extract object with: permanent_load_kn, variable_load_kn.\n"
            f"User description:\n{user_input}\n\nReturn ONLY valid JSON matching this structure."
        )
        
        try:
            response = call_llm(
                extraction_prompt,
                system_prompt="You are a geotechnical data extraction assistant. Return only valid JSON with all required fields.",
                temperature=0.1,
                max_tokens=2000,
            )
            
            # Use robust parsing with retry logic
            extracted = safe_parse_json(
                response=response,
                target_type=ProjectExtractionSchema,
                default_factory=lambda: ProjectExtractionSchema(
                    missing_critical_info=[
                        "foundation dimensions",
                        "soil parameters",
                        "loading information"
                    ]
                ),
                context="project information extraction",
            )
            
            # Track if we got valid extraction (non-default values)
            has_valid_data = bool(extracted.project_type or extracted.location or extracted.project_name)
            
            # If no valid data extracted and we got defaults, treat as failure
            if not has_valid_data:
                logger.warning("Project extraction returned mostly default values - treating as failure")
                return {
                    "state": self.state.value,
                    "message": "Please provide project details in structured form: type, location, geometry, loads, soil, groundwater.",
                    "questions": [],
                }
            
            # Update project data with validated fields
            self.project.project_name = extracted.project_name or ""
            self.project.project_type = extracted.project_type or ""
            self.project.location = extracted.location or ""
            self.project.description = extracted.description or ""
            self.project.applicable_codes = extracted.applicable_codes or []
            self.project.foundation_type = extracted.foundation_type or ""
            
            # Convert Pydantic models to dicts for internal use
            if extracted.soil_layers:
                self.project.soil_layers = [
                    layer.model_dump(exclude_none=True) 
                    for layer in extracted.soil_layers
                ]
            if extracted.geometry:
                self.project.geometry = extracted.geometry.model_dump(exclude_none=True)
            if extracted.loads:
                self.project.loads = extracted.loads.model_dump(exclude_none=True)
            
            self.project.gwl_depth_m = extracted.gwl_depth_m
            
            # Generate clarifying questions for missing info
            questions = self._generate_clarifying_questions(extracted.missing_critical_info or [])
            
            if questions:
                self.pending_questions = questions
                self.state = AgentState.COLLECTING_PARAMETERS
                return {"state": self.state.value, "message": "I need additional information.", "questions": questions}
            
            self.state = AgentState.IDENTIFYING_SKILLS
            return self._identify_required_skills()
            
        except Exception as e:
            logger.error(f"Project info extraction failed: {e}")
            return {
                "state": self.state.value,
                "message": "Please provide project details in structured form: type, location, geometry, loads, soil, groundwater.",
                "questions": [],
            }

    def _generate_clarifying_questions(self, missing_items: list[str]) -> list[str]:
        if not missing_items:
            return []
        return [f"Please provide: {item}" for item in missing_items[:5]]

    def _handle_parameter_input(self, user_input: str) -> dict:
        # Ensure we're in the correct state for processing
        self.state = AgentState.COLLECTING_PARAMETERS
        
        extraction_prompt = (
            "Extract numeric parameters as JSON from this answer. "
            "Use keys like: cohesion_kpa, friction_angle_deg, unit_weight_kn_m3, gwl_depth_m, "
            "foundation_width, foundation_length, foundation_depth, permanent_load_kn, variable_load_kn, "
            "slope_angle_deg, depth_to_slip_m, wall_height_m, backfill_friction_angle_deg, backfill_unit_weight, surcharge_kpa.\n"
            "Only include parameters that are explicitly mentioned or can be confidently inferred.\n"
            f"Answer: {user_input}\n\nReturn ONLY valid JSON with numeric values (no units in values)."
        )
        
        try:
            response = call_llm(
                extraction_prompt,
                system_prompt="Return only valid JSON with numeric parameter values. Use null for unknown values.",
                temperature=0.1,
                max_tokens=1000,
            )
            
            # Use robust parsing with retry logic
            params = safe_parse_json(
                response=response,
                target_type=ParameterExtractionSchema,
                default_factory=ParameterExtractionSchema,
                context="parameter extraction",
            )
            
            # Convert to dict and filter out None values
            params_dict = params.model_dump(exclude_none=True)
            
            if not params_dict:
                logger.warning("No parameters could be extracted from user input - continuing anyway")
            else:
                logger.info(f"Extracted parameters: {params_dict}")
                self.project.design_parameters.update(params_dict)
                if params.gwl_depth_m is not None:
                    self.project.gwl_depth_m = params.gwl_depth_m
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
        
        # Continue to next state regardless of extraction success
        self.state = AgentState.IDENTIFYING_SKILLS
        return self._identify_required_skills()

    def _identify_required_skills(self) -> dict:
        type_skill_map = {
            "building": ["shallow_bearing_capacity"],
            "slope": ["slope_stability_infinite"],
            "retaining_wall": ["retaining_wall_stability", "earth_pressure_rankine"],
            "excavation": ["earth_pressure_rankine"],
            "pile": ["pile_axial_capacity"],
            "deep_foundation": ["pile_axial_capacity"],
            "caisson": ["pile_axial_capacity"],
        }
        self.selected_skills = type_skill_map.get(self.project.project_type, ["shallow_bearing_capacity"])
        missing_questions = []
        for skill_id in self.selected_skills:
            skill = self.catalog.get_skill(skill_id)
            if not skill:
                continue
            for inp in skill.inputs:
                if inp.required and inp.name not in self.project.design_parameters:
                    value = self._find_parameter_value(inp.name)
                    if value is None:
                        missing_questions.append(f"{inp.name} ({inp.unit})")
                    else:
                        self.project.design_parameters[inp.name] = value
        if missing_questions:
            self.pending_questions = list(dict.fromkeys(missing_questions))[:8]
            self.state = AgentState.COLLECTING_PARAMETERS
            return {"state": self.state.value, "message": "I need these parameters:", "questions": self.pending_questions}
        self.state = AgentState.EXECUTING_CALCULATIONS
        return self._execute_calculations()

    def _find_parameter_value(self, param_name: str):
        if param_name in self.project.design_parameters:
            return self.project.design_parameters[param_name]
        geometry_map = {
            "foundation_width": self.project.geometry.get("width_m"),
            "foundation_length": self.project.geometry.get("length_m"),
            "foundation_depth": self.project.geometry.get("depth_m"),
            "wall_height_m": self.project.geometry.get("height_m"),
            "slope_angle_deg": self.project.geometry.get("slope_angle_deg"),
        }
        if param_name in geometry_map:
            return geometry_map[param_name]
        if self.project.soil_layers:
            soil = self.project.soil_layers[0]
            soil_map = {
                "soil_cohesion_kpa": soil.get("cohesion_kpa"),
                "soil_friction_angle_deg": soil.get("friction_angle_deg"),
                "soil_unit_weight": soil.get("unit_weight_kn_m3"),
                "cohesion_kpa": soil.get("cohesion_kpa"),
                "friction_angle_deg": soil.get("friction_angle_deg"),
                "unit_weight": soil.get("unit_weight_kn_m3"),
                "backfill_friction_angle_deg": soil.get("friction_angle_deg"),
                "backfill_unit_weight": soil.get("unit_weight_kn_m3"),
            }
            if param_name in soil_map:
                return soil_map[param_name]
        if param_name == "gwl_depth" and self.project.gwl_depth_m is not None:
            return self.project.gwl_depth_m
        return None

    def _execute_calculations(self) -> dict:
        results = []
        for skill_id in self.selected_skills:
            results.append({"skill_id": skill_id, "result": self.executor.execute(skill_id, self.project.design_parameters)})
        self.project.calculation_results = results
        self.state = AgentState.GENERATING_REPORT
        return self._generate_report()

    def _generate_report(self) -> dict:
        for skill_id in self.selected_skills:
            skill = self.catalog.get_skill(skill_id)
            if not skill:
                continue
            for ref in skill.code_references:
                chunks = self.store.search(query=f"{ref.description} {ref.clause}", top_k=3, document_type="code")
                self.project.retrieved_references.extend(chunks)
        report_sections = []
        for section_template in STANDARD_REPORT_TEMPLATE:
            report_sections.append(
                {
                    "number": section_template.number,
                    "title": section_template.title,
                    "content": self._generate_section(section_template),
                }
            )
        report_markdown = self._assemble_report(report_sections)
        self.state = AgentState.COMPLETE
        return {
            "state": self.state.value,
            "message": "Report generation complete.",
            "report": report_markdown,
            "calculation_results": self.project.calculation_results,
        }

    def _generate_section(self, template) -> str:
        project_context = json.dumps(
            {
                "project_name": self.project.project_name,
                "project_type": self.project.project_type,
                "location": self.project.location,
                "description": self.project.description,
                "soil_layers": self.project.soil_layers,
                "gwl_depth_m": self.project.gwl_depth_m,
                "geometry": self.project.geometry,
                "design_parameters": self.project.design_parameters,
            },
            indent=2,
        )
        calc_context = json.dumps(self.project.calculation_results, indent=2, default=str)
        prompt = (
            f"Write section {template.number} {template.title} in formal HK geotechnical style.\n"
            f"Guidance: {template.guidance}\n"
            f"Project data:\n{project_context}\n"
            f"Calculation results:\n{calc_context}\n"
            f"Subsections: {[s['title'] for s in template.subsections]}\n"
            "Use only provided data. Return markdown."
        )
        return call_llm(prompt, system_prompt="You are a senior geotechnical engineer writing report sections.", model=PRIMARY_MODEL, max_tokens=1600)

    def _assemble_report(self, sections: list[dict]) -> str:
        header_template = Template(
            "# GEOTECHNICAL DESIGN REPORT\n"
            "## {{ project_name }}\n\n"
            "**Location:** {{ location }}\n"
            "**Date:** [Date]\n"
            "**Report No.:** [Report Number]\n"
            "**Revision:** A (Draft for Review)\n\n"
            "---\n\n"
            "## Table of Contents\n"
            "{% for s in sections %}- {{ s.number }} {{ s.title }}\n{% endfor %}\n"
            "---\n"
        )
        body_template = Template(
            "{% for s in sections %}\n## {{ s.number }} {{ s.title }}\n\n{{ s.content }}\n\n---\n{% endfor %}"
        )
        header = header_template.render(project_name=self.project.project_name or "Project", location=self.project.location or "N/A", sections=sections)
        body = body_template.render(sections=sections)
        return f"{header}\n{body}"
