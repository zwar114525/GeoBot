import json
from enum import Enum
from dataclasses import dataclass, field
from jinja2 import Template
from src.vectordb.qdrant_store import GeoVectorStore
from src.utils.llm_client import call_llm
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
        extraction_prompt = (
            "Extract JSON with project_name, project_type, location, description, applicable_codes, "
            "foundation_type, soil_layers, gwl_depth_m, geometry, loads, missing_critical_info.\n"
            f"User description:\n{user_input}\nReturn ONLY valid JSON."
        )
        response = call_llm(
            extraction_prompt,
            system_prompt="You are a geotechnical data extraction assistant. Return only JSON.",
            temperature=0.1,
        )
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            extracted = json.loads(cleaned)
        except Exception:
            return {
                "state": self.state.value,
                "message": "Please provide project details in structured form: type, location, geometry, loads, soil, groundwater.",
                "questions": [],
            }
        self.project.project_name = extracted.get("project_name", "")
        self.project.project_type = extracted.get("project_type", "")
        self.project.location = extracted.get("location", "")
        self.project.description = extracted.get("description", "")
        self.project.applicable_codes = extracted.get("applicable_codes", [])
        self.project.foundation_type = extracted.get("foundation_type", "")
        self.project.soil_layers = extracted.get("soil_layers", [])
        self.project.gwl_depth_m = extracted.get("gwl_depth_m")
        self.project.geometry = extracted.get("geometry", {})
        self.project.loads = extracted.get("loads", {})
        questions = self._generate_clarifying_questions(extracted.get("missing_critical_info", []))
        if questions:
            self.pending_questions = questions
            self.state = AgentState.COLLECTING_PARAMETERS
            return {"state": self.state.value, "message": "I need additional information.", "questions": questions}
        self.state = AgentState.IDENTIFYING_SKILLS
        return self._identify_required_skills()

    def _generate_clarifying_questions(self, missing_items: list[str]) -> list[str]:
        if not missing_items:
            return []
        return [f"Please provide: {item}" for item in missing_items[:5]]

    def _handle_parameter_input(self, user_input: str) -> dict:
        extraction_prompt = (
            "Extract numeric parameters as JSON from this answer. "
            "Use keys like cohesion_kpa, friction_angle_deg, unit_weight_kn_m3, gwl_depth_m, "
            "foundation_width, foundation_length, foundation_depth, permanent_load_kn, variable_load_kn, "
            "slope_angle_deg, depth_to_slip_m.\n"
            f"Answer: {user_input}\nReturn ONLY valid JSON."
        )
        response = call_llm(extraction_prompt, system_prompt="Return only JSON.", temperature=0.1)
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            params = json.loads(cleaned)
        except Exception:
            params = {}
        self.project.design_parameters.update(params)
        if params.get("gwl_depth_m") is not None:
            self.project.gwl_depth_m = params["gwl_depth_m"]
        self.state = AgentState.IDENTIFYING_SKILLS
        return self._identify_required_skills()

    def _identify_required_skills(self) -> dict:
        type_skill_map = {
            "building": ["shallow_bearing_capacity"],
            "slope": ["slope_stability_infinite"],
            "retaining_wall": ["earth_pressure_rankine", "shallow_bearing_capacity"],
            "excavation": ["earth_pressure_rankine"],
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
