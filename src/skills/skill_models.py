from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel


class DesignCode(str, Enum):
    COP_FOUNDATIONS = "cop_foundations_2017"
    GEOGUIDE_1 = "geoguide_1"
    GEOGUIDE_2 = "geoguide_2"
    GEOGUIDE_7 = "geoguide_7"
    GEO_MANUAL_SLOPES = "geo_manual_slopes"
    EC7 = "bs_en_1997_1"
    BS8002 = "bs_8002"
    BS8004 = "bs_8004"


class ReportSection(str, Enum):
    INTRODUCTION = "1_introduction"
    SITE_DESCRIPTION = "2_site_description"
    GROUND_INVESTIGATION = "3_ground_investigation"
    GEOTECHNICAL_CONDITIONS = "4_geotechnical_conditions"
    ANALYSIS_AND_DESIGN = "5_analysis_and_design"
    RECOMMENDATIONS = "6_recommendations"
    CONSTRUCTION_CONSIDERATIONS = "7_construction_considerations"
    INSTRUMENTATION_MONITORING = "8_instrumentation_monitoring"
    REFERENCES = "9_references"
    APPENDICES = "10_appendices"


class CodeReference(BaseModel):
    code: DesignCode
    clause: str
    description: str = ""


class SkillInput(BaseModel):
    name: str
    type: str
    unit: str = ""
    description: str = ""
    required: bool = True
    default: Any = None
    valid_range: Optional[dict] = None


class SkillOutput(BaseModel):
    name: str
    type: str
    unit: str = ""
    description: str = ""


class Skill(BaseModel):
    skill_id: str
    name: str
    description: str
    category: str
    inputs: list[SkillInput]
    outputs: list[SkillOutput]
    calculation_function: str
    applicable_codes: list[DesignCode]
    code_references: list[CodeReference]
    report_sections: list[ReportSection]
    acceptance_criteria: list[str] = []
    result_description_template: str = ""
    notes: list[str] = []
