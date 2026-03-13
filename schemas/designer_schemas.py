"""
Pydantic schemas for designer agent data extraction and validation.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SoilLayerSchema(BaseModel):
    """Schema for soil layer data extraction."""
    layer_number: Optional[int] = Field(default=None, description="Layer number")
    description: Optional[str] = Field(default=None, description="Soil description")
    cohesion_kpa: Optional[float] = Field(default=None, description="Cohesion in kPa")
    friction_angle_deg: Optional[float] = Field(default=None, description="Friction angle in degrees")
    unit_weight_kn_m3: Optional[float] = Field(default=None, description="Unit weight in kN/m³")
    depth_from_m: Optional[float] = Field(default=None, description="Depth from surface in meters")
    depth_to_m: Optional[float] = Field(default=None, description="Depth to in meters")


class GeometrySchema(BaseModel):
    """Schema for foundation/project geometry."""
    width_m: Optional[float] = Field(default=None, description="Width in meters")
    length_m: Optional[float] = Field(default=None, description="Length in meters")
    depth_m: Optional[float] = Field(default=None, description="Depth in meters")
    height_m: Optional[float] = Field(default=None, description="Height in meters")
    slope_angle_deg: Optional[float] = Field(default=None, description="Slope angle in degrees")
    area_m2: Optional[float] = Field(default=None, description="Area in square meters")


class LoadsSchema(BaseModel):
    """Schema for load data extraction."""
    permanent_load_kn: Optional[float] = Field(default=None, description="Permanent/dead load in kN")
    variable_load_kn: Optional[float] = Field(default=None, description="Variable/imposed load in kN")
    horizontal_load_kn: Optional[float] = Field(default=None, description="Horizontal load in kN")
    moment_knm: Optional[float] = Field(default=None, description="Moment in kN·m")
    surcharge_kpa: Optional[float] = Field(default=None, description="Surcharge pressure in kPa")


class ProjectExtractionSchema(BaseModel):
    """Schema for extracting project information from user input."""
    project_name: Optional[str] = Field(default="", description="Project name")
    project_type: Optional[str] = Field(default="", description="Project type (building, slope, retaining_wall, excavation)")
    location: Optional[str] = Field(default="", description="Project location")
    description: Optional[str] = Field(default="", description="Project description")
    applicable_codes: List[str] = Field(default_factory=list, description="List of applicable codes/standards")
    foundation_type: Optional[str] = Field(default="", description="Foundation type (shallow, deep, pile, raft)")
    soil_layers: List[SoilLayerSchema] = Field(default_factory=list, description="Soil layer information")
    gwl_depth_m: Optional[float] = Field(default=None, description="Groundwater level depth in meters")
    geometry: GeometrySchema = Field(default_factory=GeometrySchema, description="Project geometry")
    loads: LoadsSchema = Field(default_factory=LoadsSchema, description="Load information")
    missing_critical_info: List[str] = Field(default_factory=list, description="List of missing critical information")


class ParameterExtractionSchema(BaseModel):
    """Schema for extracting numeric parameters from user responses."""
    cohesion_kpa: Optional[float] = Field(default=None, description="Cohesion in kPa")
    friction_angle_deg: Optional[float] = Field(default=None, description="Friction angle in degrees")
    unit_weight_kn_m3: Optional[float] = Field(default=None, description="Unit weight in kN/m³")
    gwl_depth_m: Optional[float] = Field(default=None, description="Groundwater depth in meters")
    foundation_width: Optional[float] = Field(default=None, description="Foundation width in meters")
    foundation_length: Optional[float] = Field(default=None, description="Foundation length in meters")
    foundation_depth: Optional[float] = Field(default=None, description="Foundation depth in meters")
    permanent_load_kn: Optional[float] = Field(default=None, description="Permanent load in kN")
    variable_load_kn: Optional[float] = Field(default=None, description="Variable load in kN")
    slope_angle_deg: Optional[float] = Field(default=None, description="Slope angle in degrees")
    depth_to_slip_m: Optional[float] = Field(default=None, description="Depth to slip surface in meters")
    wall_height_m: Optional[float] = Field(default=None, description="Wall height in meters")
    backfill_friction_angle_deg: Optional[float] = Field(default=None, description="Backfill friction angle")
    backfill_unit_weight: Optional[float] = Field(default=None, description="Backfill unit weight")
    surcharge_kpa: Optional[float] = Field(default=None, description="Surcharge pressure in kPa")


class DesignResultSchema(BaseModel):
    """Schema for design calculation results."""
    skill_id: str = Field(..., description="Skill identifier")
    success: bool = Field(..., description="Whether calculation succeeded")
    results: Dict[str, Any] = Field(default_factory=dict, description="Calculation results")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    acceptance_criteria_met: bool = Field(default=True, description="Whether acceptance criteria are met")
    notes: List[str] = Field(default_factory=list, description="Additional notes")


class SectionGenerationSchema(BaseModel):
    """Schema for report section generation validation."""
    section_number: str = Field(..., description="Section number")
    section_title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content in markdown")
    subsections_included: List[str] = Field(default_factory=list, description="Subsections included")
    data_sources: List[str] = Field(default_factory=list, description="Data sources referenced")
    warnings: List[str] = Field(default_factory=list, description="Warnings about missing data")


class CompleteReportSchema(BaseModel):
    """Schema for complete report structure."""
    project_name: str = Field(..., description="Project name")
    report_date: str = Field(default="", description="Report date")
    sections: List[SectionGenerationSchema] = Field(default_factory=list, description="Report sections")
    overall_status: str = Field(default="draft", description="Report status")
    missing_data_warnings: List[str] = Field(default_factory=list, description="Missing data warnings")
