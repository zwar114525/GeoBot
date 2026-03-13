import json
from src.skills.skill_models import Skill, SkillInput, SkillOutput, CodeReference, DesignCode, ReportSection


class SkillCatalog:
    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self._register_builtin_skills()

    def _register_builtin_skills(self):
        self.register(
            Skill(
                skill_id="shallow_bearing_capacity",
                name="Shallow Foundation Bearing Capacity Assessment",
                description="Calculate ultimate and allowable bearing capacity for shallow foundations.",
                category="foundations",
                inputs=[
                    SkillInput(name="soil_cohesion_kpa", type="float", unit="kPa", description="Soil cohesion"),
                    SkillInput(name="soil_friction_angle_deg", type="float", unit="degrees", description="Soil friction angle"),
                    SkillInput(name="soil_unit_weight", type="float", unit="kN/m³", description="Soil unit weight"),
                    SkillInput(name="foundation_width", type="float", unit="m", description="Foundation width"),
                    SkillInput(name="foundation_length", type="float", unit="m", description="Foundation length"),
                    SkillInput(name="foundation_depth", type="float", unit="m", description="Foundation depth"),
                    SkillInput(name="gwl_depth", type="float", unit="m", description="Groundwater depth", required=False, default=None),
                    SkillInput(name="factor_of_safety", type="float", unit="-", description="Factor of safety", required=False, default=3.0),
                    SkillInput(name="design_code", type="string", description="traditional or EC7_DA1", required=False, default="traditional"),
                ],
                outputs=[
                    SkillOutput(name="q_ult_kpa", type="float", unit="kPa", description="Ultimate bearing capacity"),
                    SkillOutput(name="q_allowable_kpa", type="float", unit="kPa", description="Allowable bearing capacity"),
                    SkillOutput(name="factor_of_safety", type="float", description="Applied safety factor"),
                    SkillOutput(name="method", type="string", description="Method"),
                    SkillOutput(name="pass_fail", type="bool", description="Pass status"),
                ],
                calculation_function="src.calculations.bearing_capacity.hansen_bearing_capacity",
                applicable_codes=[DesignCode.COP_FOUNDATIONS, DesignCode.GEOGUIDE_1, DesignCode.EC7],
                code_references=[
                    CodeReference(code=DesignCode.COP_FOUNDATIONS, clause="Section 2", description="Foundation design"),
                    CodeReference(code=DesignCode.EC7, clause="6.5.2", description="Bearing resistance"),
                ],
                report_sections=[ReportSection.ANALYSIS_AND_DESIGN, ReportSection.RECOMMENDATIONS],
                acceptance_criteria=["FoS >= 3.0", "EC7 utilisation <= 1.0"],
            )
        )
        self.register(
            Skill(
                skill_id="slope_stability_infinite",
                name="Infinite Slope Stability Analysis",
                description="Assess slope stability using infinite slope method.",
                category="slopes",
                inputs=[
                    SkillInput(name="slope_angle_deg", type="float", unit="degrees", description="Slope angle"),
                    SkillInput(name="friction_angle_deg", type="float", unit="degrees", description="Friction angle"),
                    SkillInput(name="cohesion_kpa", type="float", unit="kPa", description="Cohesion"),
                    SkillInput(name="unit_weight", type="float", unit="kN/m³", description="Unit weight"),
                    SkillInput(name="depth_to_slip_m", type="float", unit="m", description="Slip depth"),
                    SkillInput(name="gwl_above_slip_m", type="float", unit="m", required=False, default=0.0),
                    SkillInput(name="required_fos", type="float", required=False, default=1.4),
                ],
                outputs=[
                    SkillOutput(name="factor_of_safety", type="float", description="FoS"),
                    SkillOutput(name="pass_fail", type="bool", description="Pass status"),
                ],
                calculation_function="src.calculations.slope_stability.infinite_slope_drained",
                applicable_codes=[DesignCode.GEO_MANUAL_SLOPES, DesignCode.GEOGUIDE_1],
                code_references=[CodeReference(code=DesignCode.GEO_MANUAL_SLOPES, clause="Chapter 6", description="Slope stability")],
                report_sections=[ReportSection.ANALYSIS_AND_DESIGN, ReportSection.RECOMMENDATIONS],
                acceptance_criteria=["FoS >= 1.4"],
            )
        )
        self.register(
            Skill(
                skill_id="earth_pressure_rankine",
                name="Earth Pressure Calculation (Rankine)",
                description="Calculate active and passive earth pressures.",
                category="retaining_walls",
                inputs=[
                    SkillInput(name="wall_height_m", type="float", unit="m"),
                    SkillInput(name="backfill_friction_angle_deg", type="float", unit="degrees"),
                    SkillInput(name="backfill_cohesion_kpa", type="float", unit="kPa", required=False, default=0.0),
                    SkillInput(name="backfill_unit_weight", type="float", unit="kN/m³"),
                    SkillInput(name="surcharge_kpa", type="float", unit="kPa", required=False, default=0.0),
                ],
                outputs=[
                    SkillOutput(name="Ka", type="float"),
                    SkillOutput(name="Kp", type="float"),
                    SkillOutput(name="total_active_force_kn_m", type="float", unit="kN/m"),
                    SkillOutput(name="point_of_application_m", type="float", unit="m"),
                ],
                calculation_function="src.calculations.earth_pressure.rankine_earth_pressure",
                applicable_codes=[DesignCode.GEOGUIDE_1, DesignCode.BS8002, DesignCode.EC7],
                code_references=[CodeReference(code=DesignCode.GEOGUIDE_1, clause="Chapter 6", description="Earth pressures")],
                report_sections=[ReportSection.ANALYSIS_AND_DESIGN],
                acceptance_criteria=["Sliding FoS >= 1.5", "Overturning FoS >= 2.0"],
            )
        )
        # Pile Foundation Capacity
        self.register(
            Skill(
                skill_id="pile_axial_capacity",
                name="Pile Foundation Axial Capacity",
                description="Calculate ultimate and allowable axial capacity for driven/bored piles.",
                category="deep_foundations",
                inputs=[
                    SkillInput(name="pile_diameter_m", type="float", unit="m", description="Pile diameter"),
                    SkillInput(name="pile_length_m", type="float", unit="m", description="Pile total length"),
                    SkillInput(name="soil_cohesion_kpa", type="float", unit="kPa", description="Soil cohesion"),
                    SkillInput(name="soil_friction_angle_deg", type="float", unit="degrees", description="Soil friction angle"),
                    SkillInput(name="soil_unit_weight_kn_m3", type="float", unit="kN/m3", description="Soil unit weight"),
                    SkillInput(name="groundwater_depth_m", type="float", unit="m", description="Groundwater depth", required=False, default=None),
                    SkillInput(name="factor_of_safety", type="float", unit="-", description="Factor of safety", required=False, default=2.5),
                ],
                outputs=[
                    SkillOutput(name="ultimate_shaft_kN", type="float", unit="kN", description="Ultimate shaft friction"),
                    SkillOutput(name="ultimate_end_kN", type="float", unit="kN", description="Ultimate end bearing"),
                    SkillOutput(name="ultimate_total_kN", type="float", unit="kN", description="Total ultimate capacity"),
                    SkillOutput(name="allowable_total_kN", type="float", unit="kN", description="Allowable capacity"),
                    SkillOutput(name="factor_of_safety", type="float", description="Applied safety factor"),
                ],
                calculation_function="src.calculations.pile_capacity.pile_capacity",
                applicable_codes=[DesignCode.COP_FOUNDATIONS, DesignCode.GEOGUIDE_2, DesignCode.EC7],
                code_references=[
                    CodeReference(code=DesignCode.COP_FOUNDATIONS, clause="Section 8", description="Deep foundations"),
                    CodeReference(code=DesignCode.EC7, clause="7.6", description="Pile design"),
                ],
                report_sections=[ReportSection.ANALYSIS_AND_DESIGN, ReportSection.RECOMMENDATIONS],
                acceptance_criteria=["FoS >= 2.5", "Shaft + End bearing capacity verified"],
            )
        )
        # Retaining Wall Stability
        self.register(
            Skill(
                skill_id="retaining_wall_stability",
                name="Cantilever Retaining Wall Stability",
                description="Check stability of cantilever retaining walls (sliding, overturning, bearing).",
                category="retaining_walls",
                inputs=[
                    SkillInput(name="wall_height_m", type="float", unit="m", description="Wall height"),
                    SkillInput(name="base_width_m", type="float", unit="m", description="Total base width"),
                    SkillInput(name="toe_length_m", type="float", unit="m", description="Toe length", required=False, default=1.0),
                    SkillInput(name="stem_thickness_m", type="float", unit="m", description="Stem thickness at base", required=False, default=0.4),
                    SkillInput(name="base_thickness_m", type="float", unit="m", description="Base slab thickness", required=False, default=0.5),
                    SkillInput(name="backfill_unit_weight_kn_m3", type="float", unit="kN/m3", description="Backfill unit weight", required=False, default=18.0),
                    SkillInput(name="backfill_friction_angle_deg", type="float", unit="degrees", description="Backfill friction angle"),
                    SkillInput(name="backfill_surcharge_kpa", type="float", unit="kPa", description="Surcharge on backfill", required=False, default=0.0),
                    SkillInput(name="foundation_cohesion_kpa", type="float", unit="kPa", description="Foundation soil cohesion", required=False, default=50.0),
                    SkillInput(name="foundation_friction_angle_deg", type="float", unit="degrees", description="Foundation friction angle", required=False, default=25.0),
                ],
                outputs=[
                    SkillOutput(name="ka", type="float", description="Active pressure coefficient"),
                    SkillOutput(name="sliding_fos", type="float", description="Sliding FoS"),
                    SkillOutput(name="overturning_fos", type="float", description="Overturning FoS"),
                    SkillOutput(name="bearing_capacity_fos", type="float", description="Bearing capacity FoS"),
                    SkillOutput(name="max_bearing_pressure_kpa", type="float", unit="kPa", description="Maximum bearing pressure"),
                    SkillOutput(name="sliding_pass", type="bool", description="Sliding check pass"),
                    SkillOutput(name="overturning_pass", type="bool", description="Overturning check pass"),
                    SkillOutput(name="bearing_pass", type="bool", description="Bearing check pass"),
                ],
                calculation_function="src.calculations.retaining_wall.retaining_wall_stability",
                applicable_codes=[DesignCode.GEOGUIDE_1, DesignCode.BS8002, DesignCode.EC7],
                code_references=[
                    CodeReference(code=DesignCode.GEOGUIDE_1, clause="Chapter 7", description="Retaining wall design"),
                    CodeReference(code=DesignCode.BS8002, clause="Section 5", description="Wall stability"),
                ],
                report_sections=[ReportSection.ANALYSIS_AND_DESIGN, ReportSection.RECOMMENDATIONS],
                acceptance_criteria=["Sliding FoS >= 1.5", "Overturning FoS >= 2.0", "Bearing FoS >= 3.0"],
            )
        )

    def register(self, skill: Skill):
        self.skills[skill.skill_id] = skill

    def get_skill(self, skill_id: str) -> Skill | None:
        return self.skills.get(skill_id)

    def find_skills_for_category(self, category: str) -> list[Skill]:
        return [s for s in self.skills.values() if s.category == category]

    def find_skills_for_code(self, code: DesignCode) -> list[Skill]:
        return [s for s in self.skills.values() if code in s.applicable_codes]

    def find_skills_for_report_section(self, section: ReportSection) -> list[Skill]:
        return [s for s in self.skills.values() if section in s.report_sections]

    def get_missing_inputs(self, skill_id: str, provided_inputs: dict) -> list[SkillInput]:
        skill = self.get_skill(skill_id)
        if not skill:
            return []
        return [inp for inp in skill.inputs if inp.required and inp.name not in provided_inputs]

    def list_all(self) -> list[dict]:
        return [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "category": s.category,
                "codes": [c.value for c in s.applicable_codes],
                "inputs_required": len([i for i in s.inputs if i.required]),
            }
            for s in self.skills.values()
        ]

    def export_to_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({sid: s.model_dump() for sid, s in self.skills.items()}, f, indent=2, default=str)

    def import_from_json(self, filepath: str):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        for _, skill_data in data.items():
            self.register(Skill(**skill_data))
