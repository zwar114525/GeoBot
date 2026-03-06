from src.skills.catalog import SkillCatalog
from src.calculations.bearing_capacity import (
    SoilParameters,
    FoundationGeometry,
    GroundwaterCondition,
    hansen_bearing_capacity,
    ec7_bearing_capacity_da1,
)
from src.calculations.slope_stability import infinite_slope_drained
from src.calculations.earth_pressure import rankine_earth_pressure


class SkillExecutor:
    def __init__(self, catalog: SkillCatalog):
        self.catalog = catalog
        self._function_map = {
            "shallow_bearing_capacity": self._execute_bearing_capacity,
            "slope_stability_infinite": self._execute_slope_stability,
            "earth_pressure_rankine": self._execute_earth_pressure,
        }

    def execute(self, skill_id: str, inputs: dict) -> dict:
        skill = self.catalog.get_skill(skill_id)
        if not skill:
            return {"success": False, "error": f"Skill not found: {skill_id}"}
        missing = self.catalog.get_missing_inputs(skill_id, inputs)
        if missing:
            return {
                "success": False,
                "error": f"Missing required inputs: {[m.name for m in missing]}",
                "missing_inputs": [m.model_dump() for m in missing],
            }
        executor_fn = self._function_map.get(skill_id)
        if not executor_fn:
            return {"success": False, "error": f"No executor registered for skill: {skill_id}"}
        try:
            results = executor_fn(inputs)
            description = ""
            if skill.result_description_template:
                try:
                    description = skill.result_description_template.format(**{**inputs, **results})
                except Exception:
                    description = str(results)
            return {
                "success": True,
                "results": results,
                "skill": skill.model_dump(),
                "formatted_description": description,
                "acceptance_criteria": skill.acceptance_criteria,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_bearing_capacity(self, inputs: dict) -> dict:
        soil = SoilParameters(
            cohesion_kpa=inputs["soil_cohesion_kpa"],
            friction_angle_deg=inputs["soil_friction_angle_deg"],
            unit_weight_kn_m3=inputs["soil_unit_weight"],
        )
        foundation = FoundationGeometry(
            width_m=inputs["foundation_width"],
            length_m=inputs["foundation_length"],
            depth_m=inputs["foundation_depth"],
        )
        gwl = GroundwaterCondition(gwl_depth_m=inputs["gwl_depth"]) if inputs.get("gwl_depth") is not None else None
        fos = inputs.get("factor_of_safety", 3.0)
        code = inputs.get("design_code", "traditional")
        result = hansen_bearing_capacity(soil, foundation, gwl, fos)
        output = {
            "q_ult_kpa": result.q_ult_kpa,
            "q_allowable_kpa": result.q_allowable_kpa,
            "factor_of_safety": result.factor_of_safety,
            "method": result.method,
            "bearing_capacity_factors": result.bearing_capacity_factors,
            "shape_factors": result.shape_factors,
            "depth_factors": result.depth_factors,
            "notes": result.notes,
            "pass_fail": True,
        }
        if code == "EC7_DA1" and inputs.get("characteristic_permanent_load_kn") is not None:
            output["ec7_check"] = ec7_bearing_capacity_da1(
                soil=soil,
                foundation=foundation,
                gwl=gwl,
                characteristic_permanent_load_kn=inputs.get("characteristic_permanent_load_kn", 0),
                characteristic_variable_load_kn=inputs.get("characteristic_variable_load_kn", 0),
            )
        return output

    def _execute_slope_stability(self, inputs: dict) -> dict:
        result = infinite_slope_drained(
            slope_angle_deg=inputs["slope_angle_deg"],
            friction_angle_deg=inputs["friction_angle_deg"],
            cohesion_kpa=inputs["cohesion_kpa"],
            unit_weight_kn_m3=inputs["unit_weight"],
            depth_to_slip_m=inputs["depth_to_slip_m"],
            gwl_above_slip_m=inputs.get("gwl_above_slip_m", 0.0),
            required_fos=inputs.get("required_fos", 1.4),
        )
        return {
            "factor_of_safety": result.factor_of_safety,
            "method": result.method,
            "critical_slip_surface": result.critical_slip_surface,
            "pass_fail": result.pass_fail,
            "required_fos": result.required_fos,
            "notes": result.notes,
        }

    def _execute_earth_pressure(self, inputs: dict) -> dict:
        return rankine_earth_pressure(
            wall_height_m=inputs["wall_height_m"],
            backfill_friction_angle_deg=inputs["backfill_friction_angle_deg"],
            backfill_cohesion_kpa=inputs.get("backfill_cohesion_kpa", 0.0),
            backfill_unit_weight=inputs["backfill_unit_weight"],
            surcharge_kpa=inputs.get("surcharge_kpa", 0.0),
        )
