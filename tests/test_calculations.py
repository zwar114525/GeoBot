from src.calculations.bearing_capacity import (
    SoilParameters,
    FoundationGeometry,
    GroundwaterCondition,
    terzaghi_bearing_capacity,
    hansen_bearing_capacity,
)
from src.calculations.slope_stability import infinite_slope_drained
from src.calculations.earth_pressure import rankine_earth_pressure


def test_bearing_capacity_positive():
    soil = SoilParameters(cohesion_kpa=5, friction_angle_deg=30, unit_weight_kn_m3=19)
    foundation = FoundationGeometry(width_m=2, length_m=2, depth_m=1.5)
    result = hansen_bearing_capacity(soil, foundation)
    assert result.q_ult_kpa > 0
    assert result.q_allowable_kpa > 0


def test_terzaghi_groundwater_case():
    soil = SoilParameters(cohesion_kpa=0, friction_angle_deg=28, unit_weight_kn_m3=18)
    foundation = FoundationGeometry(width_m=1.5, length_m=3, depth_m=1.2)
    gwl = GroundwaterCondition(gwl_depth_m=1.0)
    result = terzaghi_bearing_capacity(soil, foundation, gwl=gwl)
    assert result.q_ult_kpa > 0


def test_infinite_slope_result():
    result = infinite_slope_drained(
        slope_angle_deg=30,
        friction_angle_deg=35,
        cohesion_kpa=5,
        unit_weight_kn_m3=18,
        depth_to_slip_m=2,
    )
    assert result.factor_of_safety > 0


def test_rankine_output_keys():
    result = rankine_earth_pressure(
        wall_height_m=4,
        backfill_friction_angle_deg=30,
        backfill_unit_weight=18,
        surcharge_kpa=10,
    )
    assert "Ka" in result
    assert "total_active_force_kn_m" in result
