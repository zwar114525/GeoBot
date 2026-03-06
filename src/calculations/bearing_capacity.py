import math
from dataclasses import dataclass


@dataclass
class SoilParameters:
    cohesion_kpa: float
    friction_angle_deg: float
    unit_weight_kn_m3: float
    is_undrained: bool = False


@dataclass
class FoundationGeometry:
    width_m: float
    length_m: float
    depth_m: float


@dataclass
class GroundwaterCondition:
    gwl_depth_m: float
    water_unit_weight_kn_m3: float = 9.81


@dataclass
class BearingCapacityResult:
    q_ult_kpa: float
    q_allowable_kpa: float
    factor_of_safety: float
    method: str
    bearing_capacity_factors: dict
    shape_factors: dict
    depth_factors: dict
    notes: list[str]


def terzaghi_bearing_capacity(
    soil: SoilParameters,
    foundation: FoundationGeometry,
    gwl: GroundwaterCondition | None = None,
    fos: float = 3.0,
) -> BearingCapacityResult:
    phi_rad = math.radians(soil.friction_angle_deg)
    if soil.friction_angle_deg == 0:
        Nc, Nq, Ngamma = 5.14, 1.0, 0.0
    else:
        Nq = math.exp(math.pi * math.tan(phi_rad)) * (math.tan(math.radians(45 + soil.friction_angle_deg / 2)) ** 2)
        Nc = (Nq - 1) / math.tan(phi_rad)
        Ngamma = 2 * (Nq + 1) * math.tan(phi_rad)

    B, L = foundation.width_m, foundation.length_m
    if L / B > 10:
        sc, sgamma = 1.0, 1.0
    elif abs(L - B) < 0.01:
        sc, sgamma = 1.3, 0.8
    else:
        sc = 1.0 + 0.3 * (B / L)
        sgamma = 1.0 - 0.2 * (B / L)
    sq = 1.0
    gamma_below = soil.unit_weight_kn_m3
    notes = []
    if gwl is not None and gwl.gwl_depth_m <= foundation.depth_m:
        gamma_eff_above = soil.unit_weight_kn_m3 - gwl.water_unit_weight_kn_m3
        q = gamma_eff_above * foundation.depth_m
        gamma_below = soil.unit_weight_kn_m3 - gwl.water_unit_weight_kn_m3
        notes.append("Groundwater correction applied")
    else:
        q = soil.unit_weight_kn_m3 * foundation.depth_m
        notes.append("No groundwater correction applied")
    q_ult = soil.cohesion_kpa * Nc * sc + q * Nq * sq + 0.5 * gamma_below * B * Ngamma * sgamma
    q_allowable = q_ult / fos
    return BearingCapacityResult(
        q_ult_kpa=round(q_ult, 1),
        q_allowable_kpa=round(q_allowable, 1),
        factor_of_safety=fos,
        method="Terzaghi",
        bearing_capacity_factors={"Nc": round(Nc, 2), "Nq": round(Nq, 2), "Nγ": round(Ngamma, 2)},
        shape_factors={"sc": round(sc, 2), "sq": round(sq, 2), "sγ": round(sgamma, 2)},
        depth_factors={"dc": 1.0, "dq": 1.0, "dγ": 1.0},
        notes=notes,
    )


def hansen_bearing_capacity(
    soil: SoilParameters,
    foundation: FoundationGeometry,
    gwl: GroundwaterCondition | None = None,
    fos: float = 3.0,
) -> BearingCapacityResult:
    phi_rad = math.radians(soil.friction_angle_deg)
    B, L, Df = foundation.width_m, foundation.length_m, foundation.depth_m
    if soil.friction_angle_deg == 0:
        Nc, Nq, Ngamma = 5.14, 1.0, 0.0
    else:
        Nq = math.exp(math.pi * math.tan(phi_rad)) * (math.tan(math.radians(45 + soil.friction_angle_deg / 2)) ** 2)
        Nc = (Nq - 1) / math.tan(phi_rad)
        Ngamma = 1.5 * (Nq - 1) * math.tan(phi_rad)
    sc = 1.0 + (Nq / Nc) * (B / L) if soil.friction_angle_deg > 0 else 1.0 + 0.2 * (B / L)
    sq = 1.0 + (B / L) * math.tan(phi_rad)
    sgamma = max(1.0 - 0.4 * (B / L), 0.6)
    k = Df / B if Df / B <= 1 else math.atan(Df / B)
    dc = 1.0 + 0.4 * k
    dq = 1.0 + 2 * math.tan(phi_rad) * (1 - math.sin(phi_rad)) ** 2 * k
    dgamma = 1.0
    gamma_below = soil.unit_weight_kn_m3
    notes = []
    if gwl is not None and gwl.gwl_depth_m <= Df:
        gamma_eff_above = soil.unit_weight_kn_m3 - gwl.water_unit_weight_kn_m3
        q = gamma_eff_above * Df
        gamma_below = soil.unit_weight_kn_m3 - gwl.water_unit_weight_kn_m3
        notes.append("Groundwater correction applied")
    else:
        q = soil.unit_weight_kn_m3 * Df
        notes.append("No groundwater correction applied")
    q_ult = soil.cohesion_kpa * Nc * sc * dc + q * Nq * sq * dq + 0.5 * gamma_below * B * Ngamma * sgamma * dgamma
    q_allowable = q_ult / fos
    return BearingCapacityResult(
        q_ult_kpa=round(q_ult, 1),
        q_allowable_kpa=round(q_allowable, 1),
        factor_of_safety=fos,
        method="Hansen",
        bearing_capacity_factors={"Nc": round(Nc, 2), "Nq": round(Nq, 2), "Nγ": round(Ngamma, 2)},
        shape_factors={"sc": round(sc, 3), "sq": round(sq, 3), "sγ": round(sgamma, 3)},
        depth_factors={"dc": round(dc, 3), "dq": round(dq, 3), "dγ": round(dgamma, 3)},
        notes=notes,
    )


@dataclass
class EC7PartialFactors:
    gamma_c: float = 1.25
    gamma_phi: float = 1.25
    gamma_cu: float = 1.4
    gamma_G: float = 1.35
    gamma_Q: float = 1.5
    gamma_Rv: float = 1.0
    combination: str = "C2"


def ec7_bearing_capacity_da1(
    soil: SoilParameters,
    foundation: FoundationGeometry,
    gwl: GroundwaterCondition | None = None,
    characteristic_permanent_load_kn: float = 0,
    characteristic_variable_load_kn: float = 0,
) -> dict:
    results = {}
    c1 = EC7PartialFactors(gamma_c=1.0, gamma_phi=1.0, gamma_cu=1.0, gamma_G=1.35, gamma_Q=1.5, gamma_Rv=1.0, combination="C1")
    c2 = EC7PartialFactors(gamma_c=1.25, gamma_phi=1.25, gamma_cu=1.4, gamma_G=1.0, gamma_Q=1.3, gamma_Rv=1.0, combination="C2")
    for name, factors in [("Combination_1", c1), ("Combination_2", c2)]:
        design_c = soil.cohesion_kpa / factors.gamma_c
        design_phi = math.degrees(math.atan(math.tan(math.radians(soil.friction_angle_deg)) / factors.gamma_phi))
        design_soil = SoilParameters(
            cohesion_kpa=design_c,
            friction_angle_deg=design_phi,
            unit_weight_kn_m3=soil.unit_weight_kn_m3,
        )
        bc_result = hansen_bearing_capacity(soil=design_soil, foundation=foundation, gwl=gwl, fos=1.0)
        R_d = bc_result.q_ult_kpa * foundation.width_m * foundation.length_m / factors.gamma_Rv
        V_d = characteristic_permanent_load_kn * factors.gamma_G + characteristic_variable_load_kn * factors.gamma_Q
        q_d = V_d / (foundation.width_m * foundation.length_m) if V_d > 0 else 0
        utilisation = V_d / R_d if R_d > 0 else float("inf")
        results[name] = {
            "partial_factors": factors,
            "design_soil_c": round(design_c, 1),
            "design_soil_phi": round(design_phi, 1),
            "q_ult_design_kpa": bc_result.q_ult_kpa,
            "R_d_kn": round(R_d, 1),
            "V_d_kn": round(V_d, 1),
            "q_design_kpa": round(q_d, 1),
            "utilisation_ratio": round(utilisation, 3),
            "pass": utilisation <= 1.0,
        }
    results["overall_pass"] = all(r["pass"] for r in results.values())
    results["governing"] = "Combination_1" if results["Combination_1"]["utilisation_ratio"] > results["Combination_2"]["utilisation_ratio"] else "Combination_2"
    return results
