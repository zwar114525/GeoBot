import math
from dataclasses import dataclass


@dataclass
class SlopeGeometry:
    height_m: float
    angle_deg: float


@dataclass
class SlopeStabilityResult:
    factor_of_safety: float
    method: str
    critical_slip_surface: str
    pass_fail: bool
    required_fos: float
    notes: list[str]


def infinite_slope_drained(
    slope_angle_deg: float,
    friction_angle_deg: float,
    cohesion_kpa: float,
    unit_weight_kn_m3: float,
    depth_to_slip_m: float,
    gwl_above_slip_m: float = 0.0,
    gamma_w: float = 9.81,
    required_fos: float = 1.4,
) -> SlopeStabilityResult:
    beta = math.radians(slope_angle_deg)
    phi = math.radians(friction_angle_deg)
    z = depth_to_slip_m
    hw = gwl_above_slip_m
    gamma = unit_weight_kn_m3
    c = cohesion_kpa
    sigma_n = gamma * z * math.cos(beta) ** 2
    u = gamma_w * hw * math.cos(beta) ** 2
    tau_f = c + (sigma_n - u) * math.tan(phi)
    tau_d = gamma * z * math.sin(beta) * math.cos(beta)
    fos = float("inf") if tau_d <= 0 else tau_f / tau_d
    notes = []
    if hw > 0:
        ru = u / (gamma * z * math.cos(beta) ** 2)
        notes.append(f"Pore pressure ratio ru ≈ {ru:.2f}")
    pass_fail = fos >= required_fos
    return SlopeStabilityResult(
        factor_of_safety=round(fos, 3),
        method="Infinite Slope (Drained)",
        critical_slip_surface=f"Planar at {z}m depth",
        pass_fail=pass_fail,
        required_fos=required_fos,
        notes=notes,
    )


def taylor_stability_number(
    slope_angle_deg: float,
    height_m: float,
    cohesion_kpa: float,
    unit_weight_kn_m3: float,
    friction_angle_deg: float = 0.0,
    required_fos: float = 1.4,
) -> SlopeStabilityResult:
    notes = []
    if friction_angle_deg == 0:
        beta = slope_angle_deg
        if beta >= 53:
            Ns = 5.52
        elif beta >= 45:
            Ns = 5.5 + (53 - beta) * (6.0 - 5.5) / 8
        elif beta >= 30:
            Ns = 6.0 + (45 - beta) * (7.5 - 6.0) / 15
        elif beta >= 15:
            Ns = 7.5 + (30 - beta) * (11.0 - 7.5) / 15
        else:
            Ns = 11.0
        fos = Ns * cohesion_kpa / (unit_weight_kn_m3 * height_m)
        notes.append(f"Taylor Ns ≈ {Ns:.2f}")
    else:
        phi_rad = math.radians(friction_angle_deg)
        c_component = cohesion_kpa / (unit_weight_kn_m3 * height_m)
        phi_component = math.tan(phi_rad) / math.tan(math.radians(slope_angle_deg))
        fos = c_component * 5.5 + phi_component
        notes.append("Approximation for phi>0 used")
    return SlopeStabilityResult(
        factor_of_safety=round(fos, 3),
        method="Taylor's Stability Number",
        critical_slip_surface="Circular",
        pass_fail=fos >= required_fos,
        required_fos=required_fos,
        notes=notes,
    )
