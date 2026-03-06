import math


def rankine_earth_pressure(
    wall_height_m: float,
    backfill_friction_angle_deg: float,
    backfill_cohesion_kpa: float = 0.0,
    backfill_unit_weight: float = 18.0,
    surcharge_kpa: float = 0.0,
) -> dict:
    phi = backfill_friction_angle_deg
    H = wall_height_m
    gamma = backfill_unit_weight
    c = backfill_cohesion_kpa
    q = surcharge_kpa
    Ka = math.tan(math.radians(45 - phi / 2)) ** 2
    Kp = math.tan(math.radians(45 + phi / 2)) ** 2
    sigma_a_base = max(Ka * gamma * H + Ka * q - 2 * c * math.sqrt(Ka), 0)
    Pa_soil = 0.5 * Ka * gamma * H**2
    Pa_surcharge = Ka * q * H
    Pa_cohesion = -2 * c * math.sqrt(Ka) * H
    total_Pa = max(Pa_soil + Pa_surcharge + Pa_cohesion, 0)
    if total_Pa > 0:
        moment = Pa_soil * H / 3 + Pa_surcharge * H / 2
        y_bar = moment / total_Pa
    else:
        y_bar = H / 3
    return {
        "Ka": round(Ka, 4),
        "Kp": round(Kp, 4),
        "sigma_a_base_kpa": round(sigma_a_base, 1),
        "total_active_force_kn_m": round(total_Pa, 1),
        "Pa_soil_kn_m": round(Pa_soil, 1),
        "Pa_surcharge_kn_m": round(Pa_surcharge, 1),
        "point_of_application_m": round(y_bar, 2),
    }
