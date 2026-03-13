"""
Retaining wall design calculations.
Includes cantilever retaining wall stability checks (sliding, overturning, bearing).
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetainingWallGeometry:
    """Retaining wall geometry."""
    height_m: float  # Wall stem height
    crest_width_m: float  # Width at top of stem
    base_width_m: float  # Total base width
    stem_thickness_m: float  # Stem thickness at base
    toe_length_m: float  # Length of toe (from stem to edge of base)
    heel_length_m: float  # Length of heel (from stem to back of base)
    wall_thickness_base_m: float  # Base slab thickness


@dataclass
class BackfillProperties:
    """Backfill soil properties."""
    unit_weight_kn_m3: float  # Unit weight of backfill
    friction_angle_deg: float  # Friction angle of backfill
    cohesion_kpa: float = 0.0  # Cohesion (usually 0 for backfill)
    surcharge_kpa: float = 0.0  # Surcharge load on backfill


@dataclass
class FoundationSoilProperties:
    """Foundation soil properties."""
    unit_weight_kn_m3: float
    friction_angle_deg: float
    cohesion_kpa: float


@dataclass
class WallMaterialProperties:
    """Wall material properties."""
    unit_weight_kn_m3: float = 24.0  # Reinforced concrete
    fck_mpa: float = 30.0  # Concrete characteristic strength
    fyk_mpa: float = 500.0  # Steel characteristic strength


@dataclass
class RetainingWallResult:
    """Results of retaining wall design."""
    # Earth pressures
    ka: float  # Active earth pressure coefficient
    kp: float  # Passive earth pressure coefficient
    total_active_force_kn_m: float  # Total active force per meter
    moment_about_toe_kn_m_m: float  # Moment about toe
    passive_resistance_kn_m: float  # Passive resistance at toe
    
    # Stability checks
    sliding_fos: float
    overturning_fos: float
    bearing_capacity_fos: float
    
    # Bearing pressure
    max_bearing_pressure_kpa: float
    min_bearing_pressure_kpa: float
    
    # Status
    sliding_pass: bool
    overturning_pass: bool
    bearing_pass: bool
    
    # Weights
    total_weight_kn_m: float
    resisting_moment_kn_m_m: float


def calculate_earth_pressure_coefficients(friction_angle_deg: float) -> tuple:
    """
    Calculate Rankine earth pressure coefficients.
    
    Args:
        friction_angle_deg: Soil friction angle
    
    Returns:
        (Ka, Kp) tuple
    """
    phi_rad = math.radians(friction_angle_deg)
    
    # Active earth pressure coefficient (Rankine)
    ka = math.tan(math.radians(45 - friction_angle_deg / 2)) ** 2
    
    # Passive earth pressure coefficient (Rankine)
    kp = math.tan(math.radians(45 + friction_angle_deg / 2)) ** 2
    
    return ka, kp


def calculate_cantilever_wall_stability(
    geometry: RetainingWallGeometry,
    backfill: BackfillProperties,
    foundation: FoundationSoilProperties,
    material: Optional[WallMaterialProperties] = None,
) -> RetainingWallResult:
    """
    Calculate cantilever retaining wall stability.
    
    Args:
        geometry: Wall geometry
        backfill: Backfill properties
        foundation: Foundation soil properties
        material: Wall material properties
    
    Returns:
        RetainingWallResult with all calculations
    """
    if material is None:
        material = WallMaterialProperties()
    
    H = geometry.height_m
    B = geometry.base_width_m
    toe = geometry.toe_length_m
    heel = geometry.heel_length_m
    stem_t = geometry.stem_thickness_m
    base_t = geometry.wall_thickness_base_m
    
    # Calculate earth pressure coefficients
    ka, kp = calculate_earth_pressure_coefficients(backfill.friction_angle_deg)
    
    # === Calculate Forces ===
    
    # 1. Active earth pressure (Rankine)
    # Includes surcharge
    gamma = backfill.unit_weight_kn_m3
    q = backfill.surcharge_kpa
    
    # Resultant of active pressure (trapezoidal distribution)
    # P_a = 0.5 * gamma * H^2 * Ka + q * H * Ka
    Pa1 = 0.5 * gamma * H ** 2 * ka  # Due to soil weight
    Pa2 = q * H * ka  # Due to surcharge
    Pa_total = Pa1 + Pa2
    
    # Point of application (from base)
    # For trapezoid: centroid at H/3 from base for triangular part
    # and H/2 for rectangular (surcharge) part
    y1 = H / 3  # Centroid of triangular pressure
    y2 = H / 2  # Centroid of rectangular pressure
    
    # Total moment about toe
    # Pa1 acts at H/3 from base, Pa2 acts at H/2 from base
    Ma = Pa1 * y1 + Pa2 * y2
    
    # 2. Weight of wall
    # Stem weight
    stem_vol = stem_t * H  # per meter length
    W_stem = stem_vol * material.unit_weight_kn_m3
    
    # Base weight
    base_vol = B * base_t
    W_base = base_vol * material.unit_weight_kn_m3
    
    # Weight of soil on heel
    heel_soil_vol = heel * H
    W_heel_soil = heel_soil_vol * backfill.unit_weight_kn_m3
    
    # Weight of soil on toe (if above ground)
    # Usually toe soil is excavated, so negligible
    
    # Total weight
    W_total = W_stem + W_base + W_heel_soil
    
    # 3. Resisting moment about toe
    # Weight moments about toe
    # Stem: centroid at (toe + stem_t/2)
    M_W_stem = W_stem * (toe + stem_t / 2)
    
    # Base: centroid at (toe + B/2)
    M_W_base = W_base * (toe + B / 2)
    
    # Heel soil: centroid at (toe + stem_t + heel/2)
    M_W_heel = W_heel_soil * (toe + stem_t + heel / 2)
    
    # Total resisting moment
    M_resisting = M_W_stem + M_W_base + M_W_heel
    
    # 4. Passive resistance at toe (if applicable)
    # Usually neglected for conservative design
    # Kp * gamma * D^2 / 2 where D is depth of toe below ground
    phi_f_rad = math.radians(foundation.friction_angle_deg)
    kp_foundation = math.tan(math.radians(45 + foundation.friction_angle_deg / 2)) ** 2
    toe_depth = base_t  # Depth of toe below ground
    Pp = 0.5 * foundation.unit_weight_kn_m3 * toe_depth ** 2 * kp_foundation
    
    # === Stability Checks ===
    
    # Factor of safety against sliding
    # FOS_sliding = (Resisting forces) / (Driving forces)
    # Resisting: Weight * tan(delta) + Passive + Cohesion
    # For simplicity, use friction coefficient = 0.5 * tan(phi)
    friction_coefficient = 0.5 * math.tan(math.radians(backfill.friction_angle_deg))
    F_sliding = (W_total * friction_coefficient + Pp) / Pa_total if Pa_total > 0 else float('inf')
    sliding_pass = F_sliding >= 1.5
    
    # Factor of safety against overturning
    # FOS_overturning = (Resisting moment) / (Overturning moment)
    F_overturning = M_resisting / Ma if Ma > 0 else float('inf')
    overturning_pass = F_overturning >= 2.0
    
    # Bearing pressure
    # Use eccentric loading formula
    # Eccentricity from base center
    # e = B/2 - (M_resisting - Ma) / W_total
    eccentricity = B / 2 - (M_resisting - Ma) / W_total if W_total > 0 else 0
    
    # Bearing pressures (for L-shaped area)
    # q_max = W_total / B * (1 + 6*e / B)
    # q_min = W_total / B * (1 - 6*e / B)
    
    if abs(eccentricity) <= B / 6:
        # Entire base in compression
        q_max = W_total / B * (1 + 6 * eccentricity / B)
        q_min = W_total / B * (1 - 6 * eccentricity / B)
    else:
        # Part of base in tension - use simplified
        q_max = 2 * W_total / (3 * (B / 2 - eccentricity))
        q_min = 0
    
    # Ultimate bearing capacity (simplified Terzaghi)
    phi_f_rad = math.radians(foundation.friction_angle_deg)
    Nq = math.exp(math.pi * math.tan(phi_f_rad)) * (math.tan(math.radians(45 + foundation.friction_angle_deg / 2)) ** 2)
    Nc = (Nq - 1) / math.tan(phi_f_rad) if foundation.friction_angle_deg > 0 else 5.14
    Ngamma = 0.5 * (Nq - 1) * math.tan(phi_f_rad)
    
    qu = Nc * foundation.cohesion_kpa + 0.5 * foundation.unit_weight_kn_m3 * B * Ngamma
    
    # Bearing FOS
    F_bearing = qu / q_max if q_max > 0 else float('inf')
    bearing_pass = F_bearing >= 3.0
    
    return RetainingWallResult(
        ka=round(ka, 4),
        kp=round(kp, 4),
        total_active_force_kn_m=round(Pa_total, 2),
        moment_about_toe_kn_m_m=round(Ma, 2),
        passive_resistance_kn_m=round(Pp, 2),
        sliding_fos=round(F_sliding, 2),
        overturning_fos=round(F_overturning, 2),
        bearing_capacity_fos=round(F_bearing, 2),
        max_bearing_pressure_kpa=round(q_max, 2),
        min_bearing_pressure_kpa=round(max(q_min, 0), 2),
        sliding_pass=sliding_pass,
        overturning_pass=overturning_pass,
        bearing_pass=bearing_pass,
        total_weight_kn_m=round(W_total, 2),
        resisting_moment_kn_m_m=round(M_resisting, 2),
    )


# Simplified function for skill executor
def retaining_wall_stability(
    wall_height_m: float = 5.0,
    base_width_m: float = 4.0,
    toe_length_m: float = 1.0,
    stem_thickness_m: float = 0.4,
    base_thickness_m: float = 0.5,
    backfill_unit_weight_kn_m3: float = 18.0,
    backfill_friction_angle_deg: float = 30.0,
    backfill_surcharge_kpa: float = 0.0,
    foundation_cohesion_kpa: float = 50.0,
    foundation_friction_angle_deg: float = 25.0,
    foundation_unit_weight_kn_m3: float = 18.0,
) -> dict:
    """
    Simplified retaining wall stability calculation.
    
    Args:
        wall_height_m: Wall height
        base_width_m: Total base width
        toe_length_m: Toe length
        stem_thickness_m: Stem thickness at base
        base_thickness_m: Base slab thickness
        backfill_unit_weight_kn_m3: Backfill unit weight
        backfill_friction_angle_deg: Backfill friction angle
        backfill_surcharge_kpa: Surcharge on backfill
        foundation_cohesion_kpa: Foundation soil cohesion
        foundation_friction_angle_deg: Foundation friction angle
        foundation_unit_weight_kn_m3: Foundation unit weight
    
    Returns:
        Dictionary with stability results
    """
    geometry = RetainingWallGeometry(
        height_m=wall_height_m,
        crest_width_m=stem_thickness_m,
        base_width_m=base_width_m,
        stem_thickness_m=stem_thickness_m,
        toe_length_m=toe_length_m,
        heel_length_m=base_width_m - toe_length_m - stem_thickness_m,
        wall_thickness_base_m=base_thickness_m,
    )
    
    backfill = BackfillProperties(
        unit_weight_kn_m3=backfill_unit_weight_kn_m3,
        friction_angle_deg=backfill_friction_angle_deg,
        cohesion_kpa=0.0,
        surcharge_kpa=backfill_surcharge_kpa,
    )
    
    foundation = FoundationSoilProperties(
        unit_weight_kn_m3=foundation_unit_weight_kn_m3,
        friction_angle_deg=foundation_friction_angle_deg,
        cohesion_kpa=foundation_cohesion_kpa,
    )
    
    result = calculate_cantilever_wall_stability(geometry, backfill, foundation)
    
    return {
        "ka": result.ka,
        "kp": result.kp,
        "total_active_force_kn_m": result.total_active_force_kn_m,
        "moment_about_toe_kn_m_m": result.moment_about_toe_kn_m_m,
        "sliding_fos": result.sliding_fos,
        "overturning_fos": result.overturning_fos,
        "bearing_capacity_fos": result.bearing_capacity_fos,
        "max_bearing_pressure_kpa": result.max_bearing_pressure_kpa,
        "min_bearing_pressure_kpa": result.min_bearing_pressure_kpa,
        "sliding_pass": result.sliding_pass,
        "overturning_pass": result.overturning_pass,
        "bearing_pass": result.bearing_pass,
        "total_weight_kn_m": result.total_weight_kn_m,
        "resisting_moment_kn_m_m": result.resisting_moment_kn_m_m,
    }
