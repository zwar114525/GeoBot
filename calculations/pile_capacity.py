"""
Pile foundation capacity calculations.
Includes axial capacity (skin friction + end bearing) and lateral capacity.
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PileProperties:
    """Pile geometry and properties."""
    diameter_m: float
    length_m: float
    embedment_length_m: float  # Length below ground level
    pile_type: str = "bored"  # bored, driven, CFA
    material: str = "concrete"  # concrete, steel


@dataclass
class SoilLayerPile:
    """Soil layer properties for pile design."""
    thickness_m: float
    depth_from_m: float
    cohesion_kpa: float
    friction_angle_deg: float
    unit_weight_kn_m3: float
    soil_type: str = "cohesive"  # cohesive, granular


@dataclass
class PileCapacityResult:
    """Results of pile capacity calculation."""
    ultimate_shaft_kN: float
    ultimate_end_kN: float
    ultimate_total_kN: float
    allowable_shaft_kN: float
    allowable_end_kN: float
    allowable_total_kN: float
    factor_of_safety: float
    method: str
    details: dict


def calculate_shaft_friction(
    soil_layers: list[SoilLayerPile],
    pile: PileProperties,
    alpha_method: str = "Tomlinson",
) -> float:
    """
    Calculate ultimate shaft friction (skin friction) capacity.
    
    Args:
        soil_layers: List of soil layers
        pile: Pile properties
        alpha_method: Method for calculating adhesion (Tomlinson, API, etc.)
    
    Returns:
        Ultimate shaft friction in kN
    """
    total_shaft_kN = 0.0
    pile_perimeter = math.pi * pile.diameter_m
    
    for layer in soil_layers:
        # Calculate effective stress at mid-depth of layer
        depth_mid = layer.depth_from_m + layer.thickness_m / 2
        sigma_vo = layer.unit_weight_kn_m3 * depth_mid
        
        # Simplified effective stress calculation (neglecting groundwater for now)
        if layer.soil_type == "cohesive":
            # For clay - use alpha method
            if layer.cohesion_kpa > 0:
                alpha = 1.0 if layer.cohesion_kpa < 50 else 0.5  # Simplified
                adhesion = alpha * layer.cohesion_kpa
                # Limit adhesion
                adhesion = min(adhesion, 100.0)  # kPa
                qs = adhesion
            else:
                qs = 0.0
        else:
            # For sand - use beta method
            if layer.friction_angle_deg > 0:
                beta = (1 + math.radians(layer.friction_angle_deg)) * math.tan(math.radians(layer.friction_angle_deg * 0.8))
                sigma_vo_eff = sigma_vo  # Simplified
                qs = beta * sigma_vo_eff
                # Limit shaft friction
                qs = min(qs, 100.0)  # kPa for sand
            else:
                qs = 0.0
        
        # Calculate contribution from this layer
        if pile.embedment_length_m > layer.depth_from_m:
            # Overlap between pile and layer
            overlap_start = max(layer.depth_from_m, 0)
            overlap_end = min(layer.depth_from_m + layer.thickness_m, pile.embedment_length_m)
            overlap_thickness = max(overlap_end - overlap_start, 0)
            
            layer_shaft = qs * pile_perimeter * overlap_thickness
            total_shaft_kN += layer_shaft
    
    return total_shaft_kN


def calculate_end_bearing(
    soil_at_toe: SoilLayerPile,
    pile: PileProperties,
    q_ult_method: str = "Meyerhof",
) -> float:
    """
    Calculate ultimate end bearing capacity.
    
    Args:
        soil_at_toe: Soil layer at pile toe
        pile: Pile properties
        q_ult_method: Method for ultimate bearing capacity
    
    Returns:
        Ultimate end bearing in kN
    """
    pile_area = math.pi * (pile.diameter_m / 2) ** 2
    
    # Effective stress at pile toe
    toe_depth = pile.embedment_length_m
    sigma_vo_toe = soil_at_toe.unit_weight_kn_m3 * toe_depth
    
    if soil_at_toe.soil_type == "granular" or soil_at_toe.friction_angle_deg > 0:
        # For sand - use Nq method
        phi_rad = math.radians(soil_at_toe.friction_angle_deg)
        
        if q_ult_method == "Meyerhof":
            Nq = math.exp(math.pi * math.tan(phi_rad)) * (math.tan(math.radians(45 + soil_at_toe.friction_angle_deg / 2)) ** 2)
            # Modify for pile depth effect
            if toe_depth / pile.diameter_m > 1:
                Nq *= min(toe_depth / pile.diameter_m, 15) ** 0.5
        else:
            Nq = 1.0
        
        q_ult = sigma_vo_toe * Nq
    else:
        # For clay - use bearing capacity
        Nc = 9.0  # For piles
        q_ult = soil_at_toe.cohesion_kpa * Nc
    
    # Apply limiting value
    q_ult = min(q_ult, 10000.0)  # kPa limit
    
    ultimate_end = q_ult * pile_area
    
    return ultimate_end


def calculate_pile_axial_capacity(
    pile: PileProperties,
    soil_layers: list[SoilLayerPile],
    factor_of_safety: float = 2.5,
    groundwater_depth_m: Optional[float] = None,
) -> PileCapacityResult:
    """
    Calculate complete pile axial capacity.
    
    Args:
        pile: Pile properties
        soil_layers: Soil layer data
        factor_of_safety: Safety factor for allowable capacity
        groundwater_depth_m: Groundwater depth (if any)
    
    Returns:
        PileCapacityResult with all capacity values
    """
    # Calculate shaft friction
    ultimate_shaft = calculate_shaft_friction(soil_layers, pile)
    
    # Get soil at pile toe
    soil_at_toe = soil_layers[-1] if soil_layers else None
    if soil_at_toe:
        ultimate_end = calculate_end_bearing(soil_at_toe, pile)
    else:
        ultimate_end = 0.0
    
    # Total ultimate capacity
    ultimate_total = ultimate_shaft + ultimate_end
    
    # Allowable capacities
    allowable_shaft = ultimate_shaft / factor_of_safety
    allowable_end = ultimate_end / (factor_of_safety + 0.5)  # Higher FoS for end bearing
    allowable_total = ultimate_total / factor_of_safety
    
    return PileCapacityResult(
        ultimate_shaft_kN=round(ultimate_shaft, 1),
        ultimate_end_kN=round(ultimate_end, 1),
        ultimate_total_kN=round(ultimate_total, 1),
        allowable_shaft_kN=round(allowable_shaft, 1),
        allowable_end_kN=round(allowable_end, 1),
        allowable_total_kN=round(allowable_total, 1),
        factor_of_safety=factor_of_safety,
        method="Tomlinson/Meyerhof",
        details={
            "pile_diameter_m": pile.diameter_m,
            "pile_length_m": pile.length_m,
            "pile_embedded_length_m": pile.embedment_length_m,
            "pile_type": pile.pile_type,
            "soil_layers_count": len(soil_layers),
            "groundwater_depth_m": groundwater_depth_m,
        },
    )


def calculate_pile_group_capacity(
    pile_capacity_single: PileCapacityResult,
    pile_spacing_diameter_ratio: float = 3.0,
    num_piles_x: int = 2,
    num_piles_y: int = 2,
    efficiency_method: str = "Converse-Labarre",
) -> dict:
    """
    Calculate pile group capacity considering group efficiency.
    
    Args:
        pile_capacity_single: Single pile capacity
        pile_spacing_diameter_ratio: Spacing/Diameter ratio
        num_piles_x: Number of piles in X direction
        num_piles_y: Number of piles in Y direction
        efficiency_method: Method for calculating group efficiency
    
    Returns:
        Dictionary with group capacity results
    """
    num_piles = num_piles_x * num_piles_y
    
    # Calculate group efficiency
    if efficiency_method == "Converse-Labarre":
        # Converse-Labarre formula
        theta_rad = math.atan(pile_spacing_diameter_ratio / (num_piles_x + num_piles_y))
        efficiency = 1 - theta_rad * (num_piles - 1) / (math.pi * num_piles_x * num_piles_y)
        efficiency = max(efficiency, 0.5)  # Minimum 50% efficiency
    else:
        # Simple method
        efficiency = 0.7 if pile_spacing_diameter_ratio < 3 else 0.85
    
    # Group capacity
    group_ultimate = pile_capacity_single.ultimate_total_kN * num_piles * efficiency
    group_allowable = pile_capacity_single.allowable_total_kN * num_piles * efficiency
    
    return {
        "num_piles": num_piles,
        "pile_spacing_diameter_ratio": pile_spacing_diameter_ratio,
        "group_efficiency": round(efficiency, 3),
        "group_ultimate_capacity_kN": round(group_ultimate, 1),
        "group_allowable_capacity_kN": round(group_allowable, 1),
        "capacity_per_pile_ultimate_kN": round(pile_capacity_single.ultimate_total_kN, 1),
        "capacity_per_pile_allowable_kN": round(pile_capacity_single.allowable_total_kN, 1),
    }


# Simplified function for skill executor
def pile_capacity(
    pile_diameter_m: float = 1.0,
    pile_length_m: float = 20.0,
    soil_cohesion_kpa: float = 20.0,
    soil_friction_angle_deg: float = 30.0,
    soil_unit_weight_kn_m3: float = 18.0,
    groundwater_depth_m: Optional[float] = None,
    factor_of_safety: float = 2.5,
) -> dict:
    """
    Simplified pile capacity calculation for skill executor.
    
    Args:
        pile_diameter_m: Pile diameter in meters
        pile_length_m: Pile total length
        soil_cohesion_kpa: Soil cohesion (for clay)
        soil_friction_angle_deg: Soil friction angle (for sand)
        soil_unit_weight_kn_m3: Soil unit weight
        groundwater_depth_m: Groundwater depth (optional)
        factor_of_safety: Safety factor
    
    Returns:
        Dictionary with capacity results
    """
    pile = PileProperties(
        diameter_m=pile_diameter_m,
        length_m=pile_length_m,
        embedment_length_m=pile_length_m,
    )
    
    # Determine soil type and create layer
    if soil_cohesion_kpa > 5:
        soil_type = "cohesive"
    else:
        soil_type = "granular"
    
    soil_layer = SoilLayerPile(
        thickness_m=pile_length_m,
        depth_from_m=0,
        cohesion_kpa=soil_cohesion_kpa,
        friction_angle_deg=soil_friction_angle_deg,
        unit_weight_kn_m3=soil_unit_weight_kn_m3,
        soil_type=soil_type,
    )
    
    result = calculate_pile_axial_capacity(
        pile=pile,
        soil_layers=[soil_layer],
        factor_of_safety=factor_of_safety,
        groundwater_depth_m=groundwater_depth_m,
    )
    
    return {
        "ultimate_shaft_kN": result.ultimate_shaft_kN,
        "ultimate_end_kN": result.ultimate_end_kN,
        "ultimate_total_kN": result.ultimate_total_kN,
        "allowable_shaft_kN": result.allowable_shaft_kN,
        "allowable_end_kN": result.allowable_end_kN,
        "allowable_total_kN": result.allowable_total_kN,
        "factor_of_safety": result.factor_of_safety,
        "method": result.method,
    }
