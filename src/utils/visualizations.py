"""
Calculation visualizations for geotechnical engineering.
Pressure distributions, failure surfaces, bearing capacity profiles.
"""
import io
import base64
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None


@dataclass
class BearingCapacityPlot:
    """Data for bearing capacity visualization."""
    width_m: float
    depth_m: float
    q_ult_kpa: float
    q_allowable_kpa: float
    soil_parameters: Dict
    pressure_distribution: List[float] = None


class GeotechVisualizer:
    """Create visualizations for geotechnical calculations."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize visualizer."""
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style if style in plt.style.available else "default")
    
    def plot_bearing_capacity(
        self,
        data: BearingCapacityPlot,
        figsize: Tuple[int, int] = (10, 6),
    ) -> str:
        """
        Create bearing capacity visualization.
        
        Returns:
            Base64 encoded PNG image
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._placeholder_image("Matplotlib not available")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Pressure distribution diagram
        ax1 = axes[0]
        width = data.width_m
        depth = data.depth_m
        
        # Draw foundation
        foundation = plt.Rectangle((-width/2, 0), width, depth, 
                                   fill=True, alpha=0.3, color='gray',
                                   label='Foundation')
        ax1.add_patch(foundation)
        
        # Draw pressure distribution (triangular)
        x = np.linspace(-width/2, width/2, 100)
        pressure = data.q_ult_kpa * (1 - np.abs(x) / (width/2))
        ax1.fill_between(x, 0, pressure, alpha=0.5, color='red',
                        label=f'Ultimate: {data.q_ult_kpa:.1f} kPa')
        
        ax1.axhline(y=data.q_allowable_kpa, color='green', linestyle='--',
                   label=f'Allowable: {data.q_allowable_kpa:.1f} kPa')
        
        ax1.set_xlabel('Width (m)')
        ax1.set_ylabel('Pressure (kPa)')
        ax1.set_title('Bearing Pressure Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Parameter sensitivity
        ax2 = axes[1]
        params = ['Cohesion', 'Friction\nAngle', 'Unit\nWeight', 'Width', 'Depth']
        sensitivities = [0.25, 0.35, 0.15, 0.15, 0.10]  # Example values
        
        bars = ax2.bar(params, sensitivities, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_ylabel('Sensitivity')
        ax2.set_title('Parameter Sensitivity Analysis')
        ax2.axhline(y=np.mean(sensitivities), color='red', linestyle=':',
                   label='Average sensitivity')
        ax2.legend()
        
        # Add value labels
        for bar, val in zip(bars, sensitivities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def plot_pressure_distribution(
        self,
        wall_height: float,
        ka: float,
        kp: float,
        soil_unit_weight: float,
        figsize: Tuple[int, int] = (8, 10),
    ) -> str:
        """
        Plot earth pressure distribution for retaining wall.
        
        Returns:
            Base64 encoded PNG image
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._placeholder_image("Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Depth array
        z = np.linspace(0, wall_height, 100)
        
        # Active pressure (pa = Ka * gamma * z)
        pa = ka * soil_unit_weight * z
        
        # Passive pressure (pp = Kp * gamma * z)
        pp = kp * soil_unit_weight * z
        
        # Plot active pressure (left side)
        ax.fill_betweenx(z, -pa, 0, alpha=0.5, color='red', label='Active Pressure')
        ax.plot(-pa, z, 'r-', linewidth=2)
        
        # Plot passive pressure (right side)
        ax.fill_betweenx(z, 0, pp, alpha=0.5, color='green', label='Passive Pressure')
        ax.plot(pp, z, 'g-', linewidth=2)
        
        # Labels and formatting
        ax.set_xlabel('Pressure (kPa)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Earth Pressure Distribution (H={wall_height}m)')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        max_pa = ka * soil_unit_weight * wall_height
        max_pp = kp * soil_unit_weight * wall_height
        ax.text(-max_pa - 1, wall_height * 0.5, f'Ka={ka:.3f}', ha='right', va='center')
        ax.text(max_pp + 1, wall_height * 0.5, f'Kp={kp:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def plot_slope_stability(
        self,
        slope_angle: float,
        slope_height: float,
        fos: float,
        slip_surface_points: List[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> str:
        """
        Plot slope stability cross-section.
        
        Returns:
            Base64 encoded PNG image
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._placeholder_image("Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw slope
        slope_length = slope_height / np.sin(np.radians(slope_angle))
        horizontal_run = slope_height / np.tan(np.radians(slope_angle))
        
        # Ground surface
        x_ground = [0, horizontal_run, horizontal_run + 5]
        y_ground = [slope_height, 0, 0]
        ax.fill(x_ground, y_ground, alpha=0.3, color='brown', label='Soil')
        ax.plot(x_ground, y_ground, 'b-', linewidth=2, label='Ground Surface')
        
        # Draw slip surface if provided
        if slip_surface_points:
            slip_x, slip_y = zip(*slip_surface_points)
            ax.plot(slip_x, slip_y, 'r--', linewidth=2, label='Critical Slip Surface')
        else:
            # Draw approximate circular slip surface
            theta = np.linspace(np.radians(180-slope_angle), np.radians(180), 50)
            radius = slope_height / np.sin(np.radians(slope_angle)) * 0.8
            center_x = horizontal_run * 0.3
            center_y = slope_height * 0.5
            slip_x = center_x + radius * np.cos(theta)
            slip_y = center_y + radius * np.sin(theta)
            ax.plot(slip_x, slip_y, 'r--', linewidth=2, label='Critical Slip Surface')
        
        # Add FoS annotation
        fos_color = 'green' if fos >= 1.4 else 'orange' if fos >= 1.2 else 'red'
        ax.text(horizontal_run * 0.5, slope_height * 0.7,
               f'FoS = {fos:.2f}',
               bbox=dict(boxstyle='round', facecolor=fos_color, alpha=0.3),
               fontsize=12, fontweight='bold')
        
        # Labels
        ax.set_xlabel('Horizontal Distance (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f'Slope Stability Cross-Section (β={slope_angle}°)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def _placeholder_image(self, message: str) -> str:
        """Create placeholder image when matplotlib unavailable."""
        fig, ax = plt.subplots(figsize=(8, 4)) if MATPLOTLIB_AVAILABLE else (None, None)
        if ax:
            ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
            ax.axis('off')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
        return ""
    
    def plot_settlement_profile(
        self,
        foundation_width: float,
        max_settlement_mm: float,
        settlement_points: List[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (8, 5),
    ) -> str:
        """
        Plot settlement profile under foundation.
        
        Returns:
            Base64 encoded PNG image
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._placeholder_image("Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.linspace(-foundation_width, foundation_width, 100)
        
        if settlement_points:
            x_settle, y_settle = zip(*settlement_points)
            ax.plot(x_settle, y_settle, 'bo-', label='Calculated Points')
        else:
            # Approximate settlement profile (Gaussian-like)
            sigma = foundation_width / 3
            settlement = max_settlement_mm * np.exp(-x**2 / (2 * sigma**2))
            ax.fill_between(x, 0, settlement, alpha=0.5, color='blue')
            ax.plot(x, settlement, 'b-', linewidth=2, label='Settlement Profile')
        
        # Add limit line
        allowable = 25  # mm (typical limit)
        ax.axhline(y=allowable, color='red', linestyle='--', 
                  label=f'Allowable ({allowable}mm)')
        
        ax.set_xlabel('Distance from Center (m)')
        ax.set_ylabel('Settlement (mm)')
        ax.set_title('Foundation Settlement Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"


def create_visualizer() -> GeotechVisualizer:
    """Factory function to create visualizer."""
    return GeotechVisualizer()
