"""
Multi-modal input parsers for geotechnical data.
Support for Excel/CSV lab data, borehole logs.
"""
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

try:
    import pandas as pd
    import openpyxl
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas/openpyxl not available. Excel/CSV parsing disabled.")


@dataclass
class SoilTestResult:
    """Soil test result data."""
    test_type: str
    sample_id: str
    depth_m: float
    parameters: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class BoreholeLog:
    """Borehole log data."""
    borehole_id: str
    location: Dict[str, float] = field(default_factory=dict)
    ground_level_m: float = 0.0
    water_level_m: Optional[float] = None
    layers: List[Dict] = field(default_factory=list)
    test_results: List[SoilTestResult] = field(default_factory=list)


class LabDataParser:
    """Parser for laboratory test data from Excel/CSV."""
    
    def __init__(self):
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for LabDataParser")
    
    def parse_excel(self, file_path: str, sheet_name: str = None) -> List[SoilTestResult]:
        """
        Parse Excel file with lab test results.
        
        Expected columns: Sample_ID, Depth_m, Test_Type, Parameter_Name, Value, Unit
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return self._parse_dataframe(df)
    
    def parse_csv(self, file_path: str) -> List[SoilTestResult]:
        """Parse CSV file with lab test results."""
        df = pd.read_csv(file_path)
        return self._parse_dataframe(df)
    
    def _parse_dataframe(self, df: pd.DataFrame) -> List[SoilTestResult]:
        """Parse DataFrame into SoilTestResult objects."""
        results = []
        
        # Group by sample
        if 'Sample_ID' in df.columns:
            for sample_id, group in df.groupby('Sample_ID'):
                params = {}
                for _, row in group.iterrows():
                    param_name = row.get('Parameter_Name', 'unknown')
                    value = row.get('Value')
                    if value is not None:
                        params[param_name] = float(value)
                
                results.append(SoilTestResult(
                    test_type=group['Test_Type'].iloc[0] if 'Test_Type' in group.columns else 'Unknown',
                    sample_id=sample_id,
                    depth_m=float(group['Depth_m'].iloc[0]) if 'Depth_m' in group.columns else 0.0,
                    parameters=params,
                ))
        
        return results


class BoreholeParser:
    """Parser for borehole log data."""
    
    def parse_json(self, file_path: str) -> BoreholeLog:
        """Parse JSON format borehole log."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return BoreholeLog(
            borehole_id=data.get('borehole_id', 'Unknown'),
            location=data.get('location', {}),
            ground_level_m=data.get('ground_level_m', 0.0),
            water_level_m=data.get('water_level_m'),
            layers=data.get('layers', []),
        )
    
    def parse_csv(self, file_path: str) -> BoreholeLog:
        """Parse CSV format borehole log."""
        layers = []
        metadata = {}
        
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Type') == 'layer':
                    layers.append({
                        'depth_from_m': float(row.get('Depth_From', 0)),
                        'depth_to_m': float(row.get('Depth_To', 0)),
                        'description': row.get('Description', ''),
                        'spt_n': float(row.get('SPT_N', 0)) if row.get('SPT_N') else None,
                    })
                elif row.get('Type') == 'metadata':
                    metadata[row.get('Key')] = row.get('Value')
        
        return BoreholeLog(
            borehole_id=metadata.get('borehole_id', 'Unknown'),
            ground_level_m=float(metadata.get('ground_level_m', 0)),
            water_level_m=float(metadata['water_level_m']) if metadata.get('water_level_m') else None,
            layers=layers,
        )


class MultiModalParser:
    """Unified parser for multiple input formats."""
    
    def __init__(self):
        self.lab_parser = LabDataParser() if PANDAS_AVAILABLE else None
        self.borehole_parser = BoreholeParser()
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Auto-detect file type and parse.
        
        Returns:
            Parsed data dictionary
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext in ['.xlsx', '.xls']:
            if self.lab_parser:
                return {
                    'type': 'lab_data',
                    'data': self.lab_parser.parse_excel(file_path),
                }
        elif ext == '.csv':
            # Try to detect CSV type
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if 'Sample_ID' in first_line or 'Test_Type' in first_line:
                    if self.lab_parser:
                        return {
                            'type': 'lab_data',
                            'data': self.lab_parser.parse_csv(file_path),
                        }
                elif 'Type' in first_line or 'Depth_From' in first_line:
                    return {
                        'type': 'borehole',
                        'data': self.borehole_parser.parse_csv(file_path),
                    }
        elif ext == '.json':
            return {
                'type': 'borehole',
                'data': self.borehole_parser.parse_json(file_path),
            }
        
        raise ValueError(f"Unsupported file format: {ext}")
    
    def parse_directory(self, dir_path: str) -> Dict[str, List[Dict]]:
        """
        Parse all supported files in directory.
        
        Returns:
            Dictionary mapping file paths to parsed data
        """
        results = {}
        dir_path = Path(dir_path)
        
        for ext in ['*.xlsx', '*.xls', '*.csv', '*.json']:
            for file_path in dir_path.glob(ext):
                try:
                    results[str(file_path)] = self.parse_file(str(file_path))
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")
        
        return results


def parse_geotechnical_data(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to parse geotechnical data file.
    
    Args:
        file_path: Path to data file
        
    Returns:
        Parsed data dictionary
    """
    parser = MultiModalParser()
    return parser.parse_file(file_path)
