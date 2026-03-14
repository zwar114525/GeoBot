"""ML Delay Prediction for construction schedules."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class DelayPredictor:
    """Predict delay risk for upcoming activities using heuristic scoring.

    Uses multiple risk factors to estimate probability of delays:
    - Critical path status
    - Outdoor work exposure
    - Weather sensitivity
    - Predecessor dependencies
    - Current progress status
    """

    def __init__(self, tasks_df: pd.DataFrame, relationships_df: pd.DataFrame = None):
        """Initialize with tasks and optional relationships dataframe.

        Expected task columns:
        - task_id, task_name: Task identifiers
        - start_date, end_date: Task dates
        - original_duration: Original planned duration
        - remaining_duration: Remaining duration
        - percent_complete: Progress percentage
        - critical: Boolean indicating critical path
        - resource_crew: Assigned crew
        - budget_cost: Planned cost
        """
        self.tasks = tasks_df.copy()
        self.relationships = relationships_df.copy() if relationships_df is not None else None

        self.outdoor_keywords = [
            'excavation', 'concrete', 'steel', 'roofing', 'exterior',
            'paving', 'grading', 'site work', 'earthwork', 'foundation',
            'drainage', 'utility', 'landscape', 'asphalt'
        ]

        self.weather_sensitive_keywords = [
            'concrete', 'excavation', 'roofing', 'paving', 'asphalt',
            'painting', 'waterproofing'
        ]

    def is_outdoor(self, task_name: str) -> bool:
        """Check if task involves outdoor work."""
        task_lower = str(task_name).lower()
        return any(kw in task_lower for kw in self.outdoor_keywords)

    def is_weather_sensitive(self, task_name: str) -> bool:
        """Check if task is sensitive to weather conditions."""
        task_lower = str(task_name).lower()
        return any(kw in task_lower for kw in self.weather_sensitive_keywords)

    def has_predecessors(self, task_id: str) -> bool:
        """Check if task has predecessor dependencies."""
        if self.relationships is None or self.relationships.empty:
            return False
        task_str = str(task_id)
        return task_str in self.relationships['succ_task_id'].astype(str).values

    def calculate_delay_risk(self) -> pd.DataFrame:
        """Calculate delay risk score for all incomplete tasks.

        Risk scoring (0-100):
        - Critical path: 30 points
        - Outdoor work: 20 points
        - Weather sensitive: 15 points
        - Has predecessors: 15 points
        - Not started (0%): 20 points
        - Behind schedule: additional 10 points

        Returns:
            DataFrame with risk scores and levels
        """
        df = self.tasks.copy()
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        df['is_outdoor'] = df['task_name'].apply(self.is_outdoor)
        df['is_weather_sensitive'] = df['task_name'].apply(self.is_weather_sensitive)
        df['has_predecessors'] = df['task_id'].astype(str).apply(self.has_predecessors)

        df['days_until_start'] = (df['start_date'] - pd.Timestamp.now()).dt.days
        df['duration_days'] = (df['end_date'] - df['start_date']).dt.days

        df['delay_risk_score'] = (
            (df['critical'].astype(int) * 30) +
            (df['is_outdoor'].astype(int) * 20) +
            (df['is_weather_sensitive'].astype(int) * 15) +
            (df['has_predecessors'].astype(int) * 15) +
            (df['percent_complete'].apply(lambda x: 20 if x == 0 else (10 if x < 50 else 0))) +
            (df['days_until_start'].apply(lambda x: 10 if 0 < x < 7 else (5 if 0 < x < 14 else 0)))
        )

        df['risk_level'] = df['delay_risk_score'].apply(
            lambda x: 'HIGH' if x >= 50 else ('MEDIUM' if x >= 30 else 'LOW')
        )

        df['risk_factors'] = df.apply(self._get_risk_factors, axis=1)

        return df[df['percent_complete'] < 100][
            ['task_id', 'task_name', 'start_date', 'end_date', 'percent_complete',
             'critical', 'delay_risk_score', 'risk_level', 'risk_factors']
        ].sort_values('delay_risk_score', ascending=False)

    def _get_risk_factors(self, row: pd.Series) -> List[str]:
        """Get list of risk factors for a task."""
        factors = []
        if row['critical']:
            factors.append('Critical Path')
        if row['is_outdoor']:
            factors.append('Outdoor Work')
        if row['is_weather_sensitive']:
            factors.append('Weather Sensitive')
        if row['has_predecessors']:
            factors.append('Has Predecessors')
        if row['percent_complete'] == 0:
            factors.append('Not Started')
        elif row['percent_complete'] < 50:
            factors.append('Low Progress')
        if 0 < row.get('days_until_start', 0) < 14:
            factors.append('Imminent Start')
        return factors

    def predict_delay_risk(self) -> pd.DataFrame:
        """Alias for calculate_delay_risk for compatibility."""
        return self.calculate_delay_risk()

    def get_risk_summary(self) -> Dict:
        """Get summary of delay risk across the project.

        Returns:
            Dictionary with risk summary statistics
        """
        risk_df = self.calculate_delay_risk()

        if risk_df.empty:
            return {
                'total_at_risk': 0,
                'high_risk': 0,
                'medium_risk': 0,
                'low_risk': 0,
                'critical_at_risk': 0,
                'avg_risk_score': 0
            }

        return {
            'total_at_risk': len(risk_df),
            'high_risk': len(risk_df[risk_df['risk_level'] == 'HIGH']),
            'medium_risk': len(risk_df[risk_df['risk_level'] == 'MEDIUM']),
            'low_risk': len(risk_df[risk_df['risk_level'] == 'LOW']),
            'critical_at_risk': len(risk_df[(risk_df['critical'] == True) & (risk_df['risk_level'].isin(['HIGH', 'MEDIUM']))]),
            'avg_risk_score': risk_df['delay_risk_score'].mean()
        }

    def generate_risk_report(self, top_n: int = 15) -> str:
        """Generate a delay risk report.

        Args:
            top_n: Number of highest risk tasks to include

        Returns:
            Formatted risk report string
        """
        risk_df = self.calculate_delay_risk()

        if risk_df.empty:
            return "No incomplete tasks to analyze."

        summary = self.get_risk_summary()

        report = "DELAY RISK PREDICTION REPORT\n"
        report += "=" * 70 + "\n\n"

        report += "RISK SUMMARY\n"
        report += "-" * 70 + "\n"
        report += f"  Total Tasks at Risk: {summary['total_at_risk']}\n"
        report += f"  High Risk: {summary['high_risk']}\n"
        report += f"  Medium Risk: {summary['medium_risk']}\n"
        report += f"  Low Risk: {summary['low_risk']}\n"
        report += f"  Critical Tasks at Risk: {summary['critical_at_risk']}\n"
        report += f"  Average Risk Score: {summary['avg_risk_score']:.1f}\n\n"

        report += f"TOP {top_n} HIGHEST RISK ACTIVITIES\n"
        report += "-" * 70 + "\n"

        for i, (_, row) in enumerate(risk_df.head(top_n).iterrows(), 1):
            critical_tag = "[CRITICAL]" if row['critical'] else ""
            report += f"\n{i}. {critical_tag} [{row['task_id']}] {row['task_name']}\n"
            report += f"   Risk Score: {row['delay_risk_score']}/100 ({row['risk_level']})\n"
            report += f"   Start: {row['start_date'].strftime('%Y-%m-%d')} | Progress: {row['percent_complete']}%\n"
            report += f"   Risk Factors: {', '.join(row['risk_factors'])}\n"

        report += "\n" + "-" * 70 + "\n"
        report += "RECOMMENDATIONS\n"
        report += "-" * 70 + "\n"
        report += "1. Prioritize monitoring HIGH risk activities\n"
        report += "2. Review critical path tasks for schedule contingencies\n"
        report += "3. Plan weather-sensitive work during favorable conditions\n"
        report += "4. Ensure predecessor activities are on track\n"
        report += "5. Allocate resources to at-risk activities early\n"

        return report
