"""Weather Risk Analysis for construction schedules."""

import pandas as pd
from datetime import timedelta
from typing import Dict, List, Optional


class WeatherRiskAnalyzer:
    """Analyze weather risk for construction activities.

    Identifies critical and outdoor activities at risk from weather delays.
    Uses historical weather data by month.
    """

    def __init__(self, tasks_df: pd.DataFrame):
        """Initialize with tasks dataframe.

        Expected columns:
        - task_id, task_name: Task identifiers
        - start_date, end_date: Task dates
        - critical: Boolean indicating critical path tasks
        - percent_complete: Progress percentage
        """
        self.tasks = tasks_df.copy()

        self.weather_risk = {
            1: {'rain_days': 8, 'temp_low': -5, 'temp_high': 5, 'name': 'January'},
            2: {'rain_days': 7, 'temp_low': -3, 'temp_high': 8, 'name': 'February'},
            3: {'rain_days': 9, 'temp_low': 2, 'temp_high': 12, 'name': 'March'},
            4: {'rain_days': 10, 'temp_low': 5, 'temp_high': 15, 'name': 'April'},
            5: {'rain_days': 11, 'temp_low': 10, 'temp_high': 20, 'name': 'May'},
            6: {'rain_days': 12, 'temp_low': 15, 'temp_high': 25, 'name': 'June'},
            7: {'rain_days': 10, 'temp_low': 18, 'temp_high': 28, 'name': 'July'},
            8: {'rain_days': 9, 'temp_low': 17, 'temp_high': 27, 'name': 'August'},
            9: {'rain_days': 8, 'temp_low': 14, 'temp_high': 23, 'name': 'September'},
            10: {'rain_days': 9, 'temp_low': 8, 'temp_high': 18, 'name': 'October'},
            11: {'rain_days': 8, 'temp_low': 3, 'temp_high': 12, 'name': 'November'},
            12: {'rain_days': 7, 'temp_low': -2, 'temp_high': 6, 'name': 'December'},
        }

        self.outdoor_keywords = [
            'excavation', 'concrete', 'steel', 'roofing', 'exterior',
            'paving', 'grading', 'site work', 'earthwork', 'foundation',
            'drainage', 'utility', 'landscape', 'asphalt', 'curb',
            'sidewalk', 'parking', 'deck', 'canopy', 'fence'
        ]

    def is_outdoor_activity(self, task_name: str) -> bool:
        """Check if task is an outdoor activity.

        Args:
            task_name: Name of the task

        Returns:
            True if task involves outdoor work
        """
        task_lower = task_name.lower()
        return any(keyword in task_lower for keyword in self.outdoor_keywords)

    def get_risk_level(self, rain_days: int, is_outdoor: bool) -> str:
        """Determine risk level based on weather and activity type.

        Args:
            rain_days: Expected rain days in the month
            is_outdoor: Whether activity is outdoor

        Returns:
            Risk level: LOW, MEDIUM, or HIGH
        """
        if not is_outdoor:
            return "LOW"

        if rain_days >= 10:
            return "HIGH"
        elif rain_days >= 7:
            return "MEDIUM"
        else:
            return "LOW"

    def add_weather_risk_analysis(self, days_ahead: int = 60) -> pd.DataFrame:
        """Analyze weather risk for upcoming critical activities.

        Args:
            days_ahead: Number of days to look ahead (default 60)

        Returns:
            DataFrame with weather risk analysis
        """
        df = self.tasks.copy()
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        today = pd.Timestamp.now()
        lookahead_end = today + timedelta(days=days_ahead)

        upcoming = df[
            (df['start_date'] >= today) &
            (df['start_date'] <= lookahead_end) &
            (df['percent_complete'] < 100)
        ].copy()

        upcoming['is_outdoor'] = upcoming['task_name'].apply(self.is_outdoor_activity)

        upcoming['month'] = upcoming['start_date'].apply(lambda x: x.month)
        upcoming['weather_risk'] = upcoming['month'].apply(
            lambda m: self.weather_risk.get(m, {}).get('rain_days', 5)
        )
        upcoming['risk_level'] = upcoming.apply(
            lambda r: self.get_risk_level(r['weather_risk'], r['is_outdoor']),
            axis=1
        )

        return upcoming.sort_values('weather_risk', ascending=False)

    def generate_risk_report(self, days_ahead: int = 60) -> str:
        """Generate weather risk report for upcoming activities.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            Formatted risk report string
        """
        risk_df = self.add_weather_risk_analysis(days_ahead)

        if risk_df.empty:
            return "No upcoming activities in the specified period."

        high_risk = risk_df[risk_df['risk_level'] == 'HIGH']
        medium_risk = risk_df[risk_df['risk_level'] == 'MEDIUM']

        output = "WEATHER RISK ANALYSIS REPORT\n"
        output += "=" * 60 + "\n"
        output += f"Period: Next {days_ahead} days\n"
        output += f"Total upcoming activities: {len(risk_df)}\n\n"

        if not high_risk.empty:
            output += f"HIGH RISK ACTIVITIES ({len(high_risk)}):\n"
            output += "-" * 40 + "\n"
            for _, act in high_risk.iterrows():
                month_name = self.weather_risk.get(act['month'], {}).get('name', 'Unknown')
                output += f"  [{act['task_id']}] {act['task_name']}\n"
                output += f"      Start: {act['start_date'].strftime('%Y-%m-%d')} ({month_name})\n"
                output += f"      Expected rain days: {act['weather_risk']}\n"
                output += f"      Recommendation: Plan indoor work or add contingency\n\n"

        if not medium_risk.empty:
            output += f"\nMEDIUM RISK ACTIVITIES ({len(medium_risk)}):\n"
            output += "-" * 40 + "\n"
            for _, act in medium_risk.head(10).iterrows():
                month_name = self.weather_risk.get(act['month'], {}).get('name', 'Unknown')
                output += f"  [{act['task_id']}] {act['task_name']} - {month_name} ({act['weather_risk']} rain days)\n"

        output += "\nRECOMMENDATIONS:\n"
        output += "-" * 40 + "\n"
        output += "1. Review high-risk activities for weather contingency plans\n"
        output += "2. Consider rescheduling critical outdoor activities to lower-risk months\n"
        output += "3. Ensure material delivery schedules account for weather delays\n"
        output += "4. Document weather conditions daily for potential delay claims\n"

        return output

    def get_risk_summary(self, days_ahead: int = 60) -> Dict:
        """Get summary statistics of weather risk.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            Dictionary with risk summary
        """
        risk_df = self.add_weather_risk_analysis(days_ahead)

        if risk_df.empty:
            return {
                'total_activities': 0,
                'high_risk': 0,
                'medium_risk': 0,
                'low_risk': 0,
                'critical_at_risk': 0
            }

        return {
            'total_activities': len(risk_df),
            'high_risk': len(risk_df[risk_df['risk_level'] == 'HIGH']),
            'medium_risk': len(risk_df[risk_df['risk_level'] == 'MEDIUM']),
            'low_risk': len(risk_df[risk_df['risk_level'] == 'LOW']),
            'critical_at_risk': len(risk_df[(risk_df['critical'] == True) & (risk_df['risk_level'].isin(['HIGH', 'MEDIUM']))])
        }
