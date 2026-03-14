"""Delay Claim Support Module for construction schedules."""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any


class DelayClaimAnalyzer:
    """Analyze delay events for potential construction claims.

    Helps identify schedule and cost impacts from delay events
    for potential time extension and delay claims.
    """

    def __init__(self, tasks_df: pd.DataFrame, relationships_df: pd.DataFrame = None):
        """Initialize with tasks and optional relationships dataframe.

        Expected task columns:
        - task_id, task_name: Task identifiers
        - start_date, end_date: Task dates
        - original_duration: Original planned duration
        - remaining_duration: Remaining duration
        - budget_cost: Planned cost
        - actual_cost: Actual cost incurred
        - percent_complete: Progress percentage
        - critical: Boolean indicating critical path
        - wbs: Work Breakdown Structure
        """
        self.tasks = tasks_df.copy()
        self.relationships = relationships_df.copy() if relationships_df is not None else None

    def generate_delay_claim_analysis(self, delay_event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of a delay event for potential claims.

        Args:
            delay_event: Dictionary containing:
                - event_name: Name of the delay event
                - start_date: Start date of delay (YYYY-MM-DD)
                - end_date: End date of delay (YYYY-MM-DD)
                - affected_activities: List of task IDs affected
                - cause: Cause of delay (Owner, Contractor, Force Majeure, etc.)

        Returns:
            Dictionary with delay analysis results
        """
        df = self.tasks

        affected_activities = delay_event.get('affected_activities', [])
        affected = df[df['task_id'].isin(affected_activities)] if affected_activities else df

        event_start = pd.to_datetime(delay_event.get('start_date', datetime.now()))
        event_end = pd.to_datetime(delay_event.get('end_date', datetime.now()))
        event_duration = (event_end - event_start).days

        critical_affected = affected[affected['critical'] == True]

        affected_budget = affected['budget_cost'].sum() if not affected.empty else 0
        avg_duration = affected['original_duration'].mean() if not affected.empty else 0

        cost_impact = affected_budget * (event_duration / avg_duration) if avg_duration > 0 else 0

        schedule_impact_days = event_duration if len(critical_affected) > 0 else 0

        cause = delay_event.get('cause', 'Unknown')
        if 'owner' in cause.lower() or 'design' in cause.lower():
            claim_eligibility = "High"
            claim_type = "EOT (Extension of Time)"
        elif 'force majeure' in cause.lower() or 'weather' in cause.lower():
            claim_eligibility = "Medium-High"
            claim_type = "EOT + Acceleration Costs"
        elif 'contractor' in cause.lower():
            claim_eligibility = "Low"
            claim_type = "None - Contractor responsible"
        else:
            claim_eligibility = "Medium"
            claim_type = "EOT Review Required"

        return {
            'event': delay_event.get('event_name', 'Unknown Event'),
            'cause': cause,
            'claim_type': claim_type,
            'duration_days': event_duration,
            'affected_activities': len(affected),
            'critical_activities_affected': len(critical_affected),
            'schedule_impact_days': schedule_impact_days,
            'cost_impact_estimate': cost_impact,
            'claim_eligibility': claim_eligibility,
            'affected_task_ids': affected_activities,
            'recommended_action': self._get_recommended_action(schedule_impact_days, claim_eligibility, len(critical_affected))
        }

    def _get_recommended_action(self, schedule_impact: int, eligibility: str, critical_count: int) -> str:
        """Get recommended action based on analysis.

        Args:
            schedule_impact: Number of days of schedule impact
            eligibility: Claim eligibility level
            critical_count: Number of critical activities affected

        Returns:
            Recommended action string
        """
        if schedule_impact == 0:
            return "Monitor for cumulative impact. No immediate action required."

        if eligibility == "High":
            if critical_count > 0:
                return "Submit time extension request with CPM analysis immediately. Document all impacts."
            else:
                return "Prepare EOT documentation. Impact may be absorbed by float."
        elif eligibility == "Medium-High":
            return "Document delay with daily logs and weather records. Prepare contingency claim."
        elif eligibility == "Medium":
            return "Review contract terms for notification requirements. File preliminary notice."
        else:
            return "Accelerate to recover schedule. No delay claim available."

    def create_delay_claim_report(self, delay_event: Dict[str, Any]) -> str:
        """Generate a formatted delay claim report.

        Args:
            delay_event: Delay event dictionary

        Returns:
            Formatted report string
        """
        analysis = self.generate_delay_claim_analysis(delay_event)

        report = "DELAY CLAIM ANALYSIS REPORT\n"
        report += "=" * 70 + "\n\n"

        report += f"Event: {analysis['event']}\n"
        report += f"Cause: {analysis['cause']}\n"
        report += f"Duration: {analysis['duration_days']} days\n\n"

        report += "-" * 70 + "\n"
        report += "IMPACT ANALYSIS\n"
        report += "-" * 70 + "\n"
        report += f"  Activities Affected: {analysis['affected_activities']}\n"
        report += f"  Critical Activities Affected: {analysis['critical_activities_affected']}\n"
        report += f"  Schedule Impact: {analysis['schedule_impact_days']} days\n"
        report += f"  Estimated Cost Impact: ${analysis['cost_impact_estimate']:,.2f}\n\n"

        report += "-" * 70 + "\n"
        report += "CLAIM ASSESSMENT\n"
        report += "-" * 70 + "\n"
        report += f"  Claim Type: {analysis['claim_type']}\n"
        report += f"  Eligibility: {analysis['claim_eligibility']}\n"
        report += f"  Recommended Action: {analysis['recommended_action']}\n\n"

        report += "-" * 70 + "\n"
        report += "REQUIRED DOCUMENTATION\n"
        report += "-" * 70 + "\n"
        report += "  1. Detailed delay event logs with dates and times\n"
        report += "  2. Updated CPM schedule showing impact\n"
        report += "  3. Correspondence related to the delay\n"
        report += "  4. Photographic evidence of conditions\n"
        report += "  5. Resource allocation records during delay period\n"
        report += "  6. Cost records for any acceleration efforts\n"

        return report

    def identify_potential_delays(self) -> pd.DataFrame:
        """Identify tasks that may be candidates for delay claims.

        Returns:
            DataFrame of tasks with potential delay issues
        """
        df = self.tasks.copy()
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        today = pd.Timestamp.now()

        potential_delays = df[
            (df['percent_complete'] > 0) &
            (df['percent_complete'] < 100) &
            (df['end_date'] < today)
        ].copy()

        if not potential_delays.empty:
            potential_delays['days_behind'] = (today - potential_delays['end_date']).dt.days
            potential_delays['delay_severity'] = potential_delays['days_behind'].apply(
                lambda x: 'HIGH' if x > 30 else ('MEDIUM' if x > 14 else 'LOW')
            )

        return potential_delays[['task_id', 'task_name', 'percent_complete', 'end_date', 'days_behind', 'delay_severity', 'critical', 'budget_cost']].sort_values('days_behind', ascending=False)
