"""Programme Agent - Lightweight AI agent for construction schedule queries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.llm_client import call_llm_with_context
from config.settings import PRIMARY_MODEL
import pandas as pd


class ProgrammeAgent:
    """Lightweight AI agent for construction schedule queries.

    This agent reuses the LLM client but does NOT use the vector store.
    It processes schedule data directly and provides insights via LLM.
    """

    def __init__(self):
        self.tasks = None
        self.resources = None
        self.relationships = None
        self.conversation_history = []

    def load_schedule(self, tasks: pd.DataFrame, resources: pd.DataFrame = None, relationships: pd.DataFrame = None):
        """Load schedule data into the agent."""
        self.tasks = tasks
        self.resources = resources
        self.relationships = relationships

    def _build_context(self) -> str:
        """Build context string from schedule data for LLM prompts."""
        if self.tasks is None or self.tasks.empty:
            return "No schedule data loaded."

        context_parts = []

        context_parts.append("=== CONSTRUCTION SCHEDULE DATA ===")

        summary = self._get_schedule_summary()
        context_parts.append(f"\nSchedule Summary:")
        context_parts.append(f"- Total Tasks: {summary['total_tasks']}")
        context_parts.append(f"- Completed: {summary['completed']}")
        context_parts.append(f"- In Progress: {summary['in_progress']}")
        context_parts.append(f"- Not Started: {summary['not_started']}")
        context_parts.append(f"- Critical Tasks: {summary['critical_tasks']}")
        context_parts.append(f"- Total Budget: ${summary['total_budget']:,.2f}")
        context_parts.append(f"- Actual Spent: ${summary['actual_spent']:,.2f}")
        context_parts.append(f"- Overall Progress: {summary['percent_complete']:.1f}%")
        context_parts.append(f"- Project Start: {summary['start_date'].strftime('%Y-%m-%d') if summary.get('start_date') else 'N/A'}")
        context_parts.append(f"- Project End: {summary['end_date'].strftime('%Y-%m-%d') if summary.get('end_date') else 'N/A'}")

        context_parts.append("\n=== TASK DETAILS ===")
        task_list = []
        for _, task in self.tasks.iterrows():
            task_str = (
                f"Task {task['task_id']}: {task['task_name']}\n"
                f"  WBS: {task['wbs']}, Status: {task['status']}, Progress: {task['percent_complete']}%\n"
                f"  Duration: {task['original_duration']} days, Remaining: {task['remaining_duration']:.1f} days\n"
                f"  Budget: ${task['budget_cost']:,.2f}, Actual: ${task['actual_cost']:,.2f}\n"
                f"  Start: {task['start_date'].strftime('%Y-%m-%d')}, End: {task['end_date'].strftime('%Y-%m-%d')}\n"
                f"  Critical: {'Yes' if task['critical'] else 'No'}, Float: {task['total_float']:.1f} days\n"
                f"  Crew: {task.get('resource_crew', 'N/A')}"
            )
            task_list.append(task_str)

        context_parts.append("\n\n".join(task_list[:50]))

        if self.resources is not None and not self.resources.empty:
            context_parts.append("\n=== RESOURCE ALLOCATIONS ===")
            res_summary = self.resources.groupby('resource_name')['cost_per_unit'].sum().reset_index()
            context_parts.append(res_summary.to_string(index=False))

        return "\n".join(context_parts)

    def _get_schedule_summary(self) -> dict:
        """Get schedule summary statistics."""
        if self.tasks is None or self.tasks.empty:
            return {
                'total_tasks': 0, 'completed': 0, 'in_progress': 0, 'not_started': 0,
                'critical_tasks': 0, 'total_budget': 0, 'actual_spent': 0,
                'percent_complete': 0, 'start_date': None, 'end_date': None
            }

        total_tasks = len(self.tasks)
        completed = len(self.tasks[self.tasks['percent_complete'] == 100])
        in_progress = len(self.tasks[(self.tasks['percent_complete'] > 0) & (self.tasks['percent_complete'] < 100)])
        not_started = len(self.tasks[self.tasks['percent_complete'] == 0])
        critical_tasks = len(self.tasks[self.tasks['critical'] == True])

        total_budget = self.tasks['budget_cost'].sum()
        actual_spent = self.tasks['actual_cost'].sum()

        return {
            'total_tasks': total_tasks,
            'completed': completed,
            'in_progress': in_progress,
            'not_started': not_started,
            'critical_tasks': critical_tasks,
            'total_budget': total_budget,
            'actual_spent': actual_spent,
            'percent_complete': (actual_spent / total_budget * 100) if total_budget > 0 else 0,
            'start_date': self.tasks['start_date'].min() if 'start_date' in self.tasks.columns else None,
            'end_date': self.tasks['end_date'].max() if 'end_date' in self.tasks.columns else None
        }

    def ask(self, question: str, model: str = PRIMARY_MODEL) -> dict:
        """Answer a question about the construction schedule."""
        if self.tasks is None or self.tasks.empty:
            return {
                'answer': 'No schedule data loaded. Please upload an XER file or load demo data first.',
                'sources': []
            }

        context = self._build_context()

        prompt = f"""You are a construction programme manager assistant. Use the schedule data provided below to answer the user's question.

If the question asks for specific numbers or calculations, provide them based on the data. If asked about task details, reference the specific task IDs and names.

Question: {question}

Schedule Data:
{context}

Provide a clear, concise answer based on the schedule data."""

        try:
            answer = call_llm_with_context(
                question=question,
                context_chunks=[{'text': context, 'metadata': {'source': 'schedule_data'}}],
                model=model,
                equation_mode=False
            )
        except Exception as e:
            answer = f"I encountered an error while processing your question: {str(e)}"

        self.conversation_history.append({'question': question, 'answer': answer})

        return {'answer': answer, 'sources': []}

    def get_schedule_summary(self) -> dict:
        """Get a summary of the current schedule."""
        return self._get_schedule_summary()

    def identify_critical_path(self) -> list:
        """Identify tasks on the critical path."""
        if self.tasks is None or self.tasks.empty:
            return []
        critical = self.tasks[self.tasks['critical'] == True].sort_values('start_date')
        return critical[['task_id', 'task_name', 'start_date', 'end_date', 'percent_complete']].to_dict('records')

    def find_delays(self) -> list:
        """Identify tasks that are behind schedule."""
        if self.tasks is None or self.tasks.empty:
            return []
        delayed = self.tasks[
            (self.tasks['percent_complete'] > 0) &
            (self.tasks['percent_complete'] < 100)
        ].copy()

        delayed['planned_progress'] = delayed.apply(
            lambda row: min(100, max(0,
                (pd.Timestamp.now() - row['start_date']).days / row['original_duration'] * 100
            )) if pd.notna(row['start_date']) and pd.notna(row['original_duration']) and row['original_duration'] > 0 else 0,
            axis=1
        )

        delayed_behind = delayed[delayed['percent_complete'] < delayed['planned_progress'] * 0.8]

        return delayed_behind[['task_id', 'task_name', 'percent_complete', 'planned_progress', 'status']].to_dict('records')

    def generate_lookahead_report(self, weeks: int = 4) -> str:
        """Generate 4-week look-ahead report for weekly coordination meetings.

        Args:
            weeks: Number of weeks to look ahead (default 4)

        Returns:
            Formatted lookahead report string
        """
        if self.tasks is None or self.tasks.empty:
            return "No schedule data loaded."

        from datetime import timedelta

        df = self.tasks.copy()
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        today = pd.Timestamp.now()
        lookahead_end = today + timedelta(weeks=weeks)

        lookahead = df[
            ((df['start_date'] >= today) & (df['start_date'] <= lookahead_end)) |
            ((df['end_date'] >= today) & (df['end_date'] <= lookahead_end)) |
            ((df['start_date'] <= today) & (df['end_date'] >= today))
        ].copy()

        lookahead = lookahead.sort_values('start_date')

        output = f"{weeks}-WEEK LOOK-AHEAD REPORT\n"
        output += f"Generated: {today.strftime('%Y-%m-%d')} | Period: {today.strftime('%Y-%m-%d')} to {lookahead_end.strftime('%Y-%m-%d')}\n"
        output += "=" * 70 + "\n\n"

        this_week_end = today + timedelta(days=7)
        this_week = lookahead[
            (lookahead['start_date'] <= this_week_end) &
            (lookahead['end_date'] >= today)
        ]

        output += f"THIS WEEK ({today.strftime('%Y-%m-%d')} to {this_week_end.strftime('%Y-%m-%d')}):\n"
        output += "-" * 70 + "\n"

        if this_week.empty:
            output += "  No activities scheduled this week.\n"
        else:
            for _, act in this_week.iterrows():
                critical_marker = "[CRITICAL]" if act['critical'] else ""
                output += f"  {critical_marker} [{act['task_id']}] {act['task_name']}\n"
                output += f"      Crew: {act.get('resource_crew', 'N/A')} | Progress: {act['percent_complete']}%\n"
                output += f"      {act['start_date'].strftime('%Y-%m-%d')} to {act['end_date'].strftime('%Y-%m-%d')}\n\n"

        next_weeks = lookahead[
            (lookahead['start_date'] > this_week_end) &
            (lookahead['start_date'] <= lookahead_end)
        ]

        if not next_weeks.empty:
            output += f"\nUPCOMING (Next {weeks-1} Weeks):\n"
            output += "-" * 70 + "\n"
            for _, act in next_weeks.iterrows():
                critical_marker = "[CRITICAL]" if act['critical'] else ""
                output += f"  {critical_marker} [{act['task_id']}] {act['task_name']}\n"
                output += f"      Starts: {act['start_date'].strftime('%Y-%m-%d')} | Crew: {act.get('resource_crew', 'N/A')}\n"

        output += "\nREQUIRED RESOURCES:\n"
        output += "-" * 70 + "\n"
        resource_needs = lookahead.groupby('resource_crew')['task_name'].count()
        for crew, count in resource_needs.items():
            output += f"  {crew}: {count} activities\n"

        return output

    def generate_subcontractor_coordination_report(self) -> str:
        """Generate handoff report between subcontractors.

        Identifies dependencies and coordination points between different crews.

        Returns:
            Formatted subcontractor coordination report
        """
        if self.tasks is None or self.tasks.empty:
            return "No schedule data loaded."

        df = self.tasks.copy()
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        subcontractors = df.groupby('resource_crew').agg({
            'task_name': list,
            'start_date': 'min',
            'end_date': 'max',
            'percent_complete': 'mean',
            'critical': 'sum'
        }).reset_index()

        output = "SUBCONTRACTOR COORDINATION REPORT\n"
        output += "=" * 60 + "\n\n"

        output += "CREW SCHEDULE SUMMARY:\n"
        output += "-" * 60 + "\n"

        for _, row in subcontractors.iterrows():
            crew = row['resource_crew'] if pd.notna(row['resource_crew']) else 'Unassigned'
            output += f"\n{crew}:\n"
            output += f"  Tasks: {len(row['task_name'])} | Avg Progress: {row['percent_complete']:.1f}%\n"
            output += f"  Period: {row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}\n"

        handoffs = []
        for i, (crew1, row1) in enumerate(subcontractors.iterrows()):
            for crew2, row2 in list(subcontractors.iterrows())[i+1:]:
                if abs((row1['end_date'] - row2['start_date']).days) <= 7:
                    handoffs.append({
                        'from_crew': crew1 if pd.notna(crew1) else 'Unassigned',
                        'to_crew': crew2 if pd.notna(crew2) else 'Unassigned',
                        'handoff_date': row1['end_date'],
                        'from_tasks': row1['task_name'][:2] if isinstance(row1['task_name'], list) else [],
                        'to_tasks': row2['task_name'][:2] if isinstance(row2['task_name'], list) else []
                    })

        output += f"\n\nUPCOMING HANDOFFS ({len(handoffs)} identified):\n"
        output += "-" * 60 + "\n"

        if not handoffs:
            output += "  No significant handoffs identified in the next 7 days.\n"
        else:
            for h in handoffs[:10]:
                output += f"\n  {h['handoff_date'].strftime('%Y-%m-%d')}: "
                output += f"{h['from_crew']} -> {h['to_crew']}\n"
                if h['from_tasks']:
                    output += f"    From: {', '.join(str(t) for t in h['from_tasks'])}\n"
                if h['to_tasks']:
                    output += f"    To: {', '.join(str(t) for t in h['to_tasks'])}\n"

        output += "\nCOORDINATION ACTIONS:\n"
        output += "-" * 60 + "\n"
        output += "  1. Schedule pre-handoff meetings 1 week before each date\n"
        output += "  2. Verify work completion certificates before handoff\n"
        output += "  3. Document site conditions with photos at transfer\n"
        output += "  4. Review interface details between trades\n"

        return output
