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
