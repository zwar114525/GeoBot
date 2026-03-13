"""Chart Generator - Create interactive charts for construction schedules."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


class ChartGenerator:
    """Generate interactive charts for construction schedule visualization."""

    def __init__(self):
        pass

    def create_gantt_chart(self, tasks_df: pd.DataFrame) -> go.Figure:
        """Create an interactive Gantt chart for the schedule."""
        if tasks_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No tasks to display")
            return fig

        df = tasks_df.copy()

        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        color_map = {
            'Complete': '#28a745',
            'In Progress': '#ffc107',
            'Not Started': '#6c757d'
        }
        df['color'] = df['status'].map(color_map).fillna('#6c757d')

        fig = px.timeline(
            df,
            x_start="start_date",
            x_end="end_date",
            y="task_name",
            color="status",
            color_discrete_map=color_map,
            hover_data={
                "task_id": True,
                "percent_complete": True,
                "budget_cost": ":,.0f",
                "critical": True,
            },
            labels={
                "task_name": "Task",
                "status": "Status",
                "percent_complete": "Progress %",
                "budget_cost": "Budget",
                "critical": "Critical"
            }
        )

        fig.update_yaxes(autorange="reversed")

        critical_tasks = df[df['critical'] == True]
        if not critical_tasks.empty:
            for i, row in critical_tasks.iterrows():
                fig.add_shape(
                    type="line",
                    x0=row['start_date'], x1=row['end_date'],
                    y0=row['task_name'], y1=row['task_name'],
                    line=dict(color="red", width=3),
                    layer="below"
                )

        fig.update_layout(
            title="Construction Schedule - Gantt Chart",
            xaxis_title="Date",
            yaxis_title="Task",
            height=max(400, len(df) * 25),
            showlegend=True,
            legend_title="Status",
            hovermode="x unified"
        )

        return fig

    def create_resource_chart(self, tasks_df: pd.DataFrame) -> go.Figure:
        """Create a resource allocation bar chart."""
        if tasks_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No tasks to display")
            return fig

        df = tasks_df.copy()

        resource_summary = df.groupby('resource_crew').agg({
            'budget_cost': 'sum',
            'percent_complete': 'mean',
            'task_id': 'count'
        }).reset_index()
        resource_summary.columns = ['Resource', 'Total Budget', 'Avg Progress', 'Task Count']
        resource_summary = resource_summary.sort_values('Total Budget', ascending=True)

        fig = px.bar(
            resource_summary,
            x='Total Budget',
            y='Resource',
            orientation='h',
            color='Total Budget',
            color_continuous_scale='Viridis',
            hover_data=['Task Count', 'Avg Progress'],
            labels={
                'Resource': 'Crew/Resource',
                'Total Budget': 'Total Budget ($)',
                'Task Count': 'Number of Tasks',
                'Avg Progress': 'Avg Progress %'
            }
        )

        fig.update_layout(
            title="Resource Allocation - Budget Distribution by Crew",
            xaxis_title="Total Budget ($)",
            yaxis_title="Crew/Resource",
            height=max(400, len(resource_summary) * 40),
            coloraxis_showscale=False
        )

        return fig

    def create_progress_pie(self, tasks_df: pd.DataFrame) -> go.Figure:
        """Create a pie chart showing overall progress."""
        if tasks_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No tasks to display")
            return fig

        df = tasks_df.copy()

        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']

        color_map = {
            'Complete': '#28a745',
            'In Progress': '#ffc107',
            'Not Started': '#6c757d'
        }
        colors = [color_map.get(s, '#6c757d') for s in status_counts['Status']]

        fig = px.pie(
            status_counts,
            values='Count',
            names='Status',
            color='Status',
            color_discrete_map=color_map,
            hole=0.4,
            title="Task Status Distribution"
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}: %{value} tasks (%{percent})<extra></extra>'
        )

        return fig

    def create_critical_path_chart(self, tasks_df: pd.DataFrame) -> go.Figure:
        """Create a chart highlighting critical path tasks."""
        if tasks_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No tasks to display")
            return fig

        df = tasks_df.copy()
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        critical = df[df['critical'] == True].copy()
        non_critical = df[df['critical'] == False].copy()

        fig = go.Figure()

        if not non_critical.empty:
            fig.add_trace(go.Bar(
                y=non_critical['task_name'],
                x=(non_critical['end_date'] - non_critical['start_date']).dt.days,
                orientation='h',
                name='Non-Critical',
                marker_color='#6c757d',
                hovertemplate='%{y}<br>Duration: %{x} days<extra></extra>'
            ))

        if not critical.empty:
            fig.add_trace(go.Bar(
                y=critical['task_name'],
                x=(critical['end_date'] - critical['start_date']).dt.days,
                orientation='h',
                name='Critical Path',
                marker_color='#dc3545',
                hovertemplate='%{y}<br>Duration: %{x} days<br>CRITICAL<extra></extra>'
            ))

        fig.update_layout(
            title="Task Duration - Critical Path Highlight",
            xaxis_title="Duration (days)",
            yaxis_title="Task",
            barmode='group',
            height=max(400, len(df) * 25),
            showlegend=True,
            yaxis=dict(autorange='reversed')
        )

        return fig

    def create_budget_chart(self, tasks_df: pd.DataFrame) -> go.Figure:
        """Create a budget vs actual cost chart."""
        if tasks_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No tasks to display")
            return fig

        df = tasks_df.copy()

        wbs_summary = df.groupby('wbs').agg({
            'budget_cost': 'sum',
            'actual_cost': 'sum'
        }).reset_index()

        wbs_summary = wbs_summary.sort_values('budget_cost', ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=wbs_summary['wbs'],
            x=wbs_summary['budget_cost'],
            orientation='h',
            name='Budget Cost',
            marker_color='#007bff',
            hovertemplate='%{y}<br>Budget: $%{x:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            y=wbs_summary['wbs'],
            x=wbs_summary['actual_cost'],
            orientation='h',
            name='Actual Cost',
            marker_color='#28a745',
            hovertemplate='%{y}<br>Actual: $%{x:,.0f}<extra></extra>'
        ))

        fig.update_layout(
            title="Budget vs Actual Cost by WBS",
            xaxis_title="Cost ($)",
            yaxis_title="WBS",
            barmode='group',
            height=max(400, len(wbs_summary) * 50),
            showlegend=True,
            yaxis=dict(autorange='reversed')
        )

        return fig
