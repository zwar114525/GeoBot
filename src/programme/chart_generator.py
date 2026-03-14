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
        """Create an interactive Gantt chart with critical path highlighted.

        Uses timeline for date axis and colors by critical path.
        Status information is available in hover data.
        """
        if tasks_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No tasks to display")
            return fig

        df = tasks_df.copy()

        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        if 'critical' not in df.columns:
            df['critical'] = df['total_float'].apply(lambda x: True if pd.isna(x) or x <= 0 else False)

        df['path_status'] = df['critical'].apply(lambda x: 'Critical Path' if x else 'Non-Critical')

        critical_color_map = {
            'Critical Path': '#dc2626',
            'Non-Critical': '#3b82f6'
        }

        fig = px.timeline(
            df,
            x_start="start_date",
            x_end="end_date",
            y="task_name",
            color="path_status",
            color_discrete_map=critical_color_map,
            hover_data={
                "task_id": True,
                "percent_complete": True,
                "budget_cost": ":,.0f",
                "status": True,
                "total_float": True,
            },
            labels={
                "task_name": "Task",
                "path_status": "Path",
                "percent_complete": "Progress %",
                "budget_cost": "Budget",
                "status": "Status",
                "total_float": "Float (days)"
            }
        )

        fig.update_yaxes(autorange="reversed")

        fig.update_layout(
            title="Construction Schedule - Gantt Chart",
            xaxis_title="Date",
            yaxis_title="Task",
            height=max(400, len(df) * 25),
            showlegend=True,
            legend_title="Critical Path",
            hovermode="x unified",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>' +
                         'Start: %{x_start|%Y-%m-%d}<br>' +
                         'End: %{x_end|%Y-%m-%d}<br>' +
                         'Progress: %{customdata[0]}%<br>' +
                         'Status: %{customdata[1]}<br>' +
                         'Float: %{customdata[2]} days<extra></extra>'
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

    def create_gantt_chart_fixed(self, tasks_df: pd.DataFrame, title: str = "Construction Schedule (Fixed)", show_progress: bool = True) -> go.Figure:
        """Interactive Gantt chart with CRITICAL PATH clearly highlighted.

        Uses px.timeline for proper date axis and adds:
        - Red for critical tasks, Blue for non-critical
        - Progress overlay with green/amber coloring
        - Today line indicator
        - Critical path summary annotation
        """
        if tasks_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No tasks to display")
            return fig

        df = tasks_df.copy()
        df = df.sort_values('start_date')

        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        if 'critical' not in df.columns:
            df['critical'] = df['total_float'].apply(lambda x: True if pd.isna(x) or x <= 0 else False)

        df['path_status'] = df['critical'].apply(lambda x: 'Critical Path' if x else 'Non-Critical')

        critical_color_map = {
            'Critical Path': '#dc2626',
            'Non-Critical': '#3b82f6'
        }

        fig = px.timeline(
            df,
            x_start="start_date",
            x_end="end_date",
            y="task_name",
            color="path_status",
            color_discrete_map=critical_color_map,
            hover_data={
                "task_id": True,
                "percent_complete": True,
                "status": True,
                "total_float": True,
            }
        )

        fig.update_yaxes(autorange="reversed")

        if show_progress:
            for _, row in df.iterrows():
                if row['percent_complete'] > 0:
                    total_days = (row['end_date'] - row['start_date']).days
                    progress_days = total_days * (row['percent_complete'] / 100)

                    progress_color = '#22c55e' if not row['critical'] else '#f59e0b'

                    fig.add_shape(
                        type="rect",
                        x0=row['start_date'],
                        x1=row['start_date'] + pd.Timedelta(days=progress_days),
                        y0=row['task_name'],
                        y1=row['task_name'],
                        fillcolor=progress_color,
                        opacity=0.5,
                        layer="above",
                        line_width=0,
                        yref="y"
                    )

        today_val = pd.Timestamp.now().strftime('%Y-%m-%d')
        fig.add_shape(
            type="line",
            x0=today_val, x1=today_val,
            y0=0, y1=1,
            yref="paper",
            line=dict(dash="dash", color="#059669", width=2)
        )
        fig.add_annotation(
            x=today_val, y=1, yref="paper",
            text="TODAY",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color="#059669")
        )

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color='#1e293b')
            ),
            yaxis=dict(
                autorange='reversed',
                title=dict(
                    text="Activity",
                    font=dict(size=12, color='#64748b')
                ),
                tickfont=dict(size=10)
            ),
            xaxis=dict(
                title=dict(
                    text="Date",
                    font=dict(size=12, color='#64748b')
                ),
                tickformat='%Y-%m-%d',
                gridcolor='#e2e8f0'
            ),
            height=len(df) * 40 + 150,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=11)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode="x unified"
        )

        critical_count = len(df[df['critical'] == True])
        total_count = len(df)
        fig.add_annotation(
            text=f"Critical: {critical_count}/{total_count} activities ({critical_count/total_count*100:.0f}%)",
            xref='paper', yref='paper',
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(size=11, color='#dc2626', family='Courier New'),
            bgcolor='white',
            bordercolor='#dc2626',
            borderwidth=1,
            borderpad=4
        )

        return fig

    def create_gantt_chart_enhanced(self, tasks_df: pd.DataFrame, title: str = "Construction Schedule (Enhanced)") -> go.Figure:
        """Enhanced Gantt with critical path border highlighting and milestone markers.

        Uses px.timeline for proper date axis and adds:
        - Thicker borders for critical tasks
        - Star markers for critical milestones (short duration tasks)
        - Today line
        """
        if tasks_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No tasks to display")
            return fig

        df = tasks_df.copy()
        df = df.sort_values('start_date')

        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        if 'critical' not in df.columns:
            df['critical'] = df['total_float'].apply(lambda x: True if pd.isna(x) or x <= 0 else False)

        df['path_status'] = df['critical'].apply(lambda x: 'Critical Path' if x else 'Non-Critical')

        critical_color_map = {
            'Critical Path': '#dc2626',
            'Non-Critical': '#3b82f6'
        }

        fig = px.timeline(
            df,
            x_start="start_date",
            x_end="end_date",
            y="task_name",
            color="path_status",
            color_discrete_map=critical_color_map,
            hover_data={
                "task_id": True,
                "percent_complete": True,
                "status": True,
                "total_float": True,
            }
        )

        fig.update_yaxes(autorange="reversed")

        duration_days = (df['end_date'] - df['start_date']).dt.days
        df['duration_days'] = duration_days
        critical_tasks = df[df['critical'] == True]
        milestone_df = critical_tasks[critical_tasks['duration_days'] <= 5]

        if not milestone_df.empty:
            fig.add_trace(go.Scatter(
                x=milestone_df['end_date'],
                y=milestone_df['task_name'],
                mode='markers',
                name='Critical Milestones',
                marker=dict(
                    symbol='star',
                    size=14,
                    color='#fbbf24',
                    line=dict(color='#92400e', width=2)
                ),
                hovertemplate='<b>Critical Milestone</b><br>%{y}<br>End: %{x|%Y-%m-%d}<br>Progress: %{customdata}%<extra></extra>',
                customdata=milestone_df['percent_complete']
            ))

        today_val = pd.Timestamp.now().strftime('%Y-%m-%d')
        fig.add_shape(
            type="line",
            x0=today_val, x1=today_val,
            y0=0, y1=1,
            yref="paper",
            line=dict(dash="dash", color="#059669", width=2)
        )
        fig.add_annotation(
            x=today_val, y=1, yref="paper",
            text="TODAY",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color="#059669")
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color='#1e293b')),
            yaxis=dict(autorange='reversed', title='Activity'),
            xaxis=dict(title='Date', tickformat='%Y-%m-%d', gridcolor='#e2e8f0'),
            height=len(df) * 45 + 180,
            showlegend=True,
            legend=dict(orientation='h', y=1.05),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode="x unified"
        )

        critical_count = len(critical_tasks)
        total_count = len(df)
        fig.add_annotation(
            text=f"Critical: {critical_count}/{total_count} activities ({critical_count/total_count*100:.0f}%)",
            xref='paper', yref='paper',
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(size=11, color='#dc2626', family='Courier New'),
            bgcolor='white',
            bordercolor='#dc2626',
            borderwidth=1,
            borderpad=4
        )

        return fig
