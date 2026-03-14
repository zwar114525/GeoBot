"""Earned Value Management (EVM) Calculator for construction schedules."""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any


class EVMCalculator:
    """Calculate Earned Value Management metrics for construction schedules.

    EVM Metrics:
    - PV (Planned Value): Budgeted cost of work scheduled
    - EV (Earned Value): Budgeted cost of work performed
    - AC (Actual Cost): Actual cost of work performed
    - BAC (Budget at Completion): Total budgeted cost
    - SPI (Schedule Performance Index): EV/PV
    - CPI (Cost Performance Index): EV/AC
    - EAC (Estimate at Completion): BAC/CPI
    - VAC (Variance at Completion): BAC - EAC
    - TCPI (To-Complete Performance Index): (BAC-EV)/(BAC-AC)
    """

    def __init__(self, tasks_df: pd.DataFrame):
        """Initialize with tasks dataframe.

        Expected columns:
        - start_date, end_date: Task dates
        - budget_cost: Planned budget
        - actual_cost: Actual cost incurred
        - percent_complete: Progress percentage (0-100)
        """
        self.tasks = tasks_df.copy()

    def calculate_evm_metrics(self) -> Dict[str, Any]:
        """Calculate all EVM metrics.

        Returns:
            Dictionary containing all EVM metrics and derived indices
        """
        df = self.tasks

        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        today = pd.Timestamp.now()

        pv_mask = df['start_date'] <= today
        pv = df.loc[pv_mask, 'budget_cost'].sum() if pv_mask.any() else 0

        ev = (df['budget_cost'] * df['percent_complete'] / 100).sum()

        ac = df['actual_cost'].sum()

        bac = df['budget_cost'].sum()

        spi = ev / pv if pv > 0 else 1.0
        cpi = ev / ac if ac > 0 else 1.0
        eac = bac / cpi if cpi > 0 else bac
        vac = bac - eac
        tcpi = (bac - ev) / (bac - ac) if (bac - ac) > 0 else 1.0

        schedule_variance = ev - pv
        cost_variance = ev - ac

        if spi > 1.0:
            schedule_status = "Ahead of Schedule"
        elif spi >= 0.95:
            schedule_status = "On Schedule"
        else:
            schedule_status = "Behind Schedule"

        if cpi > 1.0:
            cost_status = "Under Budget"
        elif cpi >= 0.95:
            cost_status = "On Budget"
        else:
            cost_status = "Over Budget"

        return {
            'PV': pv,
            'EV': ev,
            'AC': ac,
            'BAC': bac,
            'SPI': spi,
            'CPI': cpi,
            'EAC': eac,
            'VAC': vac,
            'TCPI': tcpi,
            'SV': schedule_variance,
            'CV': cost_variance,
            'schedule_status': schedule_status,
            'cost_status': cost_status,
            'percent_complete': (ev / bac * 100) if bac > 0 else 0
        }

    def create_evm_dashboard(self) -> go.Figure:
        """Create EVM dashboard with multiple gauges and charts.

        Returns:
            Plotly figure with EVM visualizations
        """
        metrics = self.calculate_evm_metrics()

        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['SPI'] * 100,
            title={"text": "Schedule Performance Index (SPI)"},
            gauge={
                'axis': {'range': [0, 150], 'tickwidth': 1},
                'bar': {'color': "#1e293b"},
                'steps': [
                    {'range': [0, 95], 'color': "#fee2e2"},
                    {'range': [95, 105], 'color': "#fef3c7"},
                    {'range': [105, 150], 'color': "#dcfce7"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            },
            domain={'x': [0, 0.45], 'y': [0.6, 1]}
        ))

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['CPI'] * 100,
            title={"text": "Cost Performance Index (CPI)"},
            gauge={
                'axis': {'range': [0, 150], 'tickwidth': 1},
                'bar': {'color': "#1e293b"},
                'steps': [
                    {'range': [0, 95], 'color': "#fee2e2"},
                    {'range': [95, 105], 'color': "#fef3c7"},
                    {'range': [105, 150], 'color': "#dcfce7"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            },
            domain={'x': [0.55, 1], 'y': [0.6, 1]}
        ))

        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics['VAC'],
            title={"text": "Variance at Completion (VAC)"},
            number={'prefix': "$", 'font': {'size': 20}}
        ))

        fig.update_layout(
            title="Earned Value Management Dashboard",
            height=500,
            showlegend=False
        )

        return fig

    def create_evm_trend_chart(self) -> go.Figure:
        """Create EVM trend chart showing PV, EV, and AC over time.

        Returns:
            Plotly figure with trend visualization
        """
        df = self.tasks.copy()
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        df_sorted = df.sort_values('start_date')

        df_sorted['cumulative_budget'] = df_sorted['budget_cost'].cumsum()
        df_sorted['cumulative_ev'] = (df_sorted['budget_cost'] * df_sorted['percent_complete'] / 100).cumsum()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_sorted['start_date'],
            y=df_sorted['cumulative_budget'],
            mode='lines',
            name='Planned Value (PV)',
            line=dict(color='#3b82f6', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=df_sorted['start_date'],
            y=df_sorted['cumulative_ev'],
            mode='lines',
            name='Earned Value (EV)',
            line=dict(color='#22c55e', width=2)
        ))

        fig.update_layout(
            title="EVM Trend: Planned vs Earned Value",
            xaxis_title="Date",
            yaxis_title="Cumulative Cost ($)",
            hovermode="x unified",
            height=400
        )

        return fig

    def get_evm_summary_html(self) -> str:
        """Generate HTML summary of EVM metrics.

        Returns:
            HTML string for display
        """
        m = self.calculate_evm_metrics()

        status_color = {
            "Ahead of Schedule": "#22c55e",
            "On Schedule": "#3b82f6",
            "Behind Schedule": "#dc2626",
            "Under Budget": "#22c55e",
            "On Budget": "#3b82f6",
            "Over Budget": "#dc2626"
        }

        html = f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; padding: 20px;">
            <div style="background: #f8fafc; padding: 15px; border-radius: 8px;">
                <h4 style="margin: 0 0 10px 0; color: #475569;">Schedule Performance</h4>
                <div style="font-size: 24px; font-weight: bold; color: {status_color.get(m['schedule_status'], '#000')};">
                    SPI: {m['SPI']:.2f}
                </div>
                <div style="color: #64748b;">{m['schedule_status']}</div>
                <div style="color: #94a3b8; margin-top: 5px;">SV: ${m['SV']:,.0f}</div>
            </div>

            <div style="background: #f8fafc; padding: 15px; border-radius: 8px;">
                <h4 style="margin: 0 0 10px 0; color: #475569;">Cost Performance</h4>
                <div style="font-size: 24px; font-weight: bold; color: {status_color.get(m['cost_status'], '#000')};">
                    CPI: {m['CPI']:.2f}
                </div>
                <div style="color: #64748b;">{m['cost_status']}</div>
                <div style="color: #94a3b8; margin-top: 5px;">CV: ${m['CV']:,.0f}</div>
            </div>

            <div style="background: #f8fafc; padding: 15px; border-radius: 8px;">
                <h4 style="margin: 0 0 10px 0; color: #475569;">Forecast</h4>
                <div style="font-size: 24px; font-weight: bold; color: {'#22c55e' if m['VAC'] >= 0 else '#dc2626'};">
                    VAC: ${m['VAC']:,.0f}
                </div>
                <div style="color: #64748b;">EAC: ${m['EAC']:,.0f}</div>
                <div style="color: #94a3b8; margin-top: 5px;">BAC: ${m['BAC']:,.0f}</div>
            </div>
        </div>
        """

        return html
