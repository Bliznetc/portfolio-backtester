"""
Plotly figure builders and time-series prep for the Streamlit UI.
"""

from typing import Dict, List

import plotly.express as px
import plotly.graph_objects as go

from ..core.portfolio_calculator import PortfolioSnapshot

MAX_TICK_LABELS = 10


def prepare_time_series(period: str, time_series: List[PortfolioSnapshot]) -> List[PortfolioSnapshot]:
    """
    Trim a period's snapshots for charting: the 1D chart shows only the last
    trading day; every longer period drops weekend points.
    """
    if period == '1d':
        if not time_series:
            return time_series
        last_date = time_series[-1].date.date()
        return [s for s in time_series if s.date.date() == last_date]
    return [s for s in time_series if s.date.weekday() < 5]


def _index_axis(date_labels: List[str]) -> dict:
    """
    X axis over sequential indices (not datetimes) so gaps such as weekends
    and overnight don't draw vertical lines; ticks are relabelled with dates.
    """
    step = max(1, len(date_labels) // MAX_TICK_LABELS)
    tick_indices = list(range(0, len(date_labels), step))
    return dict(
        tickmode='array',
        tickvals=tick_indices,
        ticktext=[date_labels[i] for i in tick_indices],
        tickangle=-45,
    )


def allocation_pie(weights: Dict[str, float], baseline_amount: float) -> go.Figure:
    dollar_values = {ticker: weight * baseline_amount for ticker, weight in weights.items()}
    fig = px.pie(
        values=list(dollar_values.values()),
        names=list(dollar_values.keys()),
        title="Portfolio Allocation",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Value: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>',
    )
    return fig


def portfolio_value_chart(
    period: str,
    date_labels: List[str],
    values: List[float],
    baseline_amount: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(values))),
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        text=date_labels,
        hovertemplate='%{text}<br>Value: $%{y:,.2f}<extra></extra>',
    ))
    fig.add_hline(
        y=baseline_amount,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Initial Value (${baseline_amount:,.0f})",
    )
    fig.update_layout(
        title=f"Portfolio Value Over Time ({period.upper()})",
        xaxis_title="Time",
        yaxis_title="Value ($)",
        hovermode='closest',
        xaxis=_index_axis(date_labels),
    )
    return fig


def return_pct_chart(
    period: str,
    date_labels: List[str],
    returns: List[float],
    final_return_pct: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(returns))),
        y=returns,
        mode='lines',
        name='Return %',
        line=dict(color='green' if final_return_pct >= 0 else 'red', width=2),
        fill='tozeroy',
        text=date_labels,
        hovertemplate='%{text}<br>Return: %{y:.2f}%<extra></extra>',
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Return Percentage Over Time ({period.upper()})",
        xaxis_title="Time",
        yaxis_title="Return (%)",
        hovermode='closest',
        xaxis=_index_axis(date_labels),
    )
    return fig
