"""
Plotly figure builders and time-series prep for the Streamlit UI.
"""

import math
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


def dependency_bubble_chart(center_label: str, categories: List[dict]) -> go.Figure:
    """
    A simple radial bubble diagram: `center_label` (the instrument) in the
    middle, one bubble per dependency category spaced evenly around it in a
    circle. Each category dict needs "category_id" and "name" - category_id
    is carried as each bubble's customdata so a caller using
    st.plotly_chart(fig, on_select="rerun") can read back which one was
    clicked; the center bubble's customdata is -1 so it can be told apart
    from an actual category.
    """
    n = len(categories)
    radius = 2.5
    xs, ys, labels, customdata, sizes = [0.0], [-0.35], [center_label], [[-1]], [70]
    line_x, line_y = [], []
    for i, cat in enumerate(categories):
        angle = 2 * math.pi * i / n
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        xs.append(x)
        ys.append(y - 0.35)
        labels.append(cat["name"])
        customdata.append([cat["category_id"]])
        sizes.append(50)
        line_x += [0, x, None]
        line_y += [0, y, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode='lines',
        line=dict(color='rgba(150,150,150,0.4)', width=1),
        hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers+text',
        text=labels, textposition='bottom center',
        marker=dict(
            size=sizes,
            color=['#4f8bf9'] + ['#f97316'] * n,
            line=dict(width=2, color='white'),
        ),
        customdata=customdata,
        hovertemplate='%{text}<extra></extra>',
    ))
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[-radius - 2, radius + 2]),
        yaxis=dict(visible=False, range=[-radius - 2, radius + 1.5], scaleanchor='x'),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    return fig
