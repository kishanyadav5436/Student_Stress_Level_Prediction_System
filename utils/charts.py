"""
Chart builder utilities using Plotly for the Student Stress Prediction System.
Provides: Stress Gauge, Probability Bars, Feature Importance Chart.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def build_stress_gauge(stress_score: float, label: str, color: str) -> go.Figure:
    """
    Build an animated circular gauge chart for stress score (0–100).
    stress_score: numeric value 0–100
    label: 'LOW', 'MEDIUM', or 'HIGH'
    color: hex color string
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=stress_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Stress Level: <b>{label}</b>", "font": {"size": 18, "color": "white"}},
        number={"font": {"size": 40, "color": "white"}, "suffix": "%"},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "rgba(255,255,255,0.5)",
                "tickfont": {"color": "rgba(255,255,255,0.7)", "size": 11}
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "rgba(76,217,123,0.15)"},
                {"range": [33, 66], "color": "rgba(255,204,0,0.15)"},
                {"range": [66, 100], "color": "rgba(255,69,58,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": stress_score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20, l=30, r=30),
        height=260,
        font={"family": "Poppins, sans-serif"}
    )
    return fig


def build_probability_chart(proba: list) -> go.Figure:
    """
    Build a horizontal bar chart showing model confidence for each class.
    proba: list of [low_prob, medium_prob, high_prob]
    """
    labels = ["😊 Low Stress", "😐 Medium Stress", "😟 High Stress"]
    colors = ["#4cd97b", "#ffcc00", "#ff453a"]
    values = [round(p * 100, 1) for p in proba]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        ),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="white", size=13, family="Poppins"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 115],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="white", size=13, family="Poppins"),
        ),
        margin=dict(t=10, b=10, l=10, r=60),
        height=180,
        bargap=0.35,
    )
    return fig


def build_feature_importance_chart(feature_names: list, importances: list) -> go.Figure:
    """
    Build a ranked horizontal bar chart of feature importances.
    """
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values("Importance", ascending=True)

    # Color scale from low to high importance
    colors = px.colors.sequential.Plasma_r[:len(df)]

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker=dict(
            color=df["Importance"],
            colorscale="Plasma",
            showscale=False,
            line=dict(color="rgba(255,255,255,0.05)", width=1)
        ),
        text=[f"{v:.3f}" for v in df["Importance"]],
        textposition="outside",
        textfont=dict(color="white", size=12, family="Poppins"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="white", size=12, family="Poppins"),
            showgrid=False,
        ),
        margin=dict(t=10, b=10, l=10, r=60),
        height=280,
        bargap=0.3,
    )
    return fig


def build_history_chart(history: list) -> go.Figure:
    """
    Build a line/scatter chart of prediction history over multiple runs.
    history: list of dicts with keys: run_num, label, score
    """
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    color_map = {"Low": "#4cd97b", "Medium": "#ffcc00", "High": "#ff453a"}

    if not history:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=220,
        )
        return fig

    runs = [h.get("run", i + 1) for i, h in enumerate(history)]
    scores = [h.get("score", 0) for h in history]
    labels = [h.get("label", "Medium") for h in history]
    colors = [color_map.get(l, "#ffcc00") for l in labels]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=runs, y=scores,
        mode="lines+markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(color="white", size=11),
        line=dict(color="rgba(255,255,255,0.3)", width=2, dash="dot"),
        marker=dict(size=14, color=colors, line=dict(color="white", width=2)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Prediction Run", font=dict(color="rgba(255,255,255,0.7)")),
            showgrid=False,
            tickfont=dict(color="rgba(255,255,255,0.7)"),
        ),
        yaxis=dict(
            title=dict(text="Stress Score", font=dict(color="rgba(255,255,255,0.7)")),
            range=[-10, 110],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="rgba(255,255,255,0.7)"),
        ),
        margin=dict(t=10, b=40, l=10, r=10),
        height=220,
    )
    return fig
