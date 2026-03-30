import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scripts.careflow_utils import (
    FEATURES_LIST,
    FEATURE_LABELS,
    MODEL_FEATURE_DESC,
    load_data,
    train_models,
)

st.set_page_config(
    page_title="CareFlow Analytics | Product Data Analyst Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = dict(
    bg="#F0F4F8",
    surface="#FFFFFF",
    card="#FFFFFF",
    border="#DDE3ED",
    teal="#0D9488",
    teal_lt="#CCFBF1",
    navy="#1E3A5F",
    slate="#475569",
    muted="#94A3B8",
    green="#16A34A",
    amber="#D97706",
    red="#DC2626",
    sky="#0EA5E9",
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: {C['bg']};
    color: {C['slate']};
}}
.stApp {{
    background-color: {C['bg']};
}}

section[data-testid="stSidebar"] {{
    background: {C['navy']};
    border-right: none;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {{
    color: #CBD5E1 !important;
}}

div[data-testid="stMetric"] {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-top: 3px solid {C['teal']};
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}}
div[data-testid="stMetricValue"] {{
    color: {C['navy']};
    font-weight: 800;
    font-size: 1.75rem;
    font-family: 'JetBrains Mono', monospace;
}}
div[data-testid="stMetricLabel"] {{
    color: {C['muted']};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
}}

.ds-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
}}
.ds-callout {{
    background: {C['teal_lt']};
    border-left: 4px solid {C['teal']};
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    color: {C['navy']};
    font-size: 0.95rem;
}}

.page-title {{
    font-size: 1.9rem;
    font-weight: 800;
    color: {C['navy']};
    line-height: 1.2;
    margin-bottom: 0.3rem;
}}
.page-sub {{
    color: {C['slate']};
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}}
.sec-title {{
    font-size: 0.78rem;
    font-weight: 700;
    color: {C['teal']};
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.75rem;
    margin-top: 0.5rem;
}}
.tag {{
    display: inline-block;
    background: {C['teal_lt']};
    color: {C['teal']};
    border: 1px solid #99F6E4;
    padding: 3px 11px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}}

.badge-green {{
    background:#DCFCE7;
    color:{C['green']};
    border:1px solid #BBF7D0;
    padding:4px 13px;
    border-radius:999px;
    font-weight:700;
    font-size:0.85rem;
}}
.badge-amber {{
    background:#FEF3C7;
    color:{C['amber']};
    border:1px solid #FDE68A;
    padding:4px 13px;
    border-radius:999px;
    font-weight:700;
    font-size:0.85rem;
}}
.badge-red {{
    background:#FEE2E2;
    color:{C['red']};
    border:1px solid #FECACA;
    padding:4px 13px;
    border-radius:999px;
    font-weight:700;
    font-size:0.85rem;
}}

hr {{
    border-color: {C['border']};
}}
</style>
""",
    unsafe_allow_html=True,
)

df = load_data()
art = train_models(df)

PLOT_BASE = dict(
    paper_bgcolor=C["card"],
    plot_bgcolor=C["card"],
    font_color=C["slate"],
    font_family="Plus Jakarta Sans, sans-serif",
    margin=dict(l=16, r=16, t=44, b=16),
    colorway=[C["teal"], C["sky"], C["amber"], C["green"], C["red"]],
)
AXIS = dict(
    gridcolor=C["border"],
    linecolor=C["border"],
    tickcolor=C["muted"],
    tickfont_color=C["muted"],
)


def style(fig):
    fig.update_layout(**PLOT_BASE)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


with st.sidebar:
    st.markdown(
        """
    <div style="padding:1rem 0 0.8rem;">
        <div style="font-size:1.5rem; font-weight:800; color:#FFFFFF;">🏥 CareFlow</div>
        <div style="color:#94A3B8; font-size:0.8rem; margin-top:3px; font-weight:500;">
            Product Data Analyst Dashboard
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#2D4A6E; margin:0 0 1rem 0'>", unsafe_allow_html=True)

    page = st.radio(
        "",
        [
            "🏠  Overview",
            "📊  Adoption Analysis",
            "👤  User Segments",
            "🤖  Engagement Model",
            "💡  SHAP Explainability",
            "📋  Project Report",
        ],
    )
    page = page.split("  ")[1].strip()

    st.markdown("<hr style='border-color:#2D4A6E; margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown(
        f"""
    <div style="background:#1E3A5F; border:1px solid #2D4A6E; border-radius:10px; padding:0.9rem;">
        <div style="color:#64748B; font-size:0.72rem; text-transform:uppercase; letter-spacing:.07em; margin-bottom:6px;">Dataset</div>
        <div style="color:#CBD5E1; font-size:0.88rem;">{len(df):,} clinician users</div>
        <div style="color:#64748B; font-size:0.78rem; margin-top:4px;">{df['power_user'].mean():.0%} power users</div>
        <div style="color:#64748B; font-size:0.78rem;">Best model: {art['best_name']}</div>
        <div style="color:#64748B; font-size:0.78rem;">ROC-AUC: {art['results']['ROC-AUC'].max():.3f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<span class="tag">Healthcare Tech · Product Data Analyst Portfolio</span>
<div class="page-title">CareFlow Feature Adoption Dashboard</div>
<div class="page-sub">
    Analysing how clinicians adopt and engage with a clinical workflow platform —
    built to demonstrate Product Data Analyst skills.
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

if page == "Overview":
    avg_features = df["features_adopted"].mean()
    ttfv_median = df["ttfv_days"].median()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Power User Rate", f"{df['power_user'].mean():.1%}")
    c2.metric("Avg Features Adopted", f"{avg_features:.1f} / 8")
    c3.metric("Median Time-to-Value", f"{int(ttfv_median)} days")
    c4.metric("Best Model ROC-AUC", f"{art['results']['ROC-AUC'].max():.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Feature Adoption Rates — All Users</div>', unsafe_allow_html=True)
    adopt_rates = pd.DataFrame({
        "feature": [FEATURE_LABELS[f] for f in FEATURES_LIST],
        "adoption_rate": [df[f"adopted_{f}"].mean() for f in FEATURES_LIST],
    }).sort_values("adoption_rate", ascending=True)

    fig = px.bar(
        adopt_rates,
        x="adoption_rate",
        y="feature",
        orientation="h",
        text=adopt_rates["adoption_rate"].apply(lambda x: f"{x:.0%}"),
        color="adoption_rate",
        color_continuous_scale=[C["red"], C["amber"], C["teal"]],
        title="% of Users Who Have Activated Each Feature",
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    style(fig)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="sec-title">Power Users by Plan</div>', unsafe_allow_html=True)
        plan_pu = df.groupby("plan")["power_user"].mean().reset_index()
        plan_pu.columns = ["plan", "power_user_rate"]

        fig2 = px.bar(
            plan_pu,
            x="plan",
            y="power_user_rate",
            color="plan",
            color_discrete_map={
                "Starter": C["amber"],
                "Professional": C["teal"],
                "Enterprise": C["sky"],
            },
            text=plan_pu["power_user_rate"].apply(lambda x: f"{x:.0%}"),
            title="Power User Rate by Subscription Plan",
        )
        fig2.update_traces(textposition="outside")
        style(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        hist_df = df.copy()
        hist_df["power_user_label"] = hist_df["power_user"].map({0: "Non-Power", 1: "Power User"})

        st.markdown('<div class="sec-title">Feature Breadth Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(
            hist_df,
            x="features_adopted",
            color="power_user_label",
            color_discrete_map={"Non-Power": C["amber"], "Power User": C["teal"]},
            barmode="overlay",
            nbins=9,
            opacity=0.8,
            labels={"power_user_label": "User Type", "features_adopted": "Features Adopted"},
            title="Number of Features Adopted — Power vs Non-Power Users",
        )
        style(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    top_driver = art["rf_fi"].iloc[0]["feature"]
    st.markdown(
        f"""
    <div class="ds-callout">
        💡 <strong>Key insight:</strong>
        Feature adoption breadth is a stronger predictor of engagement than login frequency.
        The model identifies <em>{MODEL_FEATURE_DESC.get(top_driver, top_driver)}</em>
        as the top driver of power-user status.
    </div>
    """,
        unsafe_allow_html=True,
    )

elif page == "Adoption Analysis":
    st.markdown(
        """
    <div class="ds-callout">
        This page explores how users move through the feature adoption journey
        — from first login to deeper usage of advanced features.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-title">Feature Adoption Funnel</div>', unsafe_allow_html=True)
    funnel_data = pd.DataFrame({
        "Stage": [f"At least {i} feature{'s' if i > 1 else ''}" for i in range(1, 9)],
        "Users": [int((df["features_adopted"] >= i).sum()) for i in range(1, 9)],
    })
    fig = px.funnel(
        funnel_data,
        x="Users",
        y="Stage",
        color_discrete_sequence=[C["teal"]],
        title="Feature Adoption Funnel — CareFlow Platform",
    )
    style(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title">Adoption Rate by Clinician Role</div>', unsafe_allow_html=True)
    role_adopt = df.groupby("role")[[f"adopted_{f}" for f in FEATURES_LIST]].mean().reset_index()
    role_adopt.columns = ["role"] + [FEATURE_LABELS[f] for f in FEATURES_LIST]
    role_melt = role_adopt.melt(id_vars="role", var_name="Feature", value_name="Adoption Rate")

    fig2 = px.bar(
        role_melt,
        x="Feature",
        y="Adoption Rate",
        color="role",
        barmode="group",
        title="Feature Adoption Rate by Clinician Role",
        color_discrete_sequence=[C["teal"], C["sky"], C["amber"], C["green"]],
    )
    fig2.update_xaxes(tickangle=-30)
    style(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    ttfv_df = df.copy()
    ttfv_df["power_user_label"] = ttfv_df["power_user"].map({0: "Non-Power", 1: "Power User"})

    st.markdown('<div class="sec-title">Time-to-First-Value</div>', unsafe_allow_html=True)
    fig3 = px.histogram(
        ttfv_df,
        x="ttfv_days",
        color="power_user_label",
        color_discrete_map={"Non-Power": C["amber"], "Power User": C["teal"]},
        nbins=30,
        opacity=0.8,
        barmode="overlay",
        labels={"power_user_label": "User Type", "ttfv_days": "Days to First Value"},
        title="Power users reach first value faster",
    )
    style(fig3)
    st.plotly_chart(fig3, use_container_width=True)

elif page == "User Segments":
    st.markdown('<div class="sec-title">Engagement by Department</div>', unsafe_allow_html=True)
    dept_stats = df.groupby("department").agg(
        users=("user_id", "count"),
        power_user_rate=("power_user", "mean"),
        avg_features=("features_adopted", "mean"),
    ).reset_index()

    fig = px.scatter(
        dept_stats,
        x="avg_features",
        y="power_user_rate",
        size="users",
        color="department",
        text="department",
        size_max=50,
        color_discrete_sequence=[C["teal"], C["sky"], C["amber"], C["green"], C["red"]],
        title="Department: Avg Features Adopted vs Power User Rate",
        labels={"avg_features": "Avg Features Adopted", "power_user_rate": "Power User Rate"},
    )
    fig.update_traces(textposition="top center")
    style(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title">Power User Rate by Cohort</div>', unsafe_allow_html=True)
    cohort_df = df.copy()
    cohort_df["cohort_q"] = cohort_df["cohort_month"].dt.to_period("Q").astype(str)
    cohort_stats = cohort_df.groupby("cohort_q").agg(
        users=("user_id", "count"),
        power_rate=("power_user", "mean"),
    ).reset_index()

    fig2 = px.bar(
        cohort_stats,
        x="cohort_q",
        y="power_rate",
        color="power_rate",
        color_continuous_scale=[C["amber"], C["teal"]],
        text=cohort_stats["power_rate"].apply(lambda x: f"{x:.0%}"),
        title="Power User Rate by Signup Cohort",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_coloraxes(showscale=False)
    style(fig2)
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Engagement Model":
    st.markdown('<div class="sec-title">Model Performance Comparison</div>', unsafe_allow_html=True)
    results = art["results"].copy()
    st.dataframe(results, use_container_width=True)

    fig = px.bar(
        results,
        x="Model",
        y="ROC-AUC",
        color="Model",
        text=results["ROC-AUC"].apply(lambda x: f"{x:.3f}"),
        title="Model ROC-AUC Comparison",
    )
    fig.update_traces(textposition="outside")
    style(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title">Top Random Forest Drivers</div>', unsafe_allow_html=True)
    top_rf = art["rf_fi"].head(10).copy()
    top_rf["description"] = top_rf["feature"].map(MODEL_FEATURE_DESC).fillna(top_rf["feature"])
    st.dataframe(top_rf, use_container_width=True)

elif page == "SHAP Explainability":
    st.markdown(
        """
    <div class="ds-callout">
        SHAP helps explain which features most influenced model predictions.
        This supports transparent, decision-ready analytics for stakeholders.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if art["shap_arr"] is None:
        st.warning(f"SHAP output unavailable. {art['shap_error'] or ''}")
    else:
        shap_importance = pd.DataFrame({
            "feature": art["X_tr"].columns,
            "mean_abs_shap": np.abs(art["shap_arr"]).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).head(12)

        shap_importance["description"] = (
            shap_importance["feature"].map(MODEL_FEATURE_DESC).fillna(shap_importance["feature"])
        )

        fig = px.bar(
            shap_importance.sort_values("mean_abs_shap", ascending=True),
            x="mean_abs_shap",
            y="description",
            orientation="h",
            text=shap_importance.sort_values("mean_abs_shap", ascending=True)["mean_abs_shap"].round(3),
            title="Top SHAP Drivers of Power User Status",
            color="mean_abs_shap",
            color_continuous_scale=[C["amber"], C["teal"]],
        )
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        style(fig)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Project Report":
    best_model = art["best_name"]
    best_auc = art["results"]["ROC-AUC"].max()
    power_rate = df["power_user"].mean()
    avg_features = df["features_adopted"].mean()
    median_ttfv = df["ttfv_days"].median()

    st.markdown(
        f"""
<div class="ds-card">
    <div class="sec-title">Executive Summary</div>
    <p>
        CareFlow is a <strong>Product Data Analyst portfolio project</strong> designed to show how
        healthcare product analytics, interpretable machine learning, and feature adoption analysis
        can support better product decisions.
    </p>
    <ul>
        <li><strong>{power_rate:.1%}</strong> of users are classified as power users</li>
        <li>Users adopt an average of <strong>{avg_features:.1f}</strong> features</li>
        <li>Median time-to-first-value is <strong>{int(median_ttfv)} days</strong></li>
        <li>Best-performing model: <strong>{best_model}</strong> with ROC-AUC <strong>{best_auc:.3f}</strong></li>
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="ds-card">
    <div class="sec-title">What This Project Demonstrates</div>
    <ul>
        <li>Feature adoption analysis</li>
        <li>Behavior and engagement segmentation</li>
        <li>Product KPI tracking</li>
        <li>Predictive modeling for user engagement</li>
        <li>Interpretable ML using SHAP</li>
        <li>Decision-focused analytics communication</li>
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )