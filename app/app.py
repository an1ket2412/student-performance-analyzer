"""Streamlit Dashboard: Student Performance Analyzer.

Provides early academic risk identification, cohort visual analytics,
and individualized risk probability assessment using the trained ML pipeline.
"""

from pathlib import Path
import sys
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cleaning import clean_student_data, load_raw_data
from src.features import (
    ALL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    engineer_features,
    get_feature_target_split,
)
from src.training import load_model

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563eb;
    }
    .risk-low {
        color: #16a34a;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .risk-mod {
        color: #d97706;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .risk-high {
        color: #dc2626;
        font-weight: 700;
        font-size: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Load cleaned student dataset, falling back to cleaning raw data if necessary."""
    processed_path = PROJECT_ROOT / "data" / "processed" / "student_clean.csv"
    raw_path = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"

    if processed_path.is_file():
        return pd.read_csv(processed_path)
    elif raw_path.is_file():
        raw_df = load_raw_data(raw_path)
        clean_df = clean_student_data(raw_df)
        clean_df.to_csv(processed_path, index=False)
        return clean_df
    else:
        st.error(
            "Dataset not found. Please place 'student-mat.csv' in 'data/raw/' "
            "as described in 'docs/data_source.md'."
        )
        st.stop()


@st.cache_resource
def get_pipeline():
    """Load trained Scikit-learn Pipeline from serialized joblib file."""
    model_path = PROJECT_ROOT / "models" / "final_model.joblib"
    if not model_path.is_file():
        st.error(
            "Trained model not found at 'models/final_model.joblib'. "
            "Please run 'python src/training.py' to train and serialize the model."
        )
        st.stop()
    return load_model(model_path)


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🎓 Student Performance Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Early Academic Risk Identification & Intervention Dashboard (HCL Career Shaper)</div>',
    unsafe_allow_html=True,
)

# Load data and pipeline
df_all = get_dataset()
pipeline = get_pipeline()

# -----------------------------------------------------------------------------
# Sidebar: Cohort Filters
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Cohort Filter Controls")
st.sidebar.caption("Filter aggregate cohort analytics:")

school_filter = st.sidebar.multiselect(
    "School",
    options=sorted(df_all["school"].unique()),
    default=sorted(df_all["school"].unique()),
    help="GP: Gabriel Pereira, MS: Mousinho da Silveira",
)

study_filter = st.sidebar.multiselect(
    "Study Time (Weekly)",
    options=sorted(df_all["studytime"].unique()),
    default=sorted(df_all["studytime"].unique()),
    format_func=lambda x: {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}.get(x, str(x)),
)

sex_filter = st.sidebar.multiselect(
    "Student Gender",
    options=sorted(df_all["sex"].unique()),
    default=sorted(df_all["sex"].unique()),
    format_func=lambda x: "Female" if x == "F" else "Male",
)

address_filter = st.sidebar.multiselect(
    "Address Type",
    options=sorted(df_all["address"].unique()),
    default=sorted(df_all["address"].unique()),
    format_func=lambda x: "Urban" if x == "U" else "Rural",
)

# Filter dataframe
filtered_df = df_all[
    (df_all["school"].isin(school_filter))
    & (df_all["studytime"].isin(study_filter))
    & (df_all["sex"].isin(sex_filter))
    & (df_all["address"].isin(address_filter))
]

if filtered_df.empty:
    st.warning("No student records match the active filter criteria. Please broaden your selection.")
    st.stop()

# -----------------------------------------------------------------------------
# Section 1: KPI Cards
# -----------------------------------------------------------------------------
total_students = len(filtered_df)
pass_count = int(filtered_df["passed"].sum())
at_risk_count = total_students - pass_count
pass_rate = pass_count / total_students if total_students > 0 else 0.0
avg_absences = filtered_df["absences"].mean()

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
with kpi_c1:
    st.metric(label="Total Cohort Students", value=f"{total_students}")
with kpi_c2:
    st.metric(label="Pass Rate (G3 >= 10)", value=f"{pass_rate:.1%}")
with kpi_c3:
    st.metric(label="At-Risk Students (G3 < 10)", value=f"{at_risk_count}", delta=f"-{(1-pass_rate):.1%}", delta_color="inverse")
with kpi_c4:
    st.metric(label="Average Absences", value=f"{avg_absences:.1f} days")

st.markdown("---")

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Cohort Visual Analytics",
    "🔮 Individual Risk Assessment",
    "ℹ️ Model Integrity & Info",
])

# -----------------------------------------------------------------------------
# TAB 1: Visual Analytics
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Cohort Trends & Exploratory Findings")

    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        st.markdown("**1. Pass Rate by Weekly Study Time**")
        study_grp = filtered_df.groupby("studytime")["passed"].agg(["mean", "count"]).reset_index()
        study_grp["study_label"] = study_grp["studytime"].map(
            {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}
        )
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        sns.barplot(data=study_grp, x="study_label", y="mean", palette="Blues_d", ax=ax1)
        ax1.set_ylim(0, 1.0)
        ax1.set_xlabel("Weekly Study Time")
        ax1.set_ylabel("Pass Rate")
        for i, row in study_grp.iterrows():
            ax1.text(i, row["mean"] + 0.03, f"{row['mean']:.1%}", ha="center", weight="bold")
        st.pyplot(fig1)
        st.caption("Insight: Pass rate steadily climbs from <2 hours to >10 hours of study.")

    with r1_col2:
        st.markdown("**2. Overall Cohort Academic Standing (Pass vs. At-Risk)**")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        counts = [pass_count, at_risk_count]
        labels = [f"Passing ({pass_count})", f"At-Risk ({at_risk_count})"]
        colors = ["#2ecc71", "#e74c3c"]
        ax2.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0.04, 0))
        st.pyplot(fig2)
        st.caption("Insight: Baseline class distribution displays ~2:1 pass-to-fail ratio.")

    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        st.markdown("**3. Absences Distribution by Outcome**")
        fig3, ax3 = plt.subplots(figsize=(6, 3.8))
        sns.boxplot(data=filtered_df, x="passed", y="absences", palette=["#e74c3c", "#2ecc71"], ax=ax3)
        ax3.set_xticklabels(["At-Risk (<10)", "Passing (>=10)"])
        ax3.set_xlabel("Outcome")
        ax3.set_ylabel("School Absences (Days)")
        st.pyplot(fig3)
        st.caption("Insight: Failing cohort features an extended upper tail of chronic absenteeism.")

    with r2_col2:
        st.markdown("**4. Impact of Past Course Failures on Pass Rate**")
        fig4, ax4 = plt.subplots(figsize=(6, 3.8))
        fail_grp = filtered_df.groupby("failures")["passed"].agg(["mean", "count"]).reset_index()
        sns.barplot(data=fail_grp, x="failures", y="mean", palette="Reds_r", ax=ax4)
        ax4.set_ylim(0, 1.0)
        ax4.set_xlabel("Past Failures Count (0 to 3)")
        ax4.set_ylabel("Pass Rate")
        for i, row in fail_grp.iterrows():
            ax4.text(i, row["mean"] + 0.03, f"{row['mean']:.1%}", ha="center", weight="bold")
        st.pyplot(fig4)
        st.caption("Insight: A single prior failure reduces pass likelihood by ~34 percentage points.")


# -----------------------------------------------------------------------------
# TAB 2: Individual Risk Prediction
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Individual Student Early-Warning Risk Assessment")
    st.markdown(
        "Enter pre-outcome student characteristics to evaluate academic failure probability. "
        "**Notice:** Period grades (*G1, G2*) are strictly excluded to ensure legitimate early warning."
    )

    # Preset quick-load buttons
    st.markdown("##### Quick Fill with Representative Profiles:")
    p_col1, p_col2, p_col3 = st.columns(3)
    preset_choice = None
    if p_col1.button("📋 Profile A: High Academic Risk"):
        preset_choice = "high_risk"
    if p_col2.button("📋 Profile B: Average Student"):
        preset_choice = "average"
    if p_col3.button("📋 Profile C: Low Academic Risk"):
        preset_choice = "low_risk"

    defaults = {
        "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 2, "Fedu": 2, "Mjob": "services", "Fjob": "services", "reason": "course",
        "guardian": "mother", "traveltime": 1, "studytime": 2, "failures": 0,
        "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "yes",
        "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
        "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4, "absences": 4
    }

    if preset_choice == "high_risk":
        defaults.update({
            "studytime": 1, "failures": 2, "Dalc": 3, "Walc": 4, "absences": 16,
            "higher": "no", "schoolsup": "yes", "famsup": "no", "internet": "no"
        })
    elif preset_choice == "low_risk":
        defaults.update({
            "studytime": 3, "failures": 0, "Dalc": 1, "Walc": 1, "absences": 2,
            "Medu": 4, "Fedu": 4, "higher": "yes", "internet": "yes"
        })

    with st.form("risk_prediction_form"):
        f_c1, f_c2, f_c3 = st.columns(3)

        with f_c1:
            st.markdown("##### 🏛️ Institutional & Demographics")
            in_school = st.selectbox("School", options=["GP", "MS"], index=["GP", "MS"].index(defaults["school"]))
            in_sex = st.selectbox("Gender", options=["F", "M"], index=["F", "M"].index(defaults["sex"]))
            in_age = st.number_input("Age", min_value=15, max_value=22, value=defaults["age"])
            in_address = st.selectbox("Address Type", options=["U", "R"], index=["U", "R"].index(defaults["address"]), format_func=lambda x: "Urban (U)" if x=="U" else "Rural (R)")
            in_famsize = st.selectbox("Family Size", options=["GT3", "LE3"], index=["GT3", "LE3"].index(defaults["famsize"]), format_func=lambda x: "> 3 Members" if x=="GT3" else "<= 3 Members")
            in_pstatus = st.selectbox("Parents Cohabitation", options=["T", "A"], index=["T", "A"].index(defaults["Pstatus"]), format_func=lambda x: "Living Together (T)" if x=="T" else "Apart (A)")
            in_traveltime = st.selectbox("Travel Time", options=[1, 2, 3, 4], index=defaults["traveltime"]-1, format_func=lambda x: {1: "< 15 min", 2: "15-30 min", 3: "30-60 min", 4: "> 60 min"}[x])

        with f_c2:
            st.markdown("##### 📚 Academic History & Effort")
            in_studytime = st.selectbox("Weekly Study Time", options=[1, 2, 3, 4], index=defaults["studytime"]-1, format_func=lambda x: {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}[x])
            in_failures = st.selectbox("Past Class Failures", options=[0, 1, 2, 3], index=defaults["failures"])
            in_absences = st.number_input("Total School Absences", min_value=0, max_value=93, value=defaults["absences"])
            in_schoolsup = st.selectbox("School Educational Support", options=["no", "yes"], index=["no", "yes"].index(defaults["schoolsup"]))
            in_famsup = st.selectbox("Family Educational Support", options=["no", "yes"], index=["no", "yes"].index(defaults["famsup"]))
            in_paid = st.selectbox("Extra Paid Tutoring", options=["no", "yes"], index=["no", "yes"].index(defaults["paid"]))
            in_higher = st.selectbox("Aspirations for Higher Ed", options=["yes", "no"], index=["yes", "no"].index(defaults["higher"]))

        with f_c3:
            st.markdown("##### 🏡 Family & Lifestyle")
            in_medu = st.selectbox("Mother's Education", options=[0, 1, 2, 3, 4], index=defaults["Medu"], format_func=lambda x: {0: "None", 1: "Primary (4th)", 2: "5th-9th Grade", 3: "Secondary", 4: "Higher Ed"}[x])
            in_fedu = st.selectbox("Father's Education", options=[0, 1, 2, 3, 4], index=defaults["Fedu"], format_func=lambda x: {0: "None", 1: "Primary (4th)", 2: "5th-9th Grade", 3: "Secondary", 4: "Higher Ed"}[x])
            in_mjob = st.selectbox("Mother's Job", options=["at_home", "health", "services", "teacher", "other"], index=["at_home", "health", "services", "teacher", "other"].index(defaults["Mjob"]))
            in_fjob = st.selectbox("Father's Job", options=["at_home", "health", "services", "teacher", "other"], index=["at_home", "health", "services", "teacher", "other"].index(defaults["Fjob"]))
            in_guardian = st.selectbox("Guardian", options=["mother", "father", "other"], index=["mother", "father", "other"].index(defaults["guardian"]))
            in_reason = st.selectbox("School Choice Reason", options=["course", "home", "reputation", "other"], index=["course", "home", "reputation", "other"].index(defaults["reason"]))
            in_internet = st.selectbox("Home Internet Access", options=["yes", "no"], index=["yes", "no"].index(defaults["internet"]))

        with st.expander("Additional Behavioral Factors (Health, Social, Alcohol)"):
            b_c1, b_c2, b_c3 = st.columns(3)
            with b_c1:
                in_activities = st.selectbox("Extracurricular Activities", options=["yes", "no"], index=["yes", "no"].index(defaults["activities"]))
                in_nursery = st.selectbox("Attended Nursery", options=["yes", "no"], index=["yes", "no"].index(defaults["nursery"]))
                in_romantic = st.selectbox("In a Romantic Relationship", options=["no", "yes"], index=["no", "yes"].index(defaults["romantic"]))
            with b_c2:
                in_famrel = st.slider("Family Relationship Quality", 1, 5, defaults["famrel"], help="1: Very bad, 5: Excellent")
                in_freetime = st.slider("Free Time After School", 1, 5, defaults["freetime"], help="1: Very low, 5: Very high")
                in_goout = st.slider("Going Out with Friends", 1, 5, defaults["goout"], help="1: Very low, 5: Very high")
            with b_c3:
                in_dalc = st.slider("Workday Alcohol Consumption", 1, 5, defaults["Dalc"], help="1: Very low, 5: Very high")
                in_walc = st.slider("Weekend Alcohol Consumption", 1, 5, defaults["Walc"], help="1: Very low, 5: Very high")
                in_health = st.slider("Current Health Status", 1, 5, defaults["health"], help="1: Very bad, 5: Very good")

        submit_btn = st.form_submit_button("🔍 Compute Early Risk Prediction", use_container_width=True)

    if submit_btn:
        # Construct raw observation matching pipeline inputs
        raw_input_dict = {
            "school": in_school, "sex": in_sex, "age": in_age, "address": in_address,
            "famsize": in_famsize, "Pstatus": in_pstatus, "Medu": in_medu, "Fedu": in_fedu,
            "Mjob": in_mjob, "Fjob": in_fjob, "reason": in_reason, "guardian": in_guardian,
            "traveltime": in_traveltime, "studytime": in_studytime, "failures": in_failures,
            "schoolsup": in_schoolsup, "famsup": in_famsup, "paid": in_paid,
            "activities": in_activities, "nursery": in_nursery, "higher": in_higher,
            "internet": in_internet, "romantic": in_romantic, "famrel": in_famrel,
            "freetime": in_freetime, "goout": in_goout, "Dalc": in_dalc, "Walc": in_walc,
            "health": in_health, "absences": in_absences
        }
        input_df = pd.DataFrame([raw_input_dict])

        # EXACT SAME FEATURE ENGINEERING AS TRAINING
        engineered_row = engineer_features(input_df)

        # EXACT SAME SAVED PIPELINE INFERENCE
        prob_pass = float(pipeline.predict_proba(engineered_row)[0, 1])
        prob_risk = 1.0 - prob_pass

        # Risk Tier Classification
        # Documented Thresholds:
        # Low Risk: < 35%
        # Moderate Risk: 35% - 55%
        # High Risk: >= 55%
        if prob_risk >= 0.55:
            risk_category = "High Academic Risk"
            risk_css = "risk-high"
            action_guidance = (
                "⚠️ **Urgent Intervention Recommended:** Assign direct mentor check-ins, "
                "schedule weekly supplemental tutoring sessions, and engage family coordinator."
            )
        elif prob_risk >= 0.35:
            risk_category = "Moderate Academic Risk"
            risk_css = "risk-mod"
            action_guidance = (
                "🔍 **Proactive Monitoring:** Monitor homework completion closely, recommend study groups, "
                "and conduct bi-weekly academic progress reviews."
            )
        else:
            risk_category = "Low Academic Risk"
            risk_css = "risk-low"
            action_guidance = (
                "✅ **Satisfactory Trajectory:** Continue current academic program and provide enrichment encouragement."
            )

        st.markdown("### Risk Evaluation Results")
        res_c1, res_c2, res_c3 = st.columns(3)

        with res_c1:
            st.metric(label="Predicted Class", value="Pass" if prob_pass >= 0.50 else "At-Risk")
        with res_c2:
            st.metric(label="Calculated Risk Probability", value=f"{prob_risk:.1%}")
        with res_c3:
            st.markdown(f"**Risk Category:** <div class='{risk_css}'>{risk_category}</div>", unsafe_allow_html=True)

        st.progress(float(prob_risk))
        st.info(action_guidance)


# -----------------------------------------------------------------------------
# TAB 3: Model Integrity & Info
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Model Governance, Architecture & Leakage Isolation")

    st.markdown("""
    ### 🛡️ Strict Zero-Leakage Architecture
    This application enforces strict isolation between pre-outcome variables and final grades:
    - **Excluded Period Grades:** Neither First Period Grade ($G1$) nor Second Period Grade ($G2$) are utilized.
    - **Deployment Realism:** Teachers receive predictions at term onset, well before mid-term evaluations.
    - **Engineering Consistency:** The Streamlit app imports the exact `engineer_features()` routine from `src/features.py` and feeds data through the saved `models/final_model.joblib` pipeline without duplication.

    ### 📈 Model Details
    - **Architecture:** Standardized Regularized Balanced Logistic Regression.
    - **Optimization Objective:** Maximizing Recall for Fail/At-Risk Class.
    - **Risk Threshold Tiers:**
      - **Low Risk:** Probability $< 0.35$
      - **Moderate Risk:** $0.35 \le \text{Probability} < 0.55$
      - **High Risk:** $\text{Probability} \ge 0.55$
    """)

    st.markdown("##### Transformed Feature Importances (Top Drivers of Risk vs. Pass):")
    if hasattr(pipeline.named_steps["classifier"], "coef_"):
        classifier = pipeline.named_steps["classifier"]
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        coefs = classifier.coef_[0]
        imp_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        imp_df["association"] = imp_df["coefficient"].apply(lambda c: "Associated with Pass" if c > 0 else "Associated with Risk")
        imp_df["abs_coef"] = imp_df["coefficient"].abs()
        top_imp = imp_df.sort_values(by="abs_coef", ascending=False).head(12)

        fig_imp, ax_imp = plt.subplots(figsize=(10, 4.5))
        sns.barplot(data=top_imp, x="coefficient", y="feature", hue="association", palette={"Associated with Pass": "#2ecc71", "Associated with Risk": "#e74c3c"}, ax=ax_imp)
        ax_imp.axvline(0, color="black", linestyle="--", alpha=0.7)
        ax_imp.set_title("Top 12 Model Feature Coefficients (Non-Causal Statistical Association)")
        st.pyplot(fig_imp)
