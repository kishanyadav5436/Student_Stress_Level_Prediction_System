import streamlit as st
import pandas as pd
import joblib
import numpy as np

from utils.recommendations import get_all_recommendations
from utils.charts import (
    build_stress_gauge,
    build_probability_chart,
    build_feature_importance_chart,
    build_history_chart,
)

try:
    from utils.pdf_report import generate_pdf_report
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = joblib.load("student_stress_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# -------------------- GLOBAL CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1140 35%, #302b63 70%, #24243e 100%);
    min-height: 100vh;
}

/* Hide Streamlit chrome */
#MainMenu {display:none !important;}
footer {display:none !important;}
header {display:none !important;}
div[data-testid="stToolbar"] {display:none !important;}
div[data-testid="stDecoration"] {display:none !important;}
div[data-testid="stStatusWidget"] {display:none !important;}
[data-testid="stHeader"] {display:none !important;}

/* Main layout */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 32px 40px;
    margin-bottom: 28px;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    text-align: center;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    letter-spacing: -1px;
    margin: 0;
    background: linear-gradient(90deg, #fff 0%, #c4b5fd 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: rgba(255,255,255,0.6);
    font-size: 16px;
    font-weight: 400;
    margin-top: 8px;
}

/* Glass Cards */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(16px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.card-title {
    font-size: 16px;
    font-weight: 700;
    color: rgba(255,255,255,0.9);
    letter-spacing: 0.5px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Input labels */
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
}

/* Input widgets */
div[data-baseweb="select"] > div,
.stNumberInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: white !important;
    transition: border-color 0.2s;
}
div[data-baseweb="select"] > div:hover,
.stNumberInput input:focus {
    border-color: rgba(196,181,253,0.6) !important;
}

.stNumberInput button {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
}

/* Slider track */
.stSlider [data-baseweb="slider"] > div > div {
    background: rgba(255,255,255,0.15) !important;
}

/* Predict button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 50%, #06b6d4 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 0 !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.4) !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 14px 32px rgba(124,58,237,0.55) !important;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 16px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: white;
}
.metric-label {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    margin-top: 4px;
}
.metric-status-good { border-top: 3px solid #4cd97b; }
.metric-status-warn { border-top: 3px solid #ffcc00; }
.metric-status-bad  { border-top: 3px solid #ff453a; }

/* Result banner */
.result-low    { background: linear-gradient(135deg,#064e3b,#065f46); border:1px solid #4cd97b44; }
.result-medium { background: linear-gradient(135deg,#78350f,#92400e); border:1px solid #ffcc0044; }
.result-high   { background: linear-gradient(135deg,#7f1d1d,#991b1b); border:1px solid #ff453a44; }

.result-banner {
    border-radius: 18px;
    padding: 24px 32px;
    margin-bottom: 20px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.3);
}
.result-emoji { font-size: 48px; }
.result-title { font-size: 26px; font-weight: 800; color: white; margin: 8px 0 4px; }
.result-sub   { font-size: 14px; color: rgba(255,255,255,0.75); }

/* Recommendations */
.rec-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13.5px;
    color: rgba(255,255,255,0.88);
    line-height: 1.6;
    transition: background 0.2s;
}
.rec-card:hover { background: rgba(255,255,255,0.08); }

/* History table */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1140 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.88) !important; }
section[data-testid="stSidebar"] .stAlert {
    background: rgba(124,58,237,0.15) !important;
    border: 1px solid rgba(124,58,237,0.4) !important;
    border-radius: 10px !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Download button */
div[data-testid="stDownloadButton"] > button {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(255,255,255,0.14) !important;
    border-color: rgba(196,181,253,0.5) !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 600 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(124,58,237,0.5) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🧠 Stress Predictor")
    st.success("ML-Powered Wellness Tool")
    st.markdown("---")

    st.markdown("#### 📌 Features Used")
    features_list = [
        ("🎓", "Student Type"),
        ("😴", "Sleep Hours"),
        ("📚", "Study Hours"),
        ("📱", "Social Media"),
        ("🏫", "Attendance"),
        ("😰", "Exam Pressure"),
        ("❤️", "Family Support"),
    ]
    for icon, name in features_list:
        st.markdown(f"{icon} &nbsp; {name}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📈 Session Stats")
    total_preds = len(st.session_state.history)
    if total_preds > 0:
        labels_done = [h["label"] for h in st.session_state.history]
        st.metric("Total Predictions", total_preds)
        most_common = max(set(labels_done), key=labels_done.count)
        colors_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        st.markdown(f"**Most Common:** {colors_map.get(most_common,'')} {most_common} Stress")
    else:
        st.info("No predictions yet. Fill the form and hit Predict!")

    st.markdown("---")
    st.markdown("#### 💡 Quick Tips")
    with st.expander("What does each level mean?"):
        st.markdown("""
        **🟢 Low Stress** — You're balanced and managing well.

        **🟡 Medium Stress** — Some factors need attention. Small changes help.

        **🔴 High Stress** — Seek support immediately and reduce load.
        """)

    st.markdown("---")
    st.caption("Built by Kishan Kumar • B.Tech CS")


# -------------------- HERO --------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🧠 Student Stress Level Predictor</div>
    <div class="hero-subtitle">Enter your daily habits and let Machine Learning assess your stress level with personalized insights</div>
</div>
""", unsafe_allow_html=True)


# -------------------- TABS --------------------
tab_predict, tab_history, tab_analytics = st.tabs(["🎯 Predict", "📋 History", "📊 Analytics"])


# ===========================================================
# TAB 1 — PREDICT
# ===========================================================
with tab_predict:
    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        # Student Type
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎓 Student Profile</div>', unsafe_allow_html=True)
        student_type = st.selectbox(
            "Student Type",
            ["School", "College", "Working_Student"],
            help="Select the type that best describes you"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Time Inputs
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⏱️ Daily Time Allocation</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sleep_hours = st.number_input(
                "😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5,
                help="Average hours of sleep per night"
            )
            study_hours = st.number_input(
                "📚 Study Hours", min_value=0.0, max_value=15.0, value=4.0, step=0.5,
                help="Average hours spent studying per day"
            )
        with c2:
            social_media = st.number_input(
                "📱 Social Media Hrs", min_value=0.0, max_value=15.0, value=2.0, step=0.5,
                help="Average hours on social media per day"
            )
            attendance = st.number_input(
                "🏫 Attendance %", min_value=0.0, max_value=100.0, value=75.0, step=1.0,
                help="Your class attendance percentage"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Pressure & Support
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎚️ Pressure & Support Levels</div>', unsafe_allow_html=True)
        exam_pressure = st.slider(
            "😰 Exam Pressure", 1, 10, 5,
            help="1 = Very Low, 10 = Extremely High"
        )
        family_support = st.slider(
            "❤️ Family Support", 1, 10, 5,
            help="1 = No support, 10 = Full support"
        )

        # Quick explainers
        with st.expander("ℹ️ Why do these factors matter?"):
            st.markdown("""
            | Factor | Why it matters |
            |--------|---------------|
            | Sleep | Insufficient sleep impairs memory, focus & emotional regulation |
            | Study Hours | Over-studying leads to burnout; under-studying causes anxiety |
            | Social Media | Excessive use disrupts sleep and increases comparison anxiety |
            | Attendance | Low attendance compounds learning gaps and increases pressure |
            | Exam Pressure | High pressure spikes cortisol and impairs performance |
            | Family Support | Strong support is a key protective factor against stress |
            """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Predict Button
        predict_clicked = st.button("🚀 Predict My Stress Level", use_container_width=True)

    # ---- RIGHT COLUMN: Results ----
    with col_result:
        student_type_map = {"School": 0, "College": 1, "Working_Student": 2}
        student_type_encoded = student_type_map[student_type]

        if predict_clicked:
            input_df = pd.DataFrame({
                "Student_Type": [student_type_encoded],
                "Sleep_Hours": [sleep_hours],
                "Study_Hours": [study_hours],
                "Social_Media_Hours": [social_media],
                "Attendance": [attendance],
                "Exam_Pressure": [exam_pressure],
                "Family_Support": [family_support]
            })

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]

            LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
            EMOJI_MAP = {0: "🟢", 1: "🟡", 2: "🔴"}
            FACE_MAP  = {0: "😊", 1: "😐", 2: "😟"}
            CSS_MAP   = {0: "result-low", 1: "result-medium", 2: "result-high"}
            SCORE_MAP = {0: proba[0]*100, 1: proba[1]*100, 2: proba[2]*100}
            SUB_MAP   = {
                0: "Keep maintaining your healthy routine!",
                1: "Take regular breaks and manage your time wisely.",
                2: "Reduce workload, sleep well and seek support."
            }

            label = LABEL_MAP[prediction]
            score = round(proba[prediction] * 100, 1)
            gauge_colors = {0: "#4cd97b", 1: "#ffcc00", 2: "#ff453a"}

            # Save to session history
            run_num = len(st.session_state.history) + 1
            st.session_state.history.append({
                "run": run_num,
                "label": label,
                "score": round(proba[prediction] * 100, 1),
                "sleep": sleep_hours,
                "study": study_hours,
                "social": social_media,
                "attendance": attendance,
                "exam_pressure": exam_pressure,
                "family_support": family_support,
                "student_type": student_type,
                "proba_low": round(proba[0]*100, 1),
                "proba_med": round(proba[1]*100, 1),
                "proba_high": round(proba[2]*100, 1),
            })

            # Get recommendations
            recs = get_all_recommendations(
                prediction, sleep_hours, study_hours, social_media,
                attendance, exam_pressure, family_support, student_type
            )

            st.session_state.last_result = {
                "prediction": prediction,
                "proba": proba,
                "label": label,
                "score": score,
                "recs": recs,
                "inputs": input_df.to_dict(),
            }

        # ---- Show Results (from session state so it persists) ----
        if st.session_state.last_result:
            r = st.session_state.last_result
            prediction = r["prediction"]
            proba = r["proba"]
            label = r["label"]
            score = r["score"]
            recs = r["recs"]

            LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
            EMOJI_MAP = {0: "🟢", 1: "🟡", 2: "🔴"}
            FACE_MAP  = {0: "😊", 1: "😐", 2: "😟"}
            CSS_MAP   = {0: "result-low", 1: "result-medium", 2: "result-high"}
            gauge_colors = {0: "#4cd97b", 1: "#ffcc00", 2: "#ff453a"}

            # Result Banner
            st.markdown(f"""
            <div class="result-banner {CSS_MAP[prediction]}">
                <div class="result-emoji">{FACE_MAP[prediction]}</div>
                <div class="result-title">{EMOJI_MAP[prediction]} {label.upper()} STRESS</div>
                <div class="result-sub">{["Keep maintaining your healthy routine!","Take regular breaks and manage your time wisely.","Reduce workload, sleep well and seek support."][prediction]}</div>
                <div style="margin-top:10px;font-size:13px;color:rgba(255,255,255,0.6);">Model confidence: <b style="color:white">{score}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Stress Gauge
            gauge_fig = build_stress_gauge(score, label.upper(), gauge_colors[prediction])
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

            # Probability Chart
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Prediction Confidence</div>', unsafe_allow_html=True)
            prob_fig = build_probability_chart(list(proba))
            st.plotly_chart(prob_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            # Metric Cards
            sleep_status = "good" if sleep_hours >= 7 else ("warn" if sleep_hours >= 6 else "bad")
            study_status = "good" if 3 <= study_hours <= 8 else ("warn" if study_hours <= 10 else "bad")
            social_status = "good" if social_media <= 2 else ("warn" if social_media <= 4 else "bad")
            attend_status = "good" if attendance >= 75 else ("warn" if attendance >= 60 else "bad")

            st.markdown("**📌 Input Snapshot**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            metrics = [
                (mc1, "😴", f"{sleep_hours}h", "Sleep", sleep_status),
                (mc2, "📚", f"{study_hours}h", "Study", study_status),
                (mc3, "📱", f"{social_media}h", "Social", social_status),
                (mc4, "🏫", f"{attendance}%", "Attend", attend_status),
            ]
            for col, icon, val, lbl, status in metrics:
                with col:
                    st.markdown(f"""
                    <div class="metric-card metric-status-{status}">
                        <div style="font-size:20px">{icon}</div>
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Recommendations
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">💡 {recs["title"]}</div>', unsafe_allow_html=True)

            st.markdown("**General Wellness Tips:**")
            for tip in recs["general"]:
                st.markdown(f'<div class="rec-card">{tip}</div>', unsafe_allow_html=True)

            if recs["feature_tips"]:
                st.markdown("<br>**Personalized Tips for You:**")
                for tip in recs["feature_tips"]:
                    st.markdown(f'<div class="rec-card">{tip}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Download buttons
            st.markdown("---")
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                # CSV Download
                hist_df = pd.DataFrame(st.session_state.history)
                csv_bytes = hist_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download History (CSV)",
                    data=csv_bytes,
                    file_name="stress_prediction_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with dl_col2:
                # PDF Download
                if PDF_AVAILABLE:
                    try:
                        pdf_bytes = generate_pdf_report(
                            prediction_label=label,
                            stress_score=score,
                            proba=proba,
                            sleep_hours=sleep_hours,
                            study_hours=study_hours,
                            social_media=social_media,
                            attendance=attendance,
                            exam_pressure=exam_pressure,
                            family_support=family_support,
                            student_type=student_type,
                            recommendations=recs
                        )
                        st.download_button(
                            "📄 Download Report (PDF)",
                            data=pdf_bytes,
                            file_name="stress_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning(f"PDF generation failed: {e}")
                else:
                    st.info("PDF export requires `fpdf2`. Add it to requirements.txt.")

        else:
            st.markdown('<div class="glass-card" style="text-align:center;padding:60px 20px;">', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:64px;margin-bottom:16px">🧠</div>
            <div style="color:rgba(255,255,255,0.8);font-size:18px;font-weight:600;">Ready to Predict</div>
            <div style="color:rgba(255,255,255,0.45);font-size:14px;margin-top:8px;">
                Fill in your details on the left and click<br>
                <b style="color:rgba(196,181,253,0.9)">Predict My Stress Level</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================
# TAB 2 — HISTORY
# ===========================================================
with tab_history:
    if not st.session_state.history:
        st.info("📭 No predictions yet. Go to the **Predict** tab and make your first prediction!")
    else:
        st.markdown("### 📋 Prediction History")
        st.caption("All predictions made in this session. Use 'what-if' mode to compare scenarios.")

        hist_df = pd.DataFrame(st.session_state.history)

        # Summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Runs", len(hist_df))
        with mc2:
            low_count = (hist_df["label"] == "Low").sum()
            st.metric("🟢 Low Stress", low_count)
        with mc3:
            med_count = (hist_df["label"] == "Medium").sum()
            st.metric("🟡 Medium Stress", med_count)
        with mc4:
            high_count = (hist_df["label"] == "High").sum()
            st.metric("🔴 High Stress", high_count)

        st.markdown("---")

        # History table
        display_df = hist_df[["run", "label", "score", "sleep", "study",
                               "social", "attendance", "exam_pressure",
                               "family_support", "student_type",
                               "proba_low", "proba_med", "proba_high"]].copy()
        display_df.columns = [
            "Run #", "Prediction", "Confidence %", "Sleep Hrs", "Study Hrs",
            "Social Hrs", "Attendance %", "Exam Pressure", "Family Support",
            "Student Type", "Low %", "Medium %", "High %"
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Trend chart
        st.markdown("### 📈 Stress Confidence Trend")
        trend_fig = build_history_chart(st.session_state.history)
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()


# ===========================================================
# TAB 3 — ANALYTICS
# ===========================================================
with tab_analytics:
    st.markdown("### 📊 Model Analytics")

    # Feature Importance

    feature_names = [
        "Student Type", "Sleep Hours", "Study Hours",
        "Social Media Hrs", "Attendance %",
        "Exam Pressure", "Family Support"
    ]

    has_importance = hasattr(model, "feature_importances_")
    has_coef = hasattr(model, "coef_")

    if has_importance:
        importances = model.feature_importances_
        chart_title = "🌳 Feature Importances (from Model)"
    elif has_coef:
        importances = np.abs(model.coef_).mean(axis=0)
        chart_title = "📐 Feature Coefficients (absolute, from Model)"
    else:
        # Fallback: equal weights
        importances = np.ones(len(feature_names)) / len(feature_names)
        chart_title = "📊 Feature Weights (equal — model type not supported)"

    a1, a2 = st.columns([1.2, 0.8])

    with a1:
        st.markdown(f"#### {chart_title}")
        imp_fig = build_feature_importance_chart(feature_names, list(importances))
        st.plotly_chart(imp_fig, use_container_width=True, config={"displayModeBar": False})

    with a2:
        st.markdown("#### 🔍 Importance Breakdown")
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": [round(i, 4) for i in importances],
            "Rank": range(1, len(feature_names)+1)
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        imp_df["Rank"] = range(1, len(imp_df)+1)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

        top_feature = imp_df.iloc[0]["Feature"]
        st.markdown(f"""
        <div class="glass-card" style="margin-top:16px;">
            <div class="card-title">💡 Key Insight</div>
            <div style="color:rgba(255,255,255,0.85);font-size:14px;">
                <b style="color:#c4b5fd">{top_feature}</b> is the most influential factor
                in predicting student stress. Focus on improving this area first.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Model Info
    st.markdown("---")
    st.markdown("#### 🤖 Model Information")
    info1, info2, info3 = st.columns(3)
    with info1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">🧠 Algorithm</div>
            <div style="color:rgba(255,255,255,0.85);font-size:15px;font-weight:600;">{type(model).__name__}</div>
        </div>
        """, unsafe_allow_html=True)
    with info2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">📐 Scaler</div>
            <div style="color:rgba(255,255,255,0.85);font-size:15px;font-weight:600;">{type(scaler).__name__}</div>
        </div>
        """, unsafe_allow_html=True)
    with info3:
        n_classes = len(model.classes_) if hasattr(model, "classes_") else 3
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">🎯 Output Classes</div>
            <div style="color:rgba(255,255,255,0.85);font-size:15px;font-weight:600;">{n_classes} (Low / Medium / High)</div>
        </div>
        """, unsafe_allow_html=True)

    # Stress Level Reference
    st.markdown("---")
    st.markdown("#### 📚 Stress Level Reference")
    r1, r2, r3 = st.columns(3)
    refs = [
        (r1, "🟢 Low Stress", "#064e3b", "#4cd97b",
         "Sleep ≥7h • Study 3–6h • Social ≤2h • Attendance ≥80% • Exam Pressure ≤4"),
        (r2, "🟡 Medium Stress", "#78350f", "#ffcc00",
         "Sleep 6–7h • Study 6–9h • Social 2–4h • Attendance 60–79% • Exam Pressure 5–7"),
        (r3, "🔴 High Stress", "#7f1d1d", "#ff453a",
         "Sleep <6h • Study >9h • Social >4h • Attendance <60% • Exam Pressure ≥8"),
    ]
    for col, title, bg, border, desc in refs:
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{bg}99,{bg}44);
                        border:1px solid {border}44;border-radius:16px;padding:20px;text-align:center;">
                <div style="font-size:20px;font-weight:700;color:white;margin-bottom:8px;">{title}</div>
                <div style="font-size:12px;color:rgba(255,255,255,0.7);">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
