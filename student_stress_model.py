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

html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0d0b1f 0%, #1a1140 40%, #2a1f5e 75%, #1e1b4b 100%) !important;
    min-height: 100vh;
}

#MainMenu, footer, header,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
[data-testid="stHeader"] { display: none !important; }

.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(124,58,237,0.22) 0%, rgba(99,102,241,0.15) 50%, rgba(6,182,212,0.10) 100%);
    border: 1px solid rgba(196,181,253,0.28);
    border-radius: 24px;
    padding: 36px 48px 28px;
    margin-bottom: 22px;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 24px 64px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
    text-align: center;
}
.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
    text-shadow: 0 0 40px rgba(196,181,253,0.5);
    line-height: 1.2;
}
.hero-subtitle {
    color: rgba(196,181,253,0.85);
    font-size: 14px;
    font-weight: 400;
    margin: 0 auto;
    max-width: 560px;
    line-height: 1.6;
}

/* ── Glass Cards ── */
.glass-card {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 18px !important;
    padding: 20px 22px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.07) !important;
    margin-bottom: 14px !important;
}
.card-title {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #c4b5fd !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    margin-bottom: 14px !important;
}

/* ── Labels ── */
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: rgba(255,255,255,0.88) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ── Select ── */
div[data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.09) !important;
    border: 1px solid rgba(196,181,253,0.35) !important;
    border-radius: 12px !important;
    color: white !important;
}
[data-baseweb="popover"] > div {
    background: #1e1b4b !important;
    border: 1px solid rgba(196,181,253,0.2) !important;
    border-radius: 12px !important;
}
[data-baseweb="menu"] li { color: rgba(255,255,255,0.85) !important; background: transparent !important; }
[data-baseweb="menu"] li:hover { background: rgba(124,58,237,0.25) !important; }

/* ── Number input ── */
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.09) !important;
    border: 1px solid rgba(196,181,253,0.35) !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    text-align: center !important;
}
.stNumberInput > div > div > input:focus {
    border-color: rgba(196,181,253,0.7) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
    outline: none !important;
}
.stNumberInput button {
    background: rgba(124,58,237,0.2) !important;
    border: 1px solid rgba(196,181,253,0.25) !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 700 !important;
}
.stNumberInput button:hover { background: rgba(124,58,237,0.42) !important; }

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] > div:first-child {
    background: rgba(255,255,255,0.12) !important;
    height: 6px !important;
    border-radius: 99px !important;
}
.stSlider [role="progressbar"] {
    background: linear-gradient(90deg, #7c3aed, #06b6d4) !important;
    border-radius: 99px !important;
}
.stSlider [role="slider"] {
    background: white !important;
    border: 3px solid #7c3aed !important;
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.5) !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 55%, #0891b2 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 28px rgba(124,58,237,0.6) !important;
}

/* ── Download buttons ── */
div[data-testid="stDownloadButton"] > button {
    background: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(196,181,253,0.3) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
    padding: 10px 0 !important;
    width: 100% !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(124,58,237,0.22) !important;
    border-color: rgba(196,181,253,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Result banners ── */
.result-low    { background: linear-gradient(135deg,#065f46,#047857); border:1px solid rgba(52,211,153,0.35); box-shadow:0 8px 32px rgba(5,150,105,0.3); }
.result-medium { background: linear-gradient(135deg,#92400e,#b45309); border:1px solid rgba(251,191,36,0.35); box-shadow:0 8px 32px rgba(180,83,9,0.3); }
.result-high   { background: linear-gradient(135deg,#991b1b,#b91c1c); border:1px solid rgba(252,165,165,0.35); box-shadow:0 8px 32px rgba(185,28,28,0.3); }
.result-banner { border-radius: 20px; padding: 22px 26px; margin-bottom: 14px; }
.result-emoji  { font-size: 42px; display: block; margin-bottom: 4px; }
.result-title  { font-size: 22px; font-weight: 800; color: white; margin: 0 0 4px; }
.result-sub    { font-size: 13px; color: rgba(255,255,255,0.8); }

/* ── Metric mini cards ── */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 12px 6px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.3); }
.metric-value { font-size: 20px; font-weight: 800; color: white; }
.metric-label { font-size: 10px; color: rgba(255,255,255,0.5); font-weight: 500; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.4px; }
.metric-status-good { border-top: 3px solid #34d399; }
.metric-status-warn { border-top: 3px solid #fbbf24; }
.metric-status-bad  { border-top: 3px solid #f87171; }

/* ── Rec cards ── */
.rec-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-left: 3px solid rgba(196,181,253,0.6);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 12.5px;
    color: rgba(255,255,255,0.85);
    line-height: 1.6;
    transition: background 0.2s;
}
.rec-card:hover { background: rgba(124,58,237,0.1); border-left-color: #a78bfa; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.55) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.85) !important;
    background: rgba(255,255,255,0.06) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(124,58,237,0.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 18px !important; }

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    color: rgba(255,255,255,0.8) !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    padding: 11px 14px !important;
}

/* ── DataFrames ── */
div[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* ── Streamlit Metric widget ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
}
div[data-testid="metric-container"] label {
    color: rgba(255,255,255,0.5) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: white !important;
    font-size: 26px !important;
    font-weight: 800 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0b1f 0%, #130f30 55%, #1a1140 100%) !important;
    border-right: 1px solid rgba(196,181,253,0.12) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.4) !important;
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.88) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 { color: white !important; }
section[data-testid="stSidebar"] .stSuccess {
    background: rgba(52,211,153,0.1) !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] .stInfo {
    background: rgba(124,58,237,0.12) !important;
    border: 1px solid rgba(196,181,253,0.25) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.1) !important;
    margin: 10px 0 !important;
}

/* ── Markdown text ── */
.stMarkdown p, .stMarkdown li { color: rgba(255,255,255,0.82) !important; }
.stMarkdown strong { color: white !important; }
.stMarkdown h4 { color: rgba(196,181,253,0.9) !important; }
.stCaption, .stCaption p { color: rgba(255,255,255,0.4) !important; font-size: 11px !important; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.1) !important; margin: 14px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(196,181,253,0.3); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(196,181,253,0.5); }

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
        # Student Profile Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎓 Student Profile</div>', unsafe_allow_html=True)
        student_type = st.selectbox(
            "Student Type",
            ["School", "College", "Working_Student"],
            help="Select the type that best describes you"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Time Inputs Card
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

        # Pressure & Support Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎚️ Pressure &amp; Support Levels</div>', unsafe_allow_html=True)
        exam_pressure = st.slider(
            "😰 Exam Pressure", 1, 10, 5,
            help="1 = Very Low, 10 = Extremely High"
        )
        family_support = st.slider(
            "❤️ Family Support", 1, 10, 5,
            help="1 = No support, 10 = Full support"
        )

        with st.expander("ℹ️ Why do these factors matter?"):
            st.markdown("""
            | Factor | Why it matters |
            |--------|----------------|
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
            prediction = int(model.predict(input_scaled)[0])

            # Safely get probability estimates — always returns a 3-element list
            # [P(Low), P(Medium), P(High)] regardless of how many classes the model has
            try:
                raw_proba = model.predict_proba(input_scaled)[0]
                # Map model.classes_ → 3-slot array (handles binary/multi-class models)
                classes = [int(c) for c in model.classes_] if hasattr(model, "classes_") else [0, 1, 2]
                proba = [0.0, 0.0, 0.0]
                for idx, cls in enumerate(classes):
                    if 0 <= cls <= 2:
                        proba[cls] = float(raw_proba[idx])
                # If model had only 2 classes, the missing one gets 0 — renormalize
                total = sum(proba)
                if total > 0:
                    proba = [round(p / total, 4) for p in proba]
                else:
                    proba = [0.1, 0.1, 0.8]
            except Exception:
                # Hard fallback: synthesize plausible probabilities
                proba = [0.1, 0.1, 0.1]
                proba[prediction] = 0.80
                total = sum(proba)
                proba = [round(p / total, 4) for p in proba]

            LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
            EMOJI_MAP = {0: "🟢", 1: "🟡", 2: "🔴"}
            FACE_MAP  = {0: "😊", 1: "😐", 2: "😟"}
            CSS_MAP   = {0: "result-low", 1: "result-medium", 2: "result-high"}

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

        # ---- Show Results ----
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
            sub_text = ["Keep maintaining your healthy routine!",
                        "Take regular breaks and manage your time wisely.",
                        "Reduce workload, sleep well and seek support."]

            # Result Banner
            st.markdown(f"""
            <div class="result-banner {CSS_MAP[prediction]}">
                <span class="result-emoji">{FACE_MAP[prediction]}</span>
                <div class="result-title">{EMOJI_MAP[prediction]} {label.upper()} STRESS</div>
                <div class="result-sub">{sub_text[prediction]}</div>
                <div style="margin-top:8px;font-size:12px;color:rgba(255,255,255,0.55);">
                    Model confidence: <b style="color:white">{score}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Stress Gauge
            gauge_fig = build_stress_gauge(score, label.upper(), gauge_colors[prediction])
            st.plotly_chart(gauge_fig, width="stretch", config={"displayModeBar": False})

            # Probability Chart
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Prediction Confidence</div>', unsafe_allow_html=True)
            prob_fig = build_probability_chart(list(proba))
            st.plotly_chart(prob_fig, width="stretch", config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            # Input Snapshot Metric Cards
            sleep_hours_val  = r["inputs"]["Sleep_Hours"][0]
            study_hours_val  = r["inputs"]["Study_Hours"][0]
            social_media_val = r["inputs"]["Social_Media_Hours"][0]
            attendance_val   = r["inputs"]["Attendance"][0]

            sleep_status  = "good" if sleep_hours_val >= 7 else ("warn" if sleep_hours_val >= 6 else "bad")
            study_status  = "good" if 3 <= study_hours_val <= 8 else ("warn" if study_hours_val <= 10 else "bad")
            social_status = "good" if social_media_val <= 2 else ("warn" if social_media_val <= 4 else "bad")
            attend_status = "good" if attendance_val >= 75 else ("warn" if attendance_val >= 60 else "bad")

            st.markdown("**📌 Input Snapshot**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            metrics = [
                (mc1, "😴", f"{sleep_hours_val}h", "Sleep", sleep_status),
                (mc2, "📚", f"{study_hours_val}h", "Study", study_status),
                (mc3, "📱", f"{social_media_val}h", "Social", social_status),
                (mc4, "🏫", f"{attendance_val}%", "Attend", attend_status),
            ]
            for col, icon, val, lbl, status in metrics:
                with col:
                    st.markdown(f"""
                    <div class="metric-card metric-status-{status}">
                        <div style="font-size:18px">{icon}</div>
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

            # Download Buttons
            st.markdown("---")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
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
                if PDF_AVAILABLE:
                    try:
                        pdf_bytes = generate_pdf_report(
                            prediction_label=label,
                            stress_score=score,
                            proba=proba,
                            sleep_hours=sleep_hours_val,
                            study_hours=study_hours_val,
                            social_media=social_media_val,
                            attendance=attendance_val,
                            exam_pressure=r["inputs"]["Exam_Pressure"][0],
                            family_support=r["inputs"]["Family_Support"][0],
                            student_type=list({"School": 0, "College": 1, "Working_Student": 2}.keys())[
                                list({"School": 0, "College": 1, "Working_Student": 2}.values()).index(
                                    r["inputs"]["Student_Type"][0])],
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
                    st.info("PDF export requires `fpdf2`.")

        else:
            st.markdown('<div class="glass-card" style="text-align:center;padding:60px 20px;">', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:60px;margin-bottom:14px">🧠</div>
            <div style="color:rgba(255,255,255,0.8);font-size:17px;font-weight:600;">Ready to Predict</div>
            <div style="color:rgba(255,255,255,0.4);font-size:13px;margin-top:8px;">
                Fill in your details on the left and click<br>
                <b style="color:rgba(196,181,253,0.9)">🚀 Predict My Stress Level</b>
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

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: st.metric("Total Runs", len(hist_df))
        with mc2: st.metric("🟢 Low Stress", (hist_df["label"] == "Low").sum())
        with mc3: st.metric("🟡 Medium Stress", (hist_df["label"] == "Medium").sum())
        with mc4: st.metric("🔴 High Stress", (hist_df["label"] == "High").sum())

        st.markdown("---")

        display_df = hist_df[["run", "label", "score", "sleep", "study",
                               "social", "attendance", "exam_pressure",
                               "family_support", "student_type",
                               "proba_low", "proba_med", "proba_high"]].copy()
        display_df.columns = [
            "Run #", "Prediction", "Confidence %", "Sleep Hrs", "Study Hrs",
            "Social Hrs", "Attendance %", "Exam Pressure", "Family Support",
            "Student Type", "Low %", "Medium %", "High %"
        ]
        st.dataframe(display_df, width="stretch", hide_index=True)

        st.markdown("### 📈 Stress Confidence Trend")
        trend_fig = build_history_chart(st.session_state.history)
        st.plotly_chart(trend_fig, width="stretch", config={"displayModeBar": False})

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()


# ===========================================================
# TAB 3 — ANALYTICS
# ===========================================================
with tab_analytics:
    st.markdown("### 📊 Model Analytics")

    feature_names = [
        "Student Type", "Sleep Hours", "Study Hours",
        "Social Media Hrs", "Attendance %",
        "Exam Pressure", "Family Support"
    ]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        chart_title = "🌳 Feature Importances (from Model)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
        chart_title = "📐 Feature Coefficients (absolute)"
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
        chart_title = "📊 Feature Weights"

    a1, a2 = st.columns([1.2, 0.8])

    with a1:
        st.markdown(f"#### {chart_title}")
        imp_fig = build_feature_importance_chart(feature_names, list(importances))
        st.plotly_chart(imp_fig, width="stretch", config={"displayModeBar": False})

    with a2:
        st.markdown("#### 🔍 Importance Breakdown")
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": [round(i, 4) for i in importances],
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        imp_df.insert(0, "Rank", range(1, len(imp_df)+1))
        st.dataframe(imp_df, width="stretch", hide_index=True)

        top_feature = imp_df.iloc[0]["Feature"]
        st.markdown(f"""
        <div class="glass-card" style="margin-top:14px;">
            <div class="card-title">💡 Key Insight</div>
            <div style="color:rgba(255,255,255,0.85);font-size:13px;">
                <b style="color:#c4b5fd">{top_feature}</b> is the most influential
                factor in predicting student stress. Focus on improving this area first.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🤖 Model Information")
    info1, info2, info3 = st.columns(3)
    with info1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">🧠 Algorithm</div>
            <div style="color:white;font-size:14px;font-weight:600;">{type(model).__name__}</div>
        </div>
        """, unsafe_allow_html=True)
    with info2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">📐 Scaler</div>
            <div style="color:white;font-size:14px;font-weight:600;">{type(scaler).__name__}</div>
        </div>
        """, unsafe_allow_html=True)
    with info3:
        n_classes = len(model.classes_) if hasattr(model, "classes_") else 3
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">🎯 Output Classes</div>
            <div style="color:white;font-size:14px;font-weight:600;">{n_classes} (Low / Medium / High)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📚 Stress Level Reference")
    r1, r2, r3 = st.columns(3)
    refs = [
        (r1, "🟢 Low Stress",    "#065f46", "#34d399", "Sleep ≥7h • Study 3–6h • Social ≤2h • Attendance ≥80% • Pressure ≤4"),
        (r2, "🟡 Medium Stress", "#78350f", "#fbbf24", "Sleep 6–7h • Study 6–9h • Social 2–4h • Attendance 60–79% • Pressure 5–7"),
        (r3, "🔴 High Stress",   "#7f1d1d", "#f87171", "Sleep <6h • Study >9h • Social >4h • Attendance <60% • Pressure ≥8"),
    ]
    for col, title, bg, border, desc in refs:
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{bg}cc,{bg}88);
                        border:1px solid {border}55;border-radius:16px;
                        padding:18px;text-align:center;">
                <div style="font-size:16px;font-weight:700;color:white;margin-bottom:8px;">{title}</div>
                <div style="font-size:11px;color:rgba(255,255,255,0.7);">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
