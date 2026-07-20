import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Student Stress Level Prediction",
    page_icon="🧠",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
model = joblib.load("student_stress_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------- CSS --------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp{
    background: linear-gradient(160deg, #1a1140 0%, #3a2a8c 45%, #2e5fd9 100%);
}

/* Hide Streamlit chrome */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Kill default block padding so the card sits cleanly at the top */
.block-container{
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 760px;
}

/* Main Card */
.main-card{
    background: rgba(255,255,255,0.08);
    padding: 40px 40px 30px 40px;
    border-radius: 24px;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 12px 40px rgba(0,0,0,.35);
}

/* Title */
.title{
    text-align:center;
    color:white;
    font-size:36px;
    font-weight:800;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.subtitle{
    text-align:center;
    color:rgba(255,255,255,0.7);
    font-size:15px;
    font-weight:400;
    margin-bottom:32px;
}

/* Section labels */
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: rgba(255,255,255,0.9) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Input widgets — clean, consistent, on-brand */
div[data-baseweb="select"] > div,
.stNumberInput input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 10px !important;
    color: white !important;
}

.stNumberInput button {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] > div > div {
    background: rgba(255,255,255,0.2) !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* Predict button */
.stButton > button{
    background: linear-gradient(90deg, #ff5f6d, #ffc371);
    color: #1a1140;
    font-weight: 700;
    font-size: 16px;
    border: none;
    border-radius: 12px;
    padding: 12px 0;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0px 6px 18px rgba(255,95,109,0.35);
}
.stButton > button:hover{
    transform: translateY(-2px);
    box-shadow: 0px 10px 24px rgba(255,95,109,0.45);
    color: #1a1140;
}

/* Result Cards */
.result-card{
    padding: 24px;
    border-radius: 16px;
    text-align:center;
    font-size: 22px;
    font-weight: 700;
    box-shadow: 0px 8px 20px rgba(0,0,0,.15);
}

.result-card .sub{
    display:block;
    font-size: 14px;
    font-weight: 400;
    margin-top: 8px;
    opacity: 0.85;
}

.low{ background:#d4edda; color:#155724; }
.medium{ background:#fff3cd; color:#856404; }
.high{ background:#f8d7da; color:#721c24; }

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #150c33, #241a5c);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.92) !important;
}
section[data-testid="stSidebar"] .stAlert{
    background: rgba(76, 217, 123, 0.12) !important;
    border: 1px solid rgba(76, 217, 123, 0.35) !important;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------

st.sidebar.markdown("## 🧠 Student Stress Predictor")
st.sidebar.success("Machine Learning Project")

st.sidebar.markdown("""
---
##### Features Used

- 🎓 Student Type
- 😴 Sleep Hours
- 📚 Study Hours
- 📱 Social Media
- 🏫 Attendance
- 😰 Exam Pressure
- ❤️ Family Support
""")

# -------------------- MAIN CARD --------------------

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Student Stress Level Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict stress level using Machine Learning</div>', unsafe_allow_html=True)

student_type = st.selectbox(
    "🎓 Student Type",
    ["School", "College", "Working_Student"]
)

col1, col2 = st.columns(2)

with col1:
    sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0)
    study_hours = st.number_input("📚 Study Hours", min_value=0.0, max_value=15.0, value=4.0)

with col2:
    social_media = st.number_input("📱 Social Media Hours", min_value=0.0, max_value=15.0, value=2.0)
    attendance = st.number_input("🏫 Attendance %", min_value=0.0, max_value=100.0, value=75.0)

exam_pressure = st.slider("😰 Exam Pressure", 1, 10, 5)
family_support = st.slider("❤️ Family Support", 1, 10, 5)

student_type_map = {
    "School": 0,
    "College": 1,
    "Working_Student": 2
}
student_type_encoded = student_type_map[student_type]

st.write("")

# -------------------- PREDICT --------------------

if st.button("🚀 Predict Stress Level", use_container_width=True):

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

    st.markdown("---")

    if prediction == 0:
        st.markdown(
            '<div class="result-card low">🟢 LOW STRESS 😊'
            '<span class="sub">Keep maintaining your healthy routine.</span></div>',
            unsafe_allow_html=True
        )
    elif prediction == 1:
        st.markdown(
            '<div class="result-card medium">🟡 MEDIUM STRESS 😐'
            '<span class="sub">Take regular breaks and manage your time.</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-card high">🔴 HIGH STRESS 😟'
            '<span class="sub">Reduce workload, sleep well and seek support.</span></div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)
