import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Custom CSS (modern look)
# -----------------------------
st.markdown("""
<style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 100%);
    }

    /* Hide default streamlit menu/footer clutter */
    #MainMenu, footer {visibility: hidden;}

    /* Card container */
    .main-card {
        background: #ffffff;
        padding: 2rem 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    /* Title styling */
    .app-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6a5af9, #d66ef9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #374151;
        margin: 1.2rem 0 0.6rem 0;
        border-left: 4px solid #6a5af9;
        padding-left: 0.6rem;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6a5af9, #d66ef9);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.7rem 0;
        border-radius: 12px;
        border: none;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-top: 1rem;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(106, 90, 249, 0.35);
        color: white;
    }

    /* Result card */
    .result-card {
        text-align: center;
        padding: 1.6rem;
        border-radius: 16px;
        margin-top: 1.5rem;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .result-low {
        background: #e7f9ed;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .result-medium {
        background: #fff7e0;
        color: #b45309;
        border: 1px solid #fcd34d;
    }
    .result-high {
        background: #fde8e8;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model & scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("student_stress_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="app-title">🧠 Student Stress Level Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Fill in the details below and get an instant stress level prediction</div>', unsafe_allow_html=True)

# -----------------------------
# Main Card / Form
# -----------------------------
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-header">👤 Profile</div>', unsafe_allow_html=True)
    student_type = st.selectbox(
        "Student Type",
        ["School", "College", "Working_Student"]
    )

    st.markdown('<div class="section-header">⏱️ Daily Habits</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        study_hours = st.number_input("📚 Study Hours", min_value=0.0, max_value=15.0, value=4.0, step=0.5)
    with col2:
        social_media = st.number_input("📱 Social Media Hours", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
        attendance = st.number_input("🏫 Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

    st.markdown('<div class="section-header">💭 Wellbeing Factors</div>', unsafe_allow_html=True)
    exam_pressure = st.slider("📝 Exam Pressure", 1, 10, 5)
    family_support = st.slider("❤️ Family Support", 1, 10, 5)

    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("🔮 Predict Stress Level")

# -----------------------------
# Prediction Logic
# -----------------------------
student_type_map = {
    "School": 0,
    "College": 1,
    "Working_Student": 2
}

if predict_clicked:
    encoded_type = student_type_map[student_type]

    new_student = pd.DataFrame({
        "Student_Type": [encoded_type],
        "Sleep_Hours": [sleep_hours],
        "Study_Hours": [study_hours],
        "Social_Media_Hours": [social_media],
        "Attendance": [attendance],
        "Exam_Pressure": [exam_pressure],
        "Family_Support": [family_support]
    })

    new_student_scaled = scaler.transform(new_student)
    prediction = model.predict(new_student_scaled)[0]

    with st.spinner("Analyzing student data..."):
        if prediction == 0:
            st.markdown(
                '<div class="result-card result-low">😊 Predicted Stress Level: <br> LOW</div>',
                unsafe_allow_html=True
            )
        elif prediction == 1:
            st.markdown(
                '<div class="result-card result-medium">😐 Predicted Stress Level: <br> MEDIUM</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card result-high">😟 Predicted Stress Level: <br> HIGH</div>',
                unsafe_allow_html=True
            )
