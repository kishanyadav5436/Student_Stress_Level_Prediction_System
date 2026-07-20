import streamlit as st
import pandas as pd
import joblib

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🧠",
    layout="centered"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.stApp{
    background: linear-gradient(135deg,#6a11cb,#2575fc);
}

.main-card{
    background:white;
    padding:30px;
    border-radius:20px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.25);
}

.title{
    text-align:center;
    color:#0f172a;
    font-size:38px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

.result-low{
    background:#d4edda;
    padding:20px;
    border-radius:15px;
    color:#155724;
    font-size:22px;
    text-align:center;
    font-weight:bold;
}

.result-medium{
    background:#fff3cd;
    padding:20px;
    border-radius:15px;
    color:#856404;
    font-size:22px;
    text-align:center;
    font-weight:bold;
}

.result-high{
    background:#f8d7da;
    padding:20px;
    border-radius:15px;
    color:#721c24;
    font-size:22px;
    text-align:center;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
model = joblib.load("student_stress_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ Sidebar ------------------
st.sidebar.title("🧠 Student Stress Predictor")
st.sidebar.info("""
Predict the stress level of a student using Machine Learning.

### Features Used
- Student Type
- Sleep Hours
- Study Hours
- Social Media
- Attendance
- Exam Pressure
- Family Support
""")

# ------------------ Main Card ------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">🎓 Student Stress Level Prediction</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Fill all the details below and click Predict.</div>', unsafe_allow_html=True)

student_type = st.selectbox(
    "🎓 Student Type",
    ["School", "College", "Working_Student"]
)

col1, col2 = st.columns(2)

with col1:
    sleep_hours = st.number_input(
        "😴 Sleep Hours",
        0.0, 12.0, 7.0
    )

    study_hours = st.number_input(
        "📚 Study Hours",
        0.0, 15.0, 4.0
    )

with col2:
    social_media = st.number_input(
        "📱 Social Media Hours",
        0.0, 15.0, 2.0
    )

    attendance = st.number_input(
        "🏫 Attendance %",
        0.0, 100.0, 75.0
    )

exam_pressure = st.slider(
    "😰 Exam Pressure",
    1,10,5
)

family_support = st.slider(
    "❤️ Family Support",
    1,10,5
)

student_type_map = {
    "School":0,
    "College":1,
    "Working_Student":2
}

student_type = student_type_map[student_type]

st.write("")

if st.button("🚀 Predict Stress Level", use_container_width=True):

    new_student = pd.DataFrame({
        "Student_Type":[student_type],
        "Sleep_Hours":[sleep_hours],
        "Study_Hours":[study_hours],
        "Social_Media_Hours":[social_media],
        "Attendance":[attendance],
        "Exam_Pressure":[exam_pressure],
        "Family_Support":[family_support]
    })

    new_student_scaled = scaler.transform(new_student)

    prediction = model.predict(new_student_scaled)[0]

    st.divider()

    if prediction == 0:
        st.markdown(
            '<div class="result-low">🟢 Predicted Stress Level : LOW 😊</div>',
            unsafe_allow_html=True
        )

    elif prediction == 1:
        st.markdown(
            '<div class="result-medium">🟡 Predicted Stress Level : MEDIUM 😐</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="result-high">🔴 Predicted Stress Level : HIGH 😟</div>',
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
