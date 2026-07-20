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

.stApp{
    background: linear-gradient(135deg,#5B2CFF,#2E86FF);
}

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Card */
.main-card{
    background:rgba(255,255,255,0.12);
    padding:35px;
    border-radius:20px;
    backdrop-filter: blur(12px);
    box-shadow:0px 8px 30px rgba(0,0,0,.25);
}

/* Title */
.title{
    text-align:center;
    color:white;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#eeeeee;
    font-size:18px;
    margin-bottom:20px;
}

/* Result Cards */

.low{
    background:#d4edda;
    color:#155724;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.medium{
    background:#fff3cd;
    color:#856404;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.high{
    background:#f8d7da;
    color:#721c24;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------

st.sidebar.title("🧠 Student Stress Predictor")

st.sidebar.success("Machine Learning Project")

st.sidebar.markdown("""
### Features Used

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
    ["School","College","Working_Student"]
)

col1,col2 = st.columns(2)

with col1:

    sleep_hours = st.number_input(
        "😴 Sleep Hours",
        min_value=0.0,
        max_value=12.0,
        value=7.0
    )

    study_hours = st.number_input(
        "📚 Study Hours",
        min_value=0.0,
        max_value=15.0,
        value=4.0
    )

with col2:

    social_media = st.number_input(
        "📱 Social Media Hours",
        min_value=0.0,
        max_value=15.0,
        value=2.0
    )

    attendance = st.number_input(
        "🏫 Attendance %",
        min_value=0.0,
        max_value=100.0,
        value=75.0
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

# -------------------- PREDICT --------------------

if st.button("🚀 Predict Stress Level", use_container_width=True):

    input_df = pd.DataFrame({

        "Student_Type":[student_type],
        "Sleep_Hours":[sleep_hours],
        "Study_Hours":[study_hours],
        "Social_Media_Hours":[social_media],
        "Attendance":[attendance],
        "Exam_Pressure":[exam_pressure],
        "Family_Support":[family_support]

    })

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.markdown("---")

    if prediction == 0:

        st.markdown(
            '<div class="low">🟢 LOW STRESS 😊<br><br>Keep maintaining your healthy routine.</div>',
            unsafe_allow_html=True
        )

    elif prediction == 1:

        st.markdown(
            '<div class="medium">🟡 MEDIUM STRESS 😐<br><br>Take regular breaks and manage your time.</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="high">🔴 HIGH STRESS 😟<br><br>Reduce workload, sleep well and seek support.</div>',
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
