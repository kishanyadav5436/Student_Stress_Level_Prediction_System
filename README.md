# 🎓 Student Stress Level Prediction

A Machine Learning web application built with **Streamlit** that predicts a student's stress level based on factors such as study hours, sleep hours, attendance, exam pressure, family support, and social media usage.

---

## 📌 Features

- Predicts student stress level (Low, Medium, High)
- Interactive and user-friendly Streamlit interface
- Uses a trained Machine Learning model
- Data preprocessing using StandardScaler
- Instant prediction results

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```
Student-Stress-Prediction/
│
├── student_stress_model.py      # Streamlit Application
├── student_stress_model.pkl     # Trained Machine Learning Model
├── scaler.pkl                   # Saved StandardScaler
├── requirements.txt             # Project Dependencies
├── README.md                    # Project Documentation
└── dataset.csv                  # Dataset (Optional)
```

---

## 📊 Input Features

The application uses the following features:

- Student Type
- Sleep Hours
- Study Hours
- Social Media Hours
- Attendance (%)
- Exam Pressure
- Family Support

---

## 🎯 Output

The model predicts one of the following stress levels:

- 😊 Low Stress
- 😐 Medium Stress
- 😟 High Stress

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/student-stress-prediction.git
```

### 2. Navigate to the project folder

```bash
cd student-stress-prediction
```

### 3. Create a virtual environment (Optional)

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run student_stress_model.py
```

The application will open automatically in your browser.

---

## 📷 Application Workflow

1. Select Student Type
2. Enter Sleep Hours
3. Enter Study Hours
4. Enter Social Media Hours
5. Enter Attendance
6. Select Exam Pressure
7. Select Family Support
8. Click **Predict**
9. View the predicted stress level

---

## 📦 Model Information

- Algorithm: Machine Learning Classification
- Data Scaling: StandardScaler
- Model Storage: Joblib

---

## 📄 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Kishan Kumar**

B.Tech Computer Science

Machine Learning Enthusiast

---

## ⭐ Future Improvements

- Better UI/UX
- Feature importance visualization
- Probability scores
- User authentication
- Model comparison
- Cloud deployment (Streamlit Community Cloud)

---

## 📜 License

This project is created for educational purposes.# Student_Stress_Level_Prediction_System
