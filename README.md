# 🧠 Student Stress Level Prediction System

A **Machine Learning powered web application** built with Streamlit that predicts a student's stress level (Low, Medium, High) based on daily lifestyle factors — now with advanced analytics, personalized recommendations, interactive charts, and downloadable reports.

---

## 🚀 Live Features

### 🎯 Prediction Engine
- Predicts stress level: **Low / Medium / High**
- Shows **model confidence %** for each category
- Animated **stress gauge** (Plotly circular meter)
- **Probability breakdown** bar chart

### 💡 Personalized Recommendations
- Tailored wellness tips based on **your specific inputs**
- Separate tips for sleep, social media, attendance, exam pressure, etc.
- General stress-level-based guidance

### 📋 Session History
- Tracks all predictions in the current session
- Compare "what-if" scenarios side-by-side
- **Stress confidence trend** chart
- Export history as **CSV**

### 📊 Analytics Dashboard
- **Feature importance chart** (which factors affect stress most)
- Model information panel (algorithm, scaler, output classes)
- Stress level reference guide

### 📄 PDF Report Export
- Downloadable PDF report with:
  - Input summary table
  - Prediction result
  - Probability scores
  - Personalized recommendations
- Styled report with branding

### 🎨 Premium UI
- Glassmorphism dark mode design
- Multi-tab layout (Predict / History / Analytics)
- Metric cards with color-coded status
- Smooth hover animations
- Responsive 2-column layout

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python | Core language |
| Streamlit | Web framework |
| Scikit-learn | ML model (training + inference) |
| Pandas / NumPy | Data processing |
| Plotly | Interactive charts |
| fpdf2 | PDF report generation |
| Joblib | Model serialization |

---

## 📂 Project Structure

```
Student_Stress_Level_Prediction_System/
│
├── student_stress_model.py      # 🏠 Main Streamlit App
│
├── utils/
│   ├── recommendations.py       # 💡 Personalized tip engine
│   ├── charts.py                # 📊 Plotly chart builders
│   └── pdf_report.py            # 📄 PDF report generator
│
├── student_stress_model.pkl     # 🤖 Trained ML Model
├── scaler.pkl                   # 📐 Saved StandardScaler
├── requirements.txt             # 📦 Dependencies
└── README.md                    # 📖 Documentation
```

---

## 📊 Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| 🎓 Student Type | School / College / Working Student | Category |
| 😴 Sleep Hours | Average sleep per night | 0–12 hrs |
| 📚 Study Hours | Daily study time | 0–15 hrs |
| 📱 Social Media Hours | Daily social media usage | 0–15 hrs |
| 🏫 Attendance % | Class attendance percentage | 0–100% |
| 😰 Exam Pressure | Perceived exam pressure | 1–10 scale |
| ❤️ Family Support | Level of family support | 1–10 scale |

---

## 🎯 Stress Level Outputs

| Level | Indicator | Description |
|-------|-----------|-------------|
| 😊 Low Stress | 🟢 | Healthy balance maintained |
| 😐 Medium Stress | 🟡 | Some factors need attention |
| 😟 High Stress | 🔴 | Immediate action recommended |

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/kishanyadav5436/Student_Stress_Level_Prediction_System.git
cd Student_Stress_Level_Prediction_System
```

### 2. Create a virtual environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run student_stress_model.py
```

The app will open automatically at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit>=1.28.0
pandas
numpy
scikit-learn
joblib
plotly
fpdf2
```

---

## 👨‍💻 Author

**Kishan Kumar**
B.Tech Computer Science | Machine Learning Enthusiast

---

## 📜 License

This project is created for educational purposes.
