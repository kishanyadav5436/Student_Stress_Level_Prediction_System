"""
PDF Report Generator for Student Stress Prediction.
Generates a downloadable PDF with prediction results and recommendations.
"""

from fpdf import FPDF
import datetime


class StressReportPDF(FPDF):
    def header(self):
        self.set_fill_color(26, 17, 64)   # Dark purple
        self.rect(0, 0, 210, 30, "F")
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(255, 255, 255)
        self.cell(0, 20, "Student Stress Level Report", align="C", ln=True)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Generated on {datetime.datetime.now().strftime('%d %b %Y, %H:%M')} | Student Stress Predictor", align="C")


def generate_pdf_report(prediction_label, stress_score, proba,
                         sleep_hours, study_hours, social_media,
                         attendance, exam_pressure, family_support,
                         student_type, recommendations):
    """
    Generate a PDF report and return it as bytes.
    """
    STRESS_COLORS = {
        "Low":    (21, 87, 36),
        "Medium": (133, 100, 4),
        "High":   (114, 28, 36),
    }
    STRESS_BG = {
        "Low":    (212, 237, 218),
        "Medium": (255, 243, 205),
        "High":   (248, 215, 218),
    }

    pdf = StressReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Stress Result Banner ---
    pdf.ln(8)
    r, g, b = STRESS_BG[prediction_label]
    pdf.set_fill_color(r, g, b)
    pdf.set_draw_color(r, g, b)
    pdf.rect(15, pdf.get_y(), 180, 22, "F")
    pdf.set_font("Helvetica", "B", 16)
    r2, g2, b2 = STRESS_COLORS[prediction_label]
    pdf.set_text_color(r2, g2, b2)
    emoji_map = {"Low": "LOW STRESS", "Medium": "MEDIUM STRESS", "High": "HIGH STRESS"}
    pdf.cell(0, 22, f"  Predicted: {emoji_map[prediction_label]}  |  Confidence: {stress_score:.1f}%", ln=True, align="L")

    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # --- Input Summary Table ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(26, 17, 64)
    pdf.cell(0, 8, "Student Input Summary", ln=True)
    pdf.ln(2)

    headers = ["Feature", "Value"]
    data = [
        ["Student Type", student_type],
        ["Sleep Hours", f"{sleep_hours} hrs/day"],
        ["Study Hours", f"{study_hours} hrs/day"],
        ["Social Media Hours", f"{social_media} hrs/day"],
        ["Attendance", f"{attendance}%"],
        ["Exam Pressure", f"{exam_pressure}/10"],
        ["Family Support", f"{family_support}/10"],
    ]

    col_w = [100, 80]
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(26, 17, 64)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for i, row in enumerate(data):
        fill = i % 2 == 0
        pdf.set_fill_color(240, 240, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 30, 30)
        for j, cell in enumerate(row):
            pdf.cell(col_w[j], 7, str(cell), border=1, fill=True)
        pdf.ln()

    pdf.ln(6)

    # --- Probability Scores ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(26, 17, 64)
    pdf.cell(0, 8, "Model Confidence Breakdown", ln=True)
    pdf.ln(2)
    labels_p = ["Low Stress", "Medium Stress", "High Stress"]
    for label, p in zip(labels_p, proba):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(60, 7, f"  {label}:", border=0)
        # Draw bar
        bar_len = int(p * 100)
        pdf.set_fill_color(100, 120, 220)
        pdf.rect(pdf.get_x(), pdf.get_y() + 1.5, bar_len, 4, "F")
        pdf.set_x(pdf.get_x() + bar_len + 3)
        pdf.cell(20, 7, f"{p*100:.1f}%", ln=True)
    pdf.ln(6)

    # --- Personalized Recommendations ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(26, 17, 64)
    pdf.cell(0, 8, "Personalized Recommendations", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, "General Tips:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for tip in recommendations.get("general", []):
        clean_tip = tip.replace("✅", "").replace("🎯", "").replace("🏃", "").replace("🎵", "")
        clean_tip = clean_tip.replace("⏱️", "").replace("📓", "").replace("🧘", "").replace("💧", "")
        clean_tip = clean_tip.replace("🆘", "").replace("🛑", "").replace("📵", "").replace("🤝", "").replace("🏥", "")
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 6, f"  * {clean_tip.strip()}", border=0)

    if recommendations.get("feature_tips"):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 7, "Personalized Input-Based Tips:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for tip in recommendations["feature_tips"]:
            # Strip markdown bold and emoji
            import re
            clean = re.sub(r'\*\*.*?\*\*', '', tip)
            clean = re.sub(r'[^\x00-\x7F]+', '', clean)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, f"  * {clean.strip()}", border=0)

    # Footer note
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 5, "Disclaimer: This report is generated by an ML model for educational purposes only. "
                         "Please consult a qualified professional for medical or psychological advice.", border=0)

    return bytes(pdf.output())
