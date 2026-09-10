"""
Personalized Recommendations Engine
Generates tailored wellness tips based on the student's specific inputs and predicted stress level.
"""

TIPS_BY_LEVEL = {
    0: {  # Low Stress
        "title": "😊 You're doing great! Keep it up!",
        "general": [
            "✅ Maintain your current sleep and study schedule — it's working well!",
            "🎯 Consider setting new academic goals to stay challenged.",
            "🏃 Keep up physical activity; it helps maintain this balance.",
            "🎵 Use this calm period to explore hobbies or personal projects.",
        ]
    },
    1: {  # Medium Stress
        "title": "😐 Manageable — a few tweaks can help!",
        "general": [
            "⏱️ Use the Pomodoro Technique: 25 min study, 5 min break.",
            "📓 Keep a daily planner to reduce last-minute cramming.",
            "🧘 Try 10 minutes of deep breathing or meditation daily.",
            "💧 Stay hydrated — dehydration worsens focus and mood.",
        ]
    },
    2: {  # High Stress
        "title": "😟 High stress detected — take action now!",
        "general": [
            "🆘 Talk to a school counselor, teacher, or trusted adult today.",
            "🛑 Break your workload into small daily chunks — don't try to do everything at once.",
            "📵 Set strict screen-free hours, especially before bed.",
            "🤝 Reach out to friends or family — don't carry this alone.",
            "🏥 If stress feels overwhelming, seek professional mental health support.",
        ]
    }
}


def get_feature_tips(sleep_hours, study_hours, social_media, attendance, exam_pressure, family_support, student_type):
    """Generate personalized tips based on individual feature values."""
    tips = []

    # Sleep Hours
    if sleep_hours < 6:
        tips.append("😴 **Sleep Alert:** You're sleeping less than 6 hours. Aim for 7–8 hours for optimal brain function.")
    elif sleep_hours < 7:
        tips.append("😴 **Sleep Tip:** Try adding 30 more minutes of sleep — it can significantly improve focus and mood.")
    elif sleep_hours > 9:
        tips.append("😴 **Sleep Note:** Oversleeping (>9 hrs) can cause grogginess. Stick to a consistent 7–8 hour schedule.")

    # Study Hours
    if study_hours > 10:
        tips.append("📚 **Study Balance:** Studying 10+ hours daily can lead to burnout. Take breaks every 45–60 minutes.")
    elif study_hours < 2:
        tips.append("📚 **Study Consistency:** Less than 2 hours of study is low. Build a consistent routine — even 3–4 hours daily makes a difference.")

    # Social Media
    if social_media > 5:
        tips.append("📱 **Screen Time:** You spend 5+ hours on social media. Try a 2-hour daily limit and use apps like 'StayFocusd' to help.")
    elif social_media > 3:
        tips.append("📱 **Social Media Tip:** Reduce scrolling before bed — it disrupts sleep quality. Set a phone-free zone 1 hour before sleeping.")

    # Attendance
    if attendance < 60:
        tips.append("🏫 **Attendance Warning:** Below 60% attendance means you're missing a lot. Falling behind adds major stress — try attending more consistently.")
    elif attendance < 75:
        tips.append("🏫 **Attendance Tip:** Aim for at least 75% attendance to stay on track with coursework and reduce catch-up stress.")

    # Exam Pressure
    if exam_pressure >= 8:
        tips.append("😰 **Exam Pressure:** Very high! Break your syllabus into small daily targets. Start revision early — don't leave everything to the last day.")
    elif exam_pressure >= 6:
        tips.append("😰 **Exam Tip:** Practice past papers and mock tests to build confidence before the actual exam.")

    # Family Support
    if family_support <= 3:
        tips.append("❤️ **Support System:** Low family support detected. Consider talking to a school counselor, mentor, or joining a student support group.")
    elif family_support <= 5:
        tips.append("❤️ **Family Tip:** Share your academic concerns with family or friends — a good conversation can lighten the mental load.")

    # Student Type
    if student_type == "Working_Student":
        tips.append("💼 **Working Student:** Balancing work and studies is tough. Prioritize tasks using time-blocking and protect your weekends for study and rest.")
    elif student_type == "College":
        tips.append("🎓 **College Student:** College stress is real! Use your campus resources — library, counseling center, study groups.")

    return tips


def get_all_recommendations(prediction, sleep_hours, study_hours, social_media,
                             attendance, exam_pressure, family_support, student_type):
    """
    Returns a dict with:
    - level_tips: General tips for the predicted stress level
    - feature_tips: Personalized tips based on input values
    """
    level_data = TIPS_BY_LEVEL[prediction]
    feature_tips = get_feature_tips(
        sleep_hours, study_hours, social_media,
        attendance, exam_pressure, family_support, student_type
    )
    return {
        "title": level_data["title"],
        "general": level_data["general"],
        "feature_tips": feature_tips
    }
