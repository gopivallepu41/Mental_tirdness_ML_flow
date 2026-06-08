import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Tiredness Predictor",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = "file:///C:/Users/somas/OneDrive/Desktop/Mental_tirdness/mlruns/5/models/m-08ea8d8e78394dddb0c95f9def68c5fb/artifacts"

@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(MODEL_PATH)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🧠 Mental Tiredness Score Predictor")
st.markdown("Fill in your work and lifestyle details below, then click **Predict** to get your estimated mental tiredness score.")
st.divider()

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
st.subheader("📋 Work & Cognitive Load")
col1, col2, col3 = st.columns(3)

with col1:
    number_of_decisions_made = st.number_input("Decisions Made Today", min_value=82, max_value=165, value=120, step=1)
    context_switch_count = st.number_input("Context Switches", min_value=0, max_value=21, value=8, step=1)
    notifications_received = st.number_input("Notifications Received", min_value=30, max_value=101, value=65, step=1)
    workload_score = st.slider("Workload Score (1-10)", min_value=1.0, max_value=10.0, value=5.7, step=0.1)

with col2:
    screen_time_min = st.slider("Screen Time (minutes)", min_value=20, max_value=853, value=302, step=5)
    deep_work_min = st.slider("Deep Work (minutes)", min_value=0, max_value=313, value=97, step=5)
    task_complexity_avg = st.slider("Avg Task Complexity (1-10)", min_value=1.0, max_value=10.0, value=5.5, step=0.1)
    break_frequency = st.number_input("Number of Breaks Taken", min_value=0, max_value=13, value=4, step=1)

with col3:
    caffeine_mg = st.slider("Caffeine Intake (mg)", min_value=0.0, max_value=423.0, value=131.0, step=5.0)
    noise_level_db = st.slider("Noise Level (dB)", min_value=20.0, max_value=85.0, value=48.0, step=0.5)
    temperature_c = st.slider("Room Temperature (C)", min_value=15.0, max_value=35.0, value=23.0, step=0.5)

st.divider()
st.subheader("😴 Sleep & Health")
col4, col5, col6 = st.columns(3)

with col4:
    sleep_hours = st.slider("Sleep Hours (last night)", min_value=3.0, max_value=11.0, value=7.0, step=0.25)

with col5:
    deep_sleep_pct = st.slider("Deep Sleep (%)", min_value=0.0, max_value=41.0, value=19.0, step=0.5)

with col6:
    hydration_l = st.slider("Hydration (litres)", min_value=0.3, max_value=4.6, value=1.9, step=0.1)

st.divider()
st.subheader("🗂️ Environment & Mood")
col7, col8, col9 = st.columns(3)

with col7:
    mood = st.selectbox("Current Mood", options=["Low", "Neutral", "Happy"], index=1)

with col8:
    work_environment = st.selectbox("Work Environment", options=["Quiet", "Moderate Noise", "Noisy"], index=0)

with col9:
    work_type = st.selectbox("Work Type", options=["Remote", "Office", "Manual", "Student"], index=0)

# ─────────────────────────────────────────────
# BUILD INPUT DATAFRAME
# ─────────────────────────────────────────────
input_df = pd.DataFrame([{
    "number_of_decisions_made": number_of_decisions_made,
    "context_switch_count":     context_switch_count,
    "notifications_received":   notifications_received,
    "screen_time_min":          screen_time_min,
    "deep_work_min":            deep_work_min,
    "task_complexity_avg":      task_complexity_avg,
    "caffeine_mg":              caffeine_mg,
    "break_frequency":          break_frequency,
    "sleep_hours":              sleep_hours,
    "deep_sleep_pct":           deep_sleep_pct,
    "hydration_l":              hydration_l,
    "mood":                     mood,
    "work_type":                work_type,
    "work_environment":         work_environment,
    "noise_level_db":           noise_level_db,
    "temperature_c":            temperature_c,
    "workload_score":           workload_score,
}])

with st.expander("🔍 Show raw input sent to model"):
    st.dataframe(input_df)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
st.divider()

if st.button("⚡ Predict Mental Tiredness Score", use_container_width=True, type="primary"):
    if not model_loaded:
        st.error(f"❌ Model could not be loaded.\n\n{load_error}")
    else:
        try:
            prediction = model.predict(input_df)
            score = round(float(prediction[0]), 2)

            st.success(f"### 🧠 Predicted Mental Tiredness Score: **{score:.2f}**")

            bar_val = min(max(score / 100, 0.0), 1.0)
            st.progress(bar_val)

            if score <= 33:
                st.info("🟢 **Low Tiredness** — You are in great mental shape. Keep it up!")
            elif score <= 66:
                st.warning("🟡 **Moderate Tiredness** — Consider taking a proper break soon.")
            else:
                st.error("🔴 **High Tiredness** — Rest is strongly recommended.")

            st.divider()
            st.subheader("💡 Personalised Tips")
            tips = []
            if sleep_hours < 6:
                tips.append("😴 You slept less than 6 hours. Aim for 7-9 hours for better cognitive recovery.")
            if caffeine_mg > 300:
                tips.append("☕ High caffeine intake. Consider reducing to below 300 mg/day.")
            if break_frequency < 2:
                tips.append("⏸️ Very few breaks taken. Try the Pomodoro technique (25 min work / 5 min break).")
            if hydration_l < 1.5:
                tips.append("💧 You are under-hydrated. Aim for at least 2 litres of water per day.")
            if screen_time_min > 480:
                tips.append("🖥️ Over 8 hours of screen time. Follow the 20-20-20 rule to reduce eye strain.")
            if deep_work_min < 30:
                tips.append("🎯 Low deep work time. Try blocking distractions for focused work sessions.")

            if tips:
                for tip in tips:
                    st.markdown(f"- {tip}")
            else:
                st.markdown("✅ Your inputs look healthy! Maintain these good habits.")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Ridge Regression (Test R2 = 0.75) · Trained with scikit-learn + MLflow + Optuna")