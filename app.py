
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# Load saved model and encoders
@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "model.pkl", "rb") as file:
        model = pickle.load(file)

    with open(BASE_DIR / "encoders.pkl", "rb") as file:
        encoders = pickle.load(file)

    return model, encoders


try:
    model, encoders = load_artifacts()
except Exception:
    st.error(
        "Unable to load model.pkl or encoders.pkl. "
        "Please check that both files exist in your GitHub repository."
    )
    st.stop()


# Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #edf5ff, #f5f3ff);
    }
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(120deg, #123b70, #5269dc);
        color: white;
        padding: 28px;
        border-radius: 20px;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .hero p {
        color: #f0f5ff;
        margin-bottom: 0;
    }
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.9);
        padding: 22px;
        border: 1px solid #dbe5f5;
        border-radius: 18px;
    }
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background: #2855c5;
        color: white;
        border-radius: 10px;
        border: none;
        min-height: 44px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Header
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Dropout Prediction</h1>
        <p>
            Analyze student academic and financial details
            to estimate dropout risk using machine learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Enter Student Information")
st.write("Fill in all the details and click Predict.")

with st.form("student_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age", min_value=16, max_value=60, value=20
        )
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        admission_grade = st.number_input(
            "Admission Grade",
            min_value=0.0, max_value=200.0, value=120.0
        )
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0, max_value=100.0, value=75.0
        )

    with col2:
        study_hours = st.number_input(
            "Study Hours Per Day",
            min_value=0.0, max_value=24.0, value=3.0,
            step=0.5
        )
        scholarship = st.selectbox(
            "Scholarship", ["Yes", "No"]
        )
        tuition = st.selectbox(
            "Tuition Fees Paid", ["Yes", "No"]
        )

    with col3:
        parents = st.selectbox(
            "Parents' Education",
            ["HighSchool", "Bachelor", "Diploma", "Master"]
        )
        income = st.number_input(
            "Family Income",
            min_value=0.0, value=50000.0, step=1000.0
        )
        exam_score = st.number_input(
            "Exam Score",
            min_value=0.0, max_value=100.0, value=70.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Dropout Risk",
        use_container_width=True
    )


if submitted:
    try:
        # Encode categorical inputs using the saved encoders
        gender_encoded = encoders["gender"].transform(
            [gender]
        )[0]

        scholarship_encoded = encoders["scholarship"].transform(
            [scholarship]
        )[0]

        tuition_encoded = encoders["tuition_fees_paid"].transform(
            [tuition]
        )[0]

        parents_encoded = encoders["parents_education"].transform(
            [parents]
        )[0]

        # Keep the exact feature names and order used during training
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender_encoded,
            "admission_grade": admission_grade,
            "attendance_percentage": attendance,
            "study_hours_per_day": study_hours,
            "scholarship": scholarship_encoded,
            "tuition_fees_paid": tuition_encoded,
            "parents_education": parents_encoded,
            "family_income": income,
            "exam_score": exam_score
        }])

        prediction = int(model.predict(input_data)[0])

        # Class 1 means dropout in this project's dataset
        probabilities = model.predict_proba(input_data)[0]
        classes = [int(value) for value in model.classes_]

        if 1 not in classes:
            raise ValueError("Dropout class 1 is missing from the model.")

        dropout_probability = float(
            probabilities[classes.index(1)]
        )

        continue_probability = 1.0 - dropout_probability

        st.divider()
        st.subheader("Prediction Result")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Dropout Probability",
                f"{dropout_probability * 100:.2f}%"
            )

        with metric2:
            st.metric(
                "Continue Probability",
                f"{continue_probability * 100:.2f}%"
            )

        st.progress(dropout_probability)

        if prediction == 1:
            st.error(
                "⚠️ The model predicts a higher dropout risk."
            )
            st.write(
                "Consider reviewing attendance, academic progress, "
                "financial difficulties and available student support."
            )
        else:
            st.success(
                "✅ The model predicts a lower dropout risk."
            )
            st.write(
                "Continue monitoring the student's progress and "
                "maintain regular academic support."
            )

        st.caption(
            "This is a machine-learning estimate based on the "
            "project dataset, not a definitive assessment of a student."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
