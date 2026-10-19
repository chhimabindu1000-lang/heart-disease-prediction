import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ----------------------------------
# 2. CUSTOM WEBSITE DESIGN
# ----------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f8fc;
}

.main-title {
    text-align: center;
    color: #c62828;
    font-size: 36px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555555;
    font-size: 16px;
}

.result-card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 15px;
}

.stButton > button {
    background-color: #c62828;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# 3. LOAD TRAINED MODEL
# ----------------------------------

MODEL_PATH = "heart_disease_model.pkl"

st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found! "
        "Please place heart_disease_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# ----------------------------------
# 4. PATIENT INPUT FORM
# ----------------------------------

st.subheader("🩺 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=0,
        max_value=300,
        value=130
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=220
    )

    sex = st.selectbox(
        "Sex (dataset encoding)",
        options=[0, 1]
    )

with col2:
    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=0,
        max_value=300,
        value=150
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    cp = st.selectbox(
        "Chest Pain Type (dataset encoding)",
        options=[0, 1, 2, 3]
    )

st.caption(
    "Note: Input ranges and category encodings should match "
    "the dataset used to train your model."
)

st.write("")

# ----------------------------------
# 5. PREDICTION
# ----------------------------------

if st.button("❤️ Predict"):

    input_data = pd.DataFrame([{
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "sex": sex,
        "cp": cp
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        # Result display
        if prediction == 1:

            st.warning(
                "⚠️ Model predicted Class 1."
            )

            st.write(
                "Check your dataset's target-label definition "
                "to understand what Class 1 represents."
            )

        elif prediction == 0:

            st.success(
                "✅ Model predicted Class 0."
            )

            st.write(
                "Check your dataset's target-label definition "
                "to understand what Class 0 represents."
            )

        else:
            st.info(f"Predicted class: {prediction}")

        # ----------------------------------
        # 6. PREDICTION PROBABILITY
        # ----------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            st.subheader("📈 Model Prediction Probabilities")

            probability_df = pd.DataFrame({
                "Class": [str(c) for c in classes],
                "Probability (%)": [
                    round(float(p) * 100, 2)
                    for p in probabilities
                ]
            })

            st.bar_chart(
                probability_df.set_index("Class")
            )

            predicted_index = classes.index(prediction)

            st.metric(
                "Probability for predicted class",
                f"{probabilities[predicted_index] * 100:.2f}%"
            )

            st.caption(
                "These are model probabilities, not a person's "
                "clinically verified probability of having heart disease."
            )

        else:
            st.info(
                "This model does not support predict_proba(), "
                "so probability display is unavailable."
            )

        # ----------------------------------
        # 7. MEDICAL DISCLAIMER
        # ----------------------------------

        st.info(
            "⚕️ This is an experimental machine-learning "
            "prediction, not a medical diagnosis. "
            "Do not use it to rule out heart disease or delay "
            "medical care. Consult a qualified healthcare professional "
            "for medical concerns."
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )

# ----------------------------------
# 8. FOOTER
# ----------------------------------

st.divider()

st.caption(
    "Heart Disease Prediction | Educational ML Project"
)