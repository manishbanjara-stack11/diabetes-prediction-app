
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Diabetes Risk Dashboard",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "diabetes_model.joblib"
    return joblib.load(model_path)

model = load_model()

st.markdown("""
<style>
.stApp {
    background-color: #10121c;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #242633;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Prediction Dashboard")
st.caption(
    "Educational machine learning project — "
    "not a medical diagnosis."
)

with st.sidebar:
    st.header("Patient Information")
    gender = st.selectbox(
        "Gender", ["Female", "Male", "Other"]
    )
    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox(
        "Hypertension", [0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )
    heart_disease = st.selectbox(
        "Heart Disease", [0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )
    smoking_history = st.selectbox(
        "Smoking History",
        ["never", "No Info", "current",
         "former", "ever", "not current"]
    )
    bmi = st.number_input(
        "BMI", min_value=10.0,
        max_value=80.0, value=24.0
    )
    hba1c = st.number_input(
        "HbA1c Level", min_value=3.0,
        max_value=20.0, value=5.5
    )
    glucose = st.number_input(
        "Blood Glucose Level (mg/dL)",
        min_value=40, max_value=500, value=100
    )

    analyze = st.button(
        "Analyze Measurements",
        type="primary",
        use_container_width=True
    )

if analyze:
    patient = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }])


    prediction = int(model.predict(patient)[0])
    probabilities = model.predict_proba(patient)[0]

    # Class labels ke saath probability check karo
    class_labels = model.classes_.tolist()
    diabetes_index = class_labels.index(1)

    probability_decimal = float(probabilities[diabetes_index])
    probability = float(probabilities[1]) * 100
    st.write("Prediction:", prediction)
    st.write("Probabilities:", probabilities)
    st.write("Diabetes probability:", probability)

    st.subheader("📊 Analysis Results")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("Model Prediction: Diabetes = YES")
        else:
            st.success("Model Prediction: Diabetes = NO")

        st.metric(
            "Estimated diabetes probability",
            f"{probability:.4f}%"
        )

    with col2:

        if probability < 33:
            risk_zone = "LOW"
            zone_color = "#22c55e"
        elif probability < 66:
            risk_zone = "MEDIUM"
            zone_color = "#f5b82e"
        else:
            risk_zone = "HIGH"
            zone_color = "#ef4444"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            number={
                "suffix": "%",
                "valueformat": ".2f",
                "font": {"size": 38, "color": "white"}
            },
            title={
                "text": f"Model Probability — {risk_zone}",
                "font": {"size": 20, "color": zone_color}
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "ticksuffix": "%",
                    "tickfont": {"color": "white"}
                },
                "bar": {
                    "color": zone_color,
                    "thickness": 0.25
                },
                "bgcolor": "#10121c",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 33],
                        "color": "#176b43"
                    },
                    {
                        "range": [33, 66],
                        "color": "#a87920"
                    },
                    {
                        "range": [66, 100],
                        "color": "#8b3038"
                    }
                ],
                "threshold": {
                    "line": {"color": "white", "width": 5},
                    "thickness": 0.85,
                    "value": probability
                }
            }
        ))

        fig.update_layout(
            height=350,
            paper_bgcolor="#10121c",
            font={"color": "white"},
            margin={"l": 30, "r": 30, "t": 75, "b": 20}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"### Current meter zone: :{('green' if probability < 33 else 'orange' if probability < 66 else 'red')}[{risk_zone}]"
        )


    st.subheader("📋 Submitted Measurements")
    st.dataframe(patient, use_container_width=True)

    st.info(
        "This probability is the model's estimate for the "
        "dataset label. It is not a clinically validated "
        "personal risk estimate. Consult a qualified "
        "healthcare professional for medical advice."
    )
else:
    st.info(
        "Enter the measurements in the sidebar, then "
        "click Analyze Measurements."
    )


