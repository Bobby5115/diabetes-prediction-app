import streamlit as st
import pickle
import numpy as np
import pandas as pd
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Page setup
st.set_page_config(page_title="Diabetes Risk System", page_icon="🩺", layout="wide")

# Load models
log_model = pickle.load(open("diabetes_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🩺 Advanced Diabetes Risk Prediction System")

# Sidebar
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Prediction Model",("Logistic Regression", "Random Forest"))

st.sidebar.markdown("---")
st.sidebar.caption('Diabetes is a chronic disease that can lead to serious complications if not detected early. This Model is a decision support system that applies machine learning techniques to predict a patients diabetic status based on medical measurements.')
st.sidebar.info('The model learns patterns from patient data to support early diabetes detection.')
st.sidebar.warning("It is not a medical diagnosis tool but a decision-support system")

# Input section
st.header("🧾 Enter Patient Medical Records")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level (mg/dL)")
    blood_pressure = st.number_input("Blood Pressure (mm Hg)")
    skin_thickness = st.number_input("Skin Thickness (mm)")
        
with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)")
    bmi = st.number_input("Body Mass Index (BMI)")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age", min_value=1)

if st.button("🔍 Predict"):

    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)

    if model_choice == "Logistic Regression":
        model = log_model
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
    else:
        model = rf_model
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    st.write(f"**You have {probability:.2%} Probability of Being Diabetic**")
    st.progress(int(probability * 100))

    if probability < 0.3:
        st.success("✅ You are good to Go")
    elif probability < 0.7:
        st.warning("⚠️ Go for checkup by weekend")
    else:
        st.error("🚨 Seek Medical attention immediately")

        # Feature Importance (for Random Forest only)
    if model_choice == "Random Forest":
        st.subheader("📈 Feature Importance")

        feature_names = ["Pregnancies","Glucose","BloodPressure",
                         "SkinThickness","Insulin","BMI",
                         "DPF","Age"]

        importance = rf_model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

    # Save results in session state
    st.session_state.prediction = prediction
    st.session_state.probability = probability

    # DISPLAY RESULT IF AVAILABLE
if "prediction" in st.session_state:

    prediction = st.session_state.prediction
    probability = st.session_state.probability


    # PDF Download
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Diabetes Risk Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Generate PDF Report
    if st.button("📄 Download Report"):

        buffer = io.BytesIO()   # Create memory buffer
        doc = SimpleDocTemplate(buffer)

        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Diabetes Risk Prediction Report", styles["Title"]))
        elements.append(Spacer(1, 12))

        data = [
            ["Feature", "Value"],
            ["Pregnancies", pregnancies],
            ["Glucose", glucose],
            ["Blood Pressure", blood_pressure],
            ["Skin Thickness", skin_thickness],
            ["Insulin", insulin],
            ["BMI", bmi],
            ["DPF", dpf],
            ["Age", age],
            ["Prediction", "Diabetic" if prediction == 1 else "Not Diabetic"],
            ["Probability", f"{probability:.2%}"]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))

        elements.append(table)
        doc.build(elements)

        buffer.seek(0)

        st.download_button(
            label="Download PDF Report",
            data=buffer,
            file_name="Diabetes_Report.pdf",
            mime="application/pdf"
        )
