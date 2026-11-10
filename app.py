import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Hospital Decision Support - Heart Disease Risk",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Management System: Heart Disease Risk Screening")
st.write(
    "Clinical decision support system evaluating cardiovascular disease risk "
    "from physiological vitals, behavioral indicators, and patient medical history."
)
st.markdown("---")

# 2. Pipeline Training and Caching
@st.cache_resource
def get_trained_pipeline():
    if os.path.exists('hospital_synthetic_dataset.csv'):
        df = pd.read_csv('hospital_synthetic_dataset.csv')
    elif os.path.exists('hospital_synthetic_dataset.xlsx'):
        df = pd.read_excel('hospital_synthetic_dataset.xlsx')
    else:
        st.error("Dataset file not found! Please place hospital_synthetic_dataset.csv or .xlsx in this folder.")
        st.stop()

    # Drop blank trailing rows
    df = df.dropna(subset=['heart_disease'])
    df['heart_disease'] = df['heart_disease'].astype(int)

    X = df.drop(columns=['heart_disease'])
    y = df['heart_disease']

    categorical_cols = ['sex', 'smoker', 'diabetes', 'family_history']
    numerical_cols = ['age', 'blood_pressure', 'cholesterol', 'bmi', 'physical_activity', 'stress_level']

    # Preprocessor using One-Hot Encoding for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    return pipeline

pipeline = get_trained_pipeline()

# 3. Patient Intake Input Form
st.subheader("Patient Clinical Intake & Diagnostic Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Demographics & Genetic History")
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=55, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    family_history = st.selectbox("Family History of Cardiovascular Disease", options=["No", "Yes"])

with col2:
    st.markdown("#### Physiological Biometrics")
    blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130, step=1)
    cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=120, max_value=350, value=220, step=1)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=50.0, value=28.5, step=0.1)

with col3:
    st.markdown("#### Lifestyle & Comorbidities")
    smoker = st.selectbox("Smoker Status", options=["No", "Yes"])
    diabetes = st.selectbox("Diagnosed with Diabetes", options=["No", "Yes"])
    physical_activity = st.slider("Weekly Physical Activity Level (0: Sedentary to 4: High)", min_value=0, max_value=4, value=2)
    stress_level = st.slider("Psychological Stress Level (1: Low to 10: Extreme)", min_value=1, max_value=10, value=5)

st.markdown("---")

# 4. Evaluation and Output Section
if st.button("Generate Diagnostic Risk Assessment", use_container_width=True):
    patient_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'bmi': bmi,
        'smoker': smoker,
        'diabetes': diabetes,
        'physical_activity': physical_activity,
        'family_history': family_history,
        'stress_level': stress_level
    }])

    prediction = pipeline.predict(patient_df)[0]
    risk_prob = pipeline.predict_proba(patient_df)[0][1]

    st.subheader("Diagnostic Risk Assessment Result")

    if prediction == 1:
        st.error(f"High Cardiovascular Risk Detected -- Calculated Risk Probability: {risk_prob * 100:.1f}%")
        st.warning(
            "Clinical Triage Directive: Flag patient chart for an immediate cardiology consultation, "
            "lipid panel re-evaluation, and continuous blood pressure monitoring."
        )
    else:
        st.success(f"Low / Normal Risk Profile -- Calculated Risk Probability: {risk_prob * 100:.1f}%")
        st.info(
            "Clinical Triage Directive: Continue routine annual preventive screening, "
            "dietary counseling, and regular cardiovascular exercise."
        )