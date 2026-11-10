# Hospital Management System: Heart Disease Screening & Risk Prediction

An end-to-end clinical machine learning decision-support application built with Scikit-Learn and Streamlit to evaluate patient cardiovascular disease risk during clinical triage.

## Project Overview
This project implements a complete predictive modeling pipeline integrated into a hospital management workflow. It takes patient biometrics, lifestyle factors, and medical history to evaluate the likelihood of heart disease (presence: 1, absence: 0) and assists clinical staff in preventative screening.

## Key Features
- **Data Preprocessing & Encoding:** Encapsulates nominal clinical variables (`sex`, `smoker`, `diabetes`, `family_history`) using Scikit-Learn's `OneHotEncoder(drop='first')` and standardizes continuous vitals (`blood_pressure`, `cholesterol`, `bmi`, `stress_level`) via `StandardScaler` inside a unified `ColumnTransformer`.
- **Classification Engine:** Supervised Random Forest Classifier trained on a 500-patient hospital cohort using stratified data splitting.
- **Interactive Medical Portal:** Web-based diagnostic assessment interface developed in Streamlit for real-time patient intake and immediate risk stratification.
- **Evaluation:** Evaluated across Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC.

## Local Setup & Run
```bash
# Clone the repository
git clone [https://github.com/Aquiarh/hospital_heart_disease.git](https://github.com/Aquiarh/hospital_heart_disease.git)
cd hospital_heart_disease

# Install dependencies
py -m pip install -r requirements.txt

# Launch the Streamlit application
py -m streamlit run app.py