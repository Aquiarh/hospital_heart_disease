import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import os

# 1. Load Data & Drop Any Blank Trailing Rows
if os.path.exists('hospital_synthetic_dataset.xlsx'):
    df = pd.read_excel('hospital_synthetic_dataset.xlsx')
elif os.path.exists('hospital_synthetic_dataset.csv'):
    df = pd.read_csv('hospital_synthetic_dataset.csv')
else:
    raise FileNotFoundError("Dataset file not found!")

# Clean out any empty rows created by Excel
df = df.dropna(subset=['heart_disease'])
df['heart_disease'] = df['heart_disease'].astype(int)

X = df.drop(columns=['heart_disease'])
y = df['heart_disease']

# 2. Define Categorical and Numerical Columns
categorical_cols = ['sex', 'smoker', 'diabetes', 'family_history']
numerical_cols = ['age', 'blood_pressure', 'cholesterol', 'bmi', 'physical_activity', 'stress_level']

# 3. Construct Preprocessor: One-Hot Encoding for categorical, Scaling for numerical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# 4. Construct End-to-End Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Split Dataset (80/20 Stratified Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Fit Pipeline
pipeline.fit(X_train, y_train)

# 7. Evaluate
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("=== HOSPITAL HEART DISEASE MODEL METRICS ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Save Pipeline Model
joblib.dump(pipeline, 'heart_disease_model.pkl')
print("\nModel saved successfully as 'heart_disease_model.pkl'")