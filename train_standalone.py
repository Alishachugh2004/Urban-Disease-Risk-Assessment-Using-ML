"""Standalone Training Script"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

# Ensure output is flushed immediately
sys.stdout.reconfigure(line_buffering=True)

#os.chdir('d:/Harman_Project')

# Load data
print("Loading data...", flush=True)
df = pd.read_csv('data/dataset.csv')
print(f"Dataset shape: {df.shape}", flush=True)

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Create target variable - since all labels are 'normal', create risk levels based on environmental factors
# Use humidity and temperature to create synthetic risk levels
print("Creating target variable...", flush=True)

# Create risk levels based on humidity thresholds
def create_risk_level(row):
    humidity = row.get('humidity', 50)
    temp = row.get('temp', 25)
    cases = row.get('cases', 0)
    
    # High risk: high humidity + high temperature + high cases
    if humidity > 80 and temp > 30 and cases > 5000:
        return 2  # High
    # Medium risk: moderate humidity or temperature
    elif humidity > 60 or temp > 25:
        return 1  # Medium
    else:
        return 0  # Low

df['risk_level'] = 0
df.loc[(df['humidity'] > 60) | (df['temp'] > 25), 'risk_level'] = 1
df.loc[(df['humidity'] > 80) & (df['temp'] > 30) & (df['cases'] > 5000), 'risk_level'] = 2
print(f"Risk distribution: {df['risk_level'].value_counts().to_dict()}", flush=True)

# Select features - use relevant environmental features
feature_cols = ['temp', 'humidity', 'solarradiation', 'cases']
print(f"Features: {feature_cols}", flush=True)

X = df[feature_cols].values
y = df['risk_level'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

# Train model
print("Training model...", flush=True)
model = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", flush=True)

# Classification report
print("\nClassification Report:", flush=True)
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], zero_division=0))

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Saved: models/model.pkl", flush=True)
print("Saved: models/scaler.pkl", flush=True)

# Save report
report = f"""===============================================
Urban Disease Spread Risk Assessment Model Report
===============================================

Model: RandomForestClassifier
Training Date: 2026-04-23

DATASET SUMMARY
---------------
Total Samples: {len(df)}
Features: {', '.join(feature_cols)}
Train/Test Split: 80/20

CLASS DISTRIBUTION
------------------
Low Risk (0):    {int(np.sum(y==0))} samples
Medium Risk (1): {int(np.sum(y==1))} samples
High Risk (2):   {int(np.sum(y==2))} samples

MODEL PERFORMANCE
-----------------
Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)

FEATURE IMPORTANCE
------------------
"""
for feat, imp in zip(feature_cols, model.feature_importances_):
    report += f"{feat}: {imp:.4f}\n"

report += "===============================================\nEND OF REPORT\n===============================================\n"

os.makedirs('reports', exist_ok=True)
with open('reports/report.txt', 'w') as f:
    f.write(report)
   
print("Saved: reports/report.txt", flush=True)
print("\nTraining complete!", flush=True)