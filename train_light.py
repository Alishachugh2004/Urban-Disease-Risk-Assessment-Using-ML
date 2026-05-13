"""Low Memory Training Script - Uses Decision Tree instead of RandomForest"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import gc

os.chdir('d:/Alisha_Project')

print("Loading data...")
df = pd.read_csv('data/dataset.csv')
print(f"Dataset: {df.shape}")

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Create risk levels based on environmental factors
def create_risk_level(row):
    humidity = row.get('humidity', 50)
    temp = row.get('temp', 25)
    cases = row.get('cases', 0)
    
    if humidity > 80 and temp > 30 and cases > 5000:
        return 2
    elif humidity > 60 or temp > 25:
        return 1
    else:
        return 0

df['risk_level'] = df.apply(create_risk_level, axis=1)
print(f"Risk distribution: {df['risk_level'].value_counts().to_dict()}")

# Features
feature_cols = ['temp', 'humidity', 'solarradiation', 'cases']
X = df[feature_cols].values
y = df['risk_level'].values

# Clear memory
del df
gc.collect()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Train Decision Tree (uses less memory than RandomForest)
print("Training Decision Tree...")
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl', compress=3)
joblib.dump(scaler, 'models/scaler.pkl', compress=3)
print("Saved: models/model.pkl")
print("Saved: models/scaler.pkl")

# Report
report = f"""===============================================
Urban Disease Spread Risk Assessment Model Report
===============================================

Model: DecisionTreeClassifier
Training Date: 2026-04-23

DATASET SUMMARY
---------------
Total Samples: {len(y)}
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

report += "===============================================\n"

with open('reports/report.txt', 'w') as f:
    f.write(report)
print("Saved: reports/report.txt")
print("\nDone!")