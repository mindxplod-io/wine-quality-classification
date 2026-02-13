"""
Wine Quality Classification - Model Training Script
This script trains 6 different classification models and evaluates them
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading Wine Quality Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

print(f"Dataset shape: {df.shape}")
print(f"\nDataset info:")
print(df.info())
print(f"\nTarget distribution:")
print(df['quality'].value_counts().sort_index())

# Prepare features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Convert to binary classification for simplicity (Good wine >= 6, Bad wine < 6)
y_binary = (y >= 6).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0,
        'Precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'F1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

    return results

# Dictionary to store results
all_results = []

# 1. Logistic Regression
print("\n" + "="*50)
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
all_results.append(lr_results)
print(f"Logistic Regression - Accuracy: {lr_results['Accuracy']:.4f}, F1: {lr_results['F1']:.4f}")

# Save model
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# 2. Decision Tree Classifier
print("\n" + "="*50)
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train_scaled, y_train)
dt_results = evaluate_model(dt_model, X_test_scaled, y_test, "Decision Tree")
all_results.append(dt_results)
print(f"Decision Tree - Accuracy: {dt_results['Accuracy']:.4f}, F1: {dt_results['F1']:.4f}")

# Save model
with open('dt_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

# 3. K-Nearest Neighbors
print("\n" + "="*50)
print("Training K-Nearest Neighbors...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_results = evaluate_model(knn_model, X_test_scaled, y_test, "K-Nearest Neighbors")
all_results.append(knn_results)
print(f"KNN - Accuracy: {knn_results['Accuracy']:.4f}, F1: {knn_results['F1']:.4f}")

# Save model
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

# 4. Naive Bayes (Gaussian)
print("\n" + "="*50)
print("Training Naive Bayes...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_results = evaluate_model(nb_model, X_test_scaled, y_test, "Naive Bayes")
all_results.append(nb_results)
print(f"Naive Bayes - Accuracy: {nb_results['Accuracy']:.4f}, F1: {nb_results['F1']:.4f}")

# Save model
with open('nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

# 5. Random Forest (Ensemble)
print("\n" + "="*50)
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)
rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
all_results.append(rf_results)
print(f"Random Forest - Accuracy: {rf_results['Accuracy']:.4f}, F1: {rf_results['F1']:.4f}")

# Save model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# 6. XGBoost (Ensemble)
print("\n" + "="*50)
print("Training XGBoost...")
xgb_model = XGBClassifier(random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1)
xgb_model.fit(X_train_scaled, y_train)
xgb_results = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
all_results.append(xgb_results)
print(f"XGBoost - Accuracy: {xgb_results['Accuracy']:.4f}, F1: {xgb_results['F1']:.4f}")

# Save model
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Create results DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.round(4)

print("\n" + "="*70)
print("FINAL RESULTS - ALL MODELS")
print("="*70)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('model_results.csv', index=False)
print("\nResults saved to 'model_results.csv'")

# Save test data for Streamlit app
test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
test_data['quality'] = y_test.values
test_data.to_csv('test_data.csv', index=False)
print("Test data saved to 'test_data.csv'")

print("\nAll models trained and saved successfully!")
