import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap

# Load Dataset
def load_data(file_path):
    df = pd.read_csv(r"D:\3 semister\Exiton\Task 1(Customer Churn Prediction)\WA_Fn-UseC_-Telco-Customer-Churn.csv")

    return df

# Exploratory Data Analysis and Preprocessing
def preprocess_data(df):
    # Drop irrelevant columns
    df.drop(['customerID'], axis=1, inplace=True)
    
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Churn'] = label_encoder.fit_transform(df['Churn'])  # 1 for churn, 0 for no churn
    
    # Convert binary columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    # One-hot encoding for categorical features
    df = pd.get_dummies(df, drop_first=True)
    
    return df

# Train-Test Split
def train_test_split_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

# Standardize Features
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

# Model Training using Logistic Regression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Model Explainability with SHAP
def explain_model_with_shap(model, X_train):
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_train)
    
    shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)

# Main Program
if __name__ == "__main__":
    # Load dataset (replace with the actual path)
    df = load_data(r'telco_customer_churn.csv')

    # Preprocess the data
    df = preprocess_data(df)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # Standardize the data
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    # Train the Logistic Regression model
    model = train_logistic_regression(X_train_scaled, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

    # Explain the model with SHAP
    explain_model_with_shap(model, X_train_scaled)
