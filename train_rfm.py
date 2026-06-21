import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Drop biologically impossible values
    df = df[df['age'] > 0]
    df = df[df['resting_blood_pressure'] > 0]
    return df

def main():
    try:
        model = joblib.load('triage_rfm.joblib')
        print("Using pre-trained model.")
        return model
    except FileNotFoundError:
        pass
    
    print("Loading and cleaning data...")
    df = load_and_clean_data('patients.csv')
    
    X = df.drop('risk_label', axis=1)
    y = df['risk_label']
    
    # 1. Strict Train/Test Split FIRST (Zero Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Define Preprocessing Steps
    numeric_features = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'num_major_vessels']
    categorical_features = ['chest_pain_type']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough') 
    
    # 3. Build Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            class_weight='balanced', 
            random_state=42,
            n_estimators=100
        ))
    ])
    
    print("Training model...")
    model_pipeline.fit(X_train, y_train)
    
    # 4. Evaluation
    print("Evaluating on held-out test set...\n")
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    # Custom threshold to favor recall in medical triage
    threshold = 0.4
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print("--- Classification Report (Threshold = 0.4) ---")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Artifact
    joblib.dump(model_pipeline, 'triage_rfm.joblib')
    print("Model saved to triage_rfm.joblib")
    return model_pipeline

if __name__ == "__main__":
    main()