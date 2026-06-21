import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

def preprocess_data(df):
    df.info(True)  # Check data types and missing values

    print(df.describe(include='all'))  # Check for outliers in age
    
    # age and resting_blood_pressure should be > 0. Let's check how many rows violate this and if other columns look suspicious in those rows.

    bad_rest_bp_rows = df[df['resting_blood_pressure'] <= 0]
    print(f"Count: {len(bad_rest_bp_rows)}")
    print(bad_rest_bp_rows)  # look at the full rows — do other columns look corrupt too?
    
    bad_age_rows = df[df['age'] <= 0]
    print(f"Count: {len(bad_age_rows)}")
    print(bad_age_rows)  # look at the full rows — do other columns look corrupt too?

    df['age'] = df['age'].where(df['age'] > 0, other=np.nan) # Replace non-positive ages with NaN
    
    df['resting_blood_pressure'] = df['resting_blood_pressure'].where(df['resting_blood_pressure'] > 0, other=np.nan)  # Replace non-positive blood pressure with NaN
    return df

def check_missing_values(df):
    label_1 = df[df['risk_label'] == 1] # Shows the rows where risk_label is 1 (high risk)
    label_0 = df[df['risk_label'] == 0] # Shows the rows where risk_label is 0 (low risk)
    print(f"Label 1 count: {label_1.shape[0]}, Label 0 count: {label_0.shape[0]}")
    missing_counts = df.isna().sum()
    # print(missing_counts)
    return missing_counts.any()

def main():
    try:
        model = joblib.load('triage_model.joblib')
        print("Using pre-trained model.")
        return model
    except FileNotFoundError:
        pass
    
    df = pd.read_csv('patients.csv')

    print("Loading and cleaning data...")
    df = preprocess_data(df)
    has_missing_values = check_missing_values(df)
    
    X = df.drop('risk_label', axis=1)
    y = df['risk_label']
    
    # 1. Strict Train/Test Split FIRST (Zero Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Define Preprocessing Steps
    numeric_features = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'num_major_vessels']
    categorical_features = ['chest_pain_type']

    num_steps = []

    if has_missing_values:
        num_steps.append(('imputer', SimpleImputer(strategy='mean')))
    
    num_steps.append(('scaler', StandardScaler()))
    
    numeric_transformer = Pipeline(steps=num_steps)

    cat_steps = []

    if has_missing_values:
        cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
    
    cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))

    categorical_transformer = Pipeline(steps=cat_steps)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough') 
    
    # We calculate scale_pos_weight due to imbalanced classes (more low risk than high risk patients).
    # This helps XGBoost focus more on the minority class.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    #3. Build Pipeline with XGBoost

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, 
            random_state=42,
            n_estimators=100,
            eval_metric='logloss' 
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
    joblib.dump(model_pipeline, 'triage_model.joblib')
    print("Model saved to triage_model.joblib")
    return model_pipeline

if __name__ == "__main__":
    main()