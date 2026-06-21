import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, average_precision_score
import xgboost as xgb
import joblib

def preprocess_data(df):
    df.info(True)  # Check data types and missing values

    print("initial data preprocessing...")
    print(df.describe(include='all'))  # Check for outliers in age
    
    bad_rest_bp_rows = df[df['resting_blood_pressure'] <= 0]
    print(f"Count: {len(bad_rest_bp_rows)}")
    print(bad_rest_bp_rows)  
    
    bad_age_rows = df[df['age'] <= 0]
    print(f"Count: {len(bad_age_rows)}")
    print(bad_age_rows)  

    df['age'] = df['age'].where(df['age'] > 0, other=np.nan) 
    df['resting_blood_pressure'] = df['resting_blood_pressure'].where(df['resting_blood_pressure'] > 0, other=np.nan)  
   
    q1 = df['cholesterol'].quantile(0.25)
    q2 = df['cholesterol'].quantile(0.75)
    iqr = q2 - q1
    df['cholesterol'] = df['cholesterol'].where(df['cholesterol'] <= q2 + 3 * iqr, other=np.nan)  
    
    print("Data after cleaning:")
    print(df.describe(include='all'))  

    return df

def check_missing_values(df):
    label_1 = df[df['risk_label'] == 1] 
    label_0 = df[df['risk_label'] == 0] 
    print(f"Label 1 count: {label_1.shape[0]}, Label 0 count: {label_0.shape[0]}")
    missing_counts = df.isna().sum()
    return missing_counts.any()

def main():
    df = pd.read_csv('patients.csv')

    print("Loading and cleaning data...")
    df = preprocess_data(df)
    has_missing_values = check_missing_values(df)
    
    X = df.drop('risk_label', axis=1)
    y = df['risk_label']
    
    # 1. Strict Train/Test Split FIRST (Zero Data Leakage for Final Evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Reset indices to ensure safe alignment inside K-Fold loops
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    
    # 2. Define Preprocessing Steps
    numeric_features = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate']
    categorical_features = ['chest_pain_type',  'num_major_vessels', 'exercise_induced_angina']

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
        ], remainder='drop') 
    
    print("\n--- Starting Stratified 3-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    # Tracking the best number of trees across folds to train the final model
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nTraining Fold {fold + 1}...")
        
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Calculate scale_pos_weight for the current fold
        fold_scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
        
        # Fit preprocessor on training fold, transform both training & validation splits
        X_tr_trans = preprocessor.fit_transform(X_tr)
        X_val_trans = preprocessor.transform(X_val)
        
        # Initialize an isolated fold model
        fold_clf = xgb.XGBClassifier(
            scale_pos_weight=fold_scale_pos_weight, 
            random_state=42,
            n_estimators=500,       # Increased max estimators to let early stopping work
            learning_rate=0.05,     # Slightly lower rate pairs well with early stopping
            eval_metric='aucpr',    # Matches your triage optimization goal (PR-AUC)
            early_stopping_rounds=15,
            verbosity=3
        )
        
        # Fit with early stopping using the preprocessed validation split
        fold_clf.fit(
            X_tr_trans, y_tr,
            eval_set=[(X_val_trans, y_val)],
            verbose=False
        )
        
        # Evaluate fold performance using PR-AUC (Average Precision)
        val_preds_proba = fold_clf.predict_proba(X_val_trans)[:, 1]
        fold_score = average_precision_score(y_val, val_preds_proba)
        cv_scores.append(fold_score)
        best_iterations.append(fold_clf.best_iteration)
        
        print(f"Fold {fold + 1} Best Iteration: {fold_clf.best_iteration}, PR-AUC: {fold_score:.4f}")

    print(f"\nMean CV PR-AUC Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    avg_best_iteration = int(np.mean(best_iterations))
    print(f"Average optimal trees found via early stopping: {avg_best_iteration}")

    # 4. Fit Final Pipeline on full X_train using average optimal trees
    print("\nRetraining final model pipeline on entire training data...")
    final_scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    final_classifier = xgb.XGBClassifier(
        scale_pos_weight=final_scale_pos_weight, 
        random_state=42,
        n_estimators=avg_best_iteration,  # Set directly to the optimal number of trees found
        learning_rate=0.05,
        eval_metric='aucpr'
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', final_classifier)
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    # 5. Evaluation on Held-Out Test Set
    print("\nEvaluating on held-out test set...\n")
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    threshold = 0.4
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print("--- Classification Report (Threshold = 0.4) ---")
    print(classification_report(y_test, y_pred))
    
    # 6. Save Artifact
    joblib.dump(model_pipeline, 'triage_cv_model.joblib')
    print("Model saved to triage_cv_model.joblib")
    return model_pipeline

if __name__ == "__main__":
    main()