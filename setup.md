# TriageML Project Setup & Demonstration Guide

This guide details how to set up the Conda environment, run the project, verify its components, and demonstrate the API to stakeholders or management.

---

## 1. Project Overview
**TriageML** is a clinical risk-scoring service powered by Machine Learning. It uses patient vitals to predict whether a patient is at high or low risk for cardiac events.
* **Backend Framework**: FastAPI (Python 3.13)
* **Model**: RandomForestClassifier with a clinical triage-adjusted recall threshold (0.4)
* **Features Used**: Age, resting blood pressure, cholesterol, max heart rate, chest pain type, exercise-induced angina, and number of major vessels.

---

## 2. Environment Setup (Conda)

### Step 1: Verify Conda Installation
Ensure Anaconda or Miniconda is installed on your system. You can verify it by opening your terminal (CMD or PowerShell) and running:
```bash
conda --version
```

### Step 2: Create the `triageml` Environment
Create the environment from the provided configuration file (`environment.yml`). This ensures all correct dependency versions (including Python 3.13, scikit-learn 1.9.0, Pandas, and FastAPI) are isolated.
```bash
conda env create -f environment.yml
```

### Step 3: Activate the Environment
To work with the project, activate the conda environment:
```bash
conda activate triageml
```

---

## 3. Data & Model Verification

### Step 1: Inspect Dataset
Ensure `patients.csv` is present in the root directory. It contains raw patient records used to train the classification model.

### Step 2: Train/Re-Train the Model
To generate the model artifact (`triage_model.joblib`) matching your environment's library versions, run the training script:
```bash
python train.py
```
**Expected Output:**
```text
Loading and cleaning data...
Training model...
Evaluating on held-out test set...

--- Classification Report (Threshold = 0.4) ---
              precision    recall  f1-score   support

           0       0.97      0.95      0.96       121
           1       0.84      0.91      0.88        35

    accuracy                           0.94       156
   macro avg       0.91      0.93      0.92       156
weighted avg       0.94      0.94      0.94       156

Model saved to triage_model.joblib
```
*(Note: A threshold of 0.4 is applied to favor recall, which is optimal for clinical triage to minimize false negatives).*

### Step 3: Run the Verification Suite
Run the comprehensive verification script to validate dependencies, data integrity, model loading, mock predictions, and API client routes:
```bash
python verify_project.py
```
**Expected Output:**
```text
======================================================================
TRIAGEML PROJECT - COMPREHENSIVE VERIFICATION
======================================================================
[1] Checking dependencies...
    [OK] All dependencies imported successfully
[2] Checking data file...
    [OK] Data loaded: 803 rows, 8 columns
[3] Checking trained model...
    [OK] Model loaded successfully
[4] Testing model prediction...
    [OK] Prediction successful: [1. 0.]
[5] Testing FastAPI application...
    [OK] API endpoint working: {'predicted_risk_label': 0, 'probability_high_risk': 0.0}
[6] Testing input validation...
    [OK] Invalid input correctly rejected (HTTP 422)
======================================================================
[OK] ALL TESTS PASSED - PROJECT IS READY TO RUN!
======================================================================
```

---

## 4. Running the API Server

Launch the FastAPI application server locally using Uvicorn:
```bash
uvicorn app:app --reload
```
The server will start and bind to `http://127.0.0.1:8000`.

---

## 5. Live Demonstration Guide

To demonstrate the API's behavior to your boss, you can send test payloads to the running server. Use either of the following methods depending on your operating system/shell:

### Option A: Using PowerShell (Windows Default)

#### 1. Test a Valid Patient Intake (Low Risk Example)
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"age": 45, "resting_blood_pressure": 120, "cholesterol": 210.5, "max_heart_rate": 150, "exercise_induced_angina": 0, "num_major_vessels": 0, "chest_pain_type": "atypical_angina"}'
```
**Expected Response:**
```json
predicted_risk_label probability_high_risk
-------------------- ---------------------
                   0                   0.0
```

#### 2. Test Input Validation Rejection (Invalid/Negative Age)
This demonstrates the robust data boundary checking built into the API.
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"age": -5, "resting_blood_pressure": 120, "cholesterol": 210.5, "max_heart_rate": 150, "exercise_induced_angina": 0, "num_major_vessels": 0, "chest_pain_type": "atypical_angina"}'
```
**Expected Response:**
*(Returns a `422 Unprocessable Entity` validation error pointing directly to the age field)*
```json
{"detail":[{"type":"greater_than","loc":["body","age"],"msg":"Input should be greater than 0","input":-5,"ctx":{"gt":0}}]}
```

---

### Option B: Using standard bash `curl` (Linux/macOS)

#### 1. Test Valid Patient Intake
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"age": 45, "resting_blood_pressure": 120, "cholesterol": 210.5, "max_heart_rate": 150, "exercise_induced_angina": 0, "num_major_vessels": 0, "chest_pain_type": "atypical_angina"}'
```

#### 2. Test Input Validation Rejection
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"age": -5, "resting_blood_pressure": 120, "cholesterol": 210.5, "max_heart_rate": 150, "exercise_induced_angina": 0, "num_major_vessels": 0, "chest_pain_type": "atypical_angina"}'
```

---

## 6. Interactive Documentation (Swagger UI)
FastAPI automatically generates a user-friendly UI to interact with and test all API endpoints:
1. Open your browser and navigate to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Expand the `/predict` POST route.
3. Click **"Try it out"**, modify the JSON payload, and click **"Execute"** to view real-time predictions and response headers.
