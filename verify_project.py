import sys

print("\n" + "="*70)
print("TRIAGEML PROJECT - COMPREHENSIVE VERIFICATION")
print("="*70)

# Test 1: Import all dependencies
print("\n[1] Checking dependencies...")
try:
    import pandas
    import numpy
    import sklearn
    import fastapi
    import uvicorn
    import pydantic
    import joblib
    import httpx
    print("    [OK] All dependencies imported successfully")
except ImportError as e:
    print(f"    [ERROR] Missing dependency: {e}")
    sys.exit(1)

# Test 2: Load data
print("\n[2] Checking data file...")
try:
    import pandas as pd
    df = pd.read_csv('patients.csv')
    print(f"    [OK] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"    [ERROR] Error loading data: {e}")
    sys.exit(1)

# Test 3: Load model
print("\n[3] Checking trained model...")
try:
    import joblib
    model = joblib.load('triage_model.joblib')
    print("    [OK] Model loaded successfully")
except Exception as e:
    print(f"    [ERROR] Error loading model: {e}")
    sys.exit(1)

# Test 4: Test prediction
print("\n[4] Testing model prediction...")
try:
    test_data = {
        'age': 45,
        'resting_blood_pressure': 120,
        'cholesterol': 210.5,
        'max_heart_rate': 150,
        'exercise_induced_angina': 0,
        'num_major_vessels': 0,
        'chest_pain_type': 'atypical_angina'
    }
    test_df = pd.DataFrame([test_data])
    prediction = model.predict_proba(test_df)[0]
    print(f"    [OK] Prediction successful: {prediction}")
except Exception as e:
    print(f"    [ERROR] Prediction error: {e}")
    sys.exit(1)

# Test 5: Test FastAPI app
print("\n[5] Testing FastAPI application...")
try:
    from fastapi.testclient import TestClient
    from app import app
    
    client = TestClient(app)
    response = client.post('/predict', json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"    [OK] API endpoint working: {result}")
    else:
        print(f"    [ERROR] API error (status {response.status_code}): {response.json()}")
        sys.exit(1)
except Exception as e:
    print(f"    [ERROR] API error: {e}")
    sys.exit(1)

# Test 6: Test input validation
print("\n[6] Testing input validation...")
try:
    invalid_data = {
        'age': -5,
        'resting_blood_pressure': 120,
        'cholesterol': 210.5,
        'max_heart_rate': 150,
        'exercise_induced_angina': 0,
        'num_major_vessels': 0,
        'chest_pain_type': 'atypical_angina'
    }
    response = client.post('/predict', json=invalid_data)
    
    if response.status_code != 200:
        print(f"    [OK] Invalid input correctly rejected (HTTP {response.status_code})")
    else:
        print(f"    [ERROR] Invalid input was not rejected!")
        sys.exit(1)
except Exception as e:
    print(f"    [ERROR] Validation error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("[OK] ALL TESTS PASSED - PROJECT IS READY TO RUN!")
print("="*70)
print("\nTo start the API server, run:")
print("  uvicorn app:app --reload")
print("\nAPI will be available at: http://127.0.0.1:8000")
print("="*70 + "\n")
