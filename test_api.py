from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Test edge cases
test_cases = [
    ('Negative age', {'age': -5, 'resting_blood_pressure': 120, 'cholesterol': 210.5, 'max_heart_rate': 150, 'exercise_induced_angina': 0, 'num_major_vessels': 0, 'chest_pain_type': 'atypical_angina'}, True),
    ('Age zero', {'age': 0, 'resting_blood_pressure': 120, 'cholesterol': 210.5, 'max_heart_rate': 150, 'exercise_induced_angina': 0, 'num_major_vessels': 0, 'chest_pain_type': 'atypical_angina'}, True),
    ('Invalid exercise_induced_angina', {'age': 45, 'resting_blood_pressure': 120, 'cholesterol': 210.5, 'max_heart_rate': 150, 'exercise_induced_angina': 2, 'num_major_vessels': 0, 'chest_pain_type': 'atypical_angina'}, True),
    ('Valid request', {'age': 45, 'resting_blood_pressure': 120, 'cholesterol': 210.5, 'max_heart_rate': 150, 'exercise_induced_angina': 0, 'num_major_vessels': 0, 'chest_pain_type': 'atypical_angina'}, False),
]

print("Testing API Validation:")
print("-" * 60)

all_pass = True
for name, data, expect_error in test_cases:
    response = client.post('/predict', json=data)
    status = response.status_code
    
    if expect_error and status != 200:
        print(f"[OK] {name}: Correctly rejected (HTTP {status})")
    elif not expect_error and status == 200:
        print(f"[OK] {name}: Correctly accepted")
        print(f"  Response: {response.json()}")
    else:
        print(f"[ERROR] {name}: FAILED - Expected {'error' if expect_error else 'success'}, got HTTP {status}")
        all_pass = False

print("-" * 60)
print("All tests passed!" if all_pass else "Some tests failed!")
