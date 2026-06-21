# TriageML 

## Setup & Execution
1. Ensure `patients.csv` is in the root directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train.py`
4. Start the API: `uvicorn app:app --reload`

## API Testing (cURL Examples)

**1. Valid Request (Successful Prediction):**
```bash
curl -X 'POST' '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' -H 'Content-Type: application/json' -d '{"age": 45, "resting_blood_pressure": 120, "cholesterol": 210.5, "max_heart_rate": 150, "exercise_induced_angina": 0, "num_major_vessels": 0, "chest_pain_type": "atypical_angina"}'