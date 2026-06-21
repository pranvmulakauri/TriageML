from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from train import main

# Load the saved model artifact globally
model = main()

app_api = FastAPI(title="TriageML Risk Scoring API")

@app_api.get("/")
def read_root():
    return {
        "message": "Welcome to the TriageML Risk Scoring API",
        "documentation": "Go to /docs for the interactive API documentation",
        "status": "online"
    }


# Strict Input Validation Schema
class PatientIntake(BaseModel):
    age: int = Field(..., gt=0, description="Age must be a positive integer")
    resting_blood_pressure: int = Field(..., gt=0, description="BP must be positive")
    cholesterol: float
    max_heart_rate: int
    exercise_induced_angina: int = Field(..., ge=0, le=1)
    num_major_vessels: int
    chest_pain_type: str

@app_api.post("/predict")
def predict_risk(patient: PatientIntake):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact not found.")
    
    input_df = pd.DataFrame([patient.model_dump()])
    
    try:
        probabilities = model.predict_proba(input_df)[0]
        prob_high_risk = float(probabilities[1])
        
        # Clinical threshold
        threshold = 0.4
        final_label = 1 if prob_high_risk >= threshold else 0
        
        return {
            "predicted_risk_label": final_label,
            "probability_high_risk": round(prob_high_risk, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    