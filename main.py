from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    Age: int

# Load the scaler for standardization
scaler = joblib.load('scaler.joblib')

# Load the saved Scikit-Learn model
diabetes_model = joblib.load('diabetes_model.joblib')

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: ModelInput):
    
    # Convert input parameters to DataFrame
    input_df = pd.DataFrame([input_parameters.dict()])
    
    # Standardize the features
    input_scaled = scaler.transform(input_df)
    
    # Make predictions using the loaded model
    predictions = diabetes_model.predict(input_scaled)
    
    # Convert predictions to text labels
    outcome_mapping = {0: 'This person is diabetic', 1: 'This person is not diabetic'}
    text_labels = [outcome_mapping[label] for label in predictions.flatten()]

    return {'prediction': text_labels[0]}
