from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler



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
scaler = StandardScaler()
scaler.mean_ = np.array([3.845052083, 120.8945313, 69.10546875, 20.53645833, 79.79947917, 31.99257813, 33.24088542])
scaler.scale_ = np.array([3.369578062, 31.95179591, 19.34320163, 15.9418294, 115.2440024, 7.88416031, 11.76023154])

# Load the saved neural network model
diabetes_model = load_model('diabetes_model.h5')

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: ModelInput):
    
    # Convert input parameters to DataFrame
    input_df = pd.DataFrame([input_parameters.dict()])
    
    # Standardize the features
    input_scaled = scaler.transform(input_df)
    
    # Make predictions using the loaded model
    predictions = (diabetes_model.predict(input_scaled) > 0.5).astype(int)
    
    # Convert predictions to text labels
    outcome_mapping = {0: 'This person is diabetic', 1: 'This person is not diabetic'}
    text_labels = [outcome_mapping[label] for label in predictions.flatten()]

    return {'prediction': text_labels[0]}

#print(sys.version)