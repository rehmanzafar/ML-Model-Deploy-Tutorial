from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import uvicorn

# Load trained model
model = joblib.load("models/diabetes_model_lr.pkl")

# Define input schema
class DiabetesFeatures(BaseModel):
    data: List[float]

app = FastAPI()

@app.post("/predict-batch")
def predict(features: List[DiabetesFeatures]):
    input_data = np.array([feature.data for feature in features])
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}

uvicorn.run(app, host="127.0.0.1", port=8000)