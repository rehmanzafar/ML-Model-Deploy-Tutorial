from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Load trained model
model = joblib.load("models/diabetes_model_lr.pkl")

# Define input schema
class DiabetesFeatures(BaseModel):
    data: list  # 10 features expected

app = FastAPI()

@app.post("/predict")
def predict(features: DiabetesFeatures):
    input_data = np.array([features.data])  # Shape: (1, 10)
    prediction = model.predict(input_data)
    return {"prediction": float(prediction[0])}

uvicorn.run(app, host="127.0.0.1", port=8000)