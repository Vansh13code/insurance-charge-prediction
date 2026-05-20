from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("insurance_model.pkl", "rb"))

class InsuranceData(BaseModel):
    age: int
    bmi: float
    children: int
    smoker: int
    sex: int

@app.get("/")
def home():
    return {"message": "Insurance Prediction API"}

@app.post("/predict")
def predict(data: InsuranceData):

    features = np.array([
        data.age,
        data.bmi,
        data.children,
        data.smoker,
        data.sex
    ]).reshape(1, -1)

    prediction = model.predict(features)

    return {
        "predicted_charge": float(prediction[0])
    }