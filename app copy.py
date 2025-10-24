from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Input format
class HouseData(BaseModel):
    area: float
    bedrooms: int
    age: int

@app.get("/")
def home():
    return {"message": " House Price Prediction API is running!"}

@app.post("/predict")
def predict(data: HouseData):
    X = np.array([[data.area, data.bedrooms, data.age]])
    predicted_price = model.predict(X)[0]
    return {"predicted_price": round(predicted_price, 2)}
