from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(
    root_path="/proxy/8000"
)

MODEL_PATH = Path(
    "../components/train_random_forest/model_export/model.pkl"
)

model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    neighbourhood_group: str
    room_type: str
    minimum_nights: int
    number_of_reviews: int
    reviews_per_month: float
    calculated_host_listings_count: int
    availability_365: int


@app.get("/")
def home():
    return {
        "message": "ML API is running"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    data = pd.DataFrame([{
        "neighbourhood_group": request.neighbourhood_group,
        "room_type": request.room_type,
        "minimum_nights": request.minimum_nights,
        "number_of_reviews": request.number_of_reviews,
        "reviews_per_month": request.reviews_per_month,
        "calculated_host_listings_count": request.calculated_host_listings_count,
        "availability_365": request.availability_365
    }])

    prediction = model.predict(data)

    return {
        "prediction": prediction.tolist()
    }