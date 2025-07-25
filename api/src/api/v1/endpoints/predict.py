from fastapi import APIRouter, Depends
from src.models.predict_input import PredictInput
from src.services.predictor import predict_pipeline
from typing import Dict, Any


router = APIRouter()

@router.post("/predict")
def predict(data:Dict[str, Any]):

    prediction = predict_pipeline(data)
    return {"prediction": prediction}
