from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

app = FastAPI()

class InputImage(BaseModel):
    text: str

class PredictionOut(BaseModel):
    flower_prediction: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(image: UploadFile):
    flower = predict_pipeline(image.file)
    return {"flower_prediction": flower}