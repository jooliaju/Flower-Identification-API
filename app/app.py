import uvicorn ##ASGI
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import shutil
from app.model.model import predict_pipeline #get_model
from app.model.model import __version__ as model_version

app = FastAPI()

class InputImage(BaseModel):
    text: str

class PredictionOut(BaseModel):
    flower_prediction: str

@app.get("/")
def home():
    """
    Returns:
        string: message indicating if API is up
    """
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict")
def predict(image: UploadFile = File(...)):
    """Pediction for flower image

    Args:
        image (UploadFile): image to upload

    Returns:
        _type_: _description_
    """

    file_location = f"/app/{image.filename}"

    #saving image to /app/ on docker container
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(image.file, file_object)
    
    prediction = predict_pipeline(file_location)
    return {"info": f"file '{image.filename}' saved at '{file_location}'",
            "prediction": prediction
            }


if __name__ == '__main__':
    uvicorn.run(app, host= "127.0.0.1", port=8000)
#uvicorn app:app --reload