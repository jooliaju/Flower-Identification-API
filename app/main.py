from fastapi import FastAPI, File, UploadFile
import shutil
from app.model.model import predict_pipeline #get_model
from app.model.model import __version__ as model_version

app = FastAPI()

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
    return {"info": f"file {image.filename} saved at {file_location}",
            "prediction": prediction
            }
