"""
    Helper module for running predictions
"""
from pathlib import Path
import pandas as pd
import torch
from app.util import PredictDataset, MyCNN, classes, flower_transform

__version__ = "0.1.0"
    
NUM_OUT_CH = [8,16]
IMAGE_W = 100
IMAGE_H = 100
BATCH_SIZE = 64
NUM_EPOCHS = 4
NUM_CLASSES = 102

BASE_DIR = Path(__file__).resolve(strict = True).parent

def predict(loader, model):

    model.eval()
    
    with torch.no_grad(): #tell pytorch its not a training loop
        for x in loader:
            
            y_hat = model(x)
            _, p_index = y_hat.max(1)
            predicted = p_index.numpy()[0]
            
            predicted_class = classes[predicted]
            print(f"predicted: {predicted_class}")
    return predicted_class
            
def predict_pipeline(image_path:str):
    """Generate prediction of flower image

    Args:
        image_path (str): Path to the image
    """
    p_df = pd.DataFrame(
        {'image_path': [image_path]}
    )    
    model = MyCNN(num_channels=3, num_out_ch=NUM_OUT_CH,img_w=IMAGE_W, img_h= IMAGE_H, num_classes= NUM_CLASSES)
    model.load_state_dict(torch.load(f"{BASE_DIR}/trained_model-{__version__}.pt"))

    p_dataset = PredictDataset(p_df, flower_transform)
    loader =torch.utils.data.DataLoader(p_dataset, batch_size=1)
    prediction = predict(loader, model)
    
    return prediction

# if __name__=='__main__':
#         #folder structure in docker and be a bit different, so this is to be explicit
