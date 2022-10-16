"""
    Helper module for runnning predictions
"""
from pathlib import Path
import numpy as np # linear algebra
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pickle
import re

__version__ = "0.1.0"
    
IMAGE_W = 100
IMAGE_H = 100

classes = [
'pink primrose',
 'hard-leaved pocket orchid',
 'canterbury bells',
 'sweet pea',
 'english mar,igold'
 'tiger lily',
 'moon orchid',
 'bird of paradise',
 'monkshood',
 'globe thistle',
 'snapdragon',
 "colt's foot",
 'king protea',
 'spear thistle',
 'yellow iris',
 'globe-flower',
 'purple coneflower',
 'peruvian lily',
 'balloon flower',
 'giant white arum lily',
 'fire lily',
 'pincushion flower',
 'fritillary',
 'red ginger',
 'grape hyacinth',
 'corn poppy',
 'prince of wales feathers',
 'stemless gentian',
 'artichoke',
 'sweet william',
 'carnation',
 'garden phlox',
 'love in the mist',
 'mexican aster',
 'alpine sea holly',
 'ruby-lipped cattleya',
 'cape flower',
 'great masterwort',
 'siam tulip',
 'lenten rose',
 'barbeton daisy',
 'daffodil',
 'sword lily',
 'poinsettia',
 'bolero deep blue',
 'wallflower',
 'marigold',
 'buttercup',
 'oxeye daisy',
 'common dandelion',
 'petunia',
 'wild pansy',
 'primula',
 'sunflower',
 'pelargonium',
 'bishop of llandaff',
 'gaura',
 'geranium',
 'orange dahlia',
 'pink-yellow dahlia?',
 'cautleya spicata',
 'japanese anemone',
 'black-eyed susan',
 'silverbush',
 'californian poppy',
 'osteospermum',
 'spring crocus',
 'bearded iris',
 'windflower',
 'tree poppy',
 'gazania',
 'azalea',
 'water lily',
 'rose',
 'thorn apple',
 'morning glory',
 'passion flower',
 'lotus',
 'toad lily',
 'anthurium',
 'frangipani',
 'clematis',
 'hibiscus',
 'columbine',
 'desert-rose',
 'tree mallow',
 'magnolia',
 'cyclamen ',
 'watercress',
 'canna lily',
 'hippeastrum ',
 'bee balm',
 'ball moss',
 'foxglove',
 'bougainvillea',
 'camellia',
 'mallow',
 'mexican petunia',
 'bromelia',
 'blanket flower',
 'trumpet creeper',
 'blackberry lily',
]

flower_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((IMAGE_W,IMAGE_H)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))
])

def get_model():
    BASE_DIR = Path(__file__).resolve(strict = True).parent
    with open(f"{BASE_DIR}/trained_model-{__version__}.pkl", "rb") as f:
        model = pickle.load(f)

    return model

class PredictDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        image_path = self.metadata.iloc[idx, 0]
        image = Image.open(image_path).convert('RGB')
        sk_image = np.array(image)
        
        if self.transform:
            sk_image = self.transform(sk_image)
        
        return sk_image

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
    # image_path = ['/Users/julia.ju/Desktop/Flower-Identification-API/prediction_image/test_5.jpg']
    
    p_df = pd.DataFrame(
        {'image_path': [image_path]}
    )

    model = get_model()
    p_dataset = PredictDataset(p_df, flower_transform)
    loader =torch.utils.data.DataLoader(p_dataset, batch_size=1)
    
    prediction = predict(loader, model)
    
    return prediction

# if __name__=='__main__':
#         #folder structure in docker and be a bit different, so this is to be explicit
