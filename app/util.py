import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

IMAGE_W = 100
IMAGE_H = 100

class PredictDataset(Dataset):
    """Converts to torch dataset

    Args:
        Dataset (_type_): _description_
    """
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

class MyCNN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,num_channels=3,num_out_ch=[8,16], img_w =100, img_h=100, num_classes=102):
        super(MyCNN,self).__init__()
        #have 2 simple convolutional layers
        #output of second layer goes into a fully connect layer, which has the end layer (output that has predictions)
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_out_ch[0], kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=num_out_ch[0], out_channels=num_out_ch[1], kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc = nn.Linear(in_features= int(img_w/4)*int(img_h/4)*num_out_ch[1], out_features= num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.fc(x.reshape(x.shape[0],-1))
        
        return x
    
flower_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((IMAGE_W,IMAGE_H)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))
])

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