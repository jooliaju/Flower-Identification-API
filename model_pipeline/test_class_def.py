import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
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
    
# model = MyCNN()

# x = torch.randn(32,3,100,100)
# y= model(x)

# print(y.shape)