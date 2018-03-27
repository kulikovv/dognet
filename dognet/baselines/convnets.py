import torch
import torch.nn as nn
import torch.nn.functional as F


class U_net(nn.Module):
    def __init__(self, in_dims, k=1):
        super(U_net, self).__init__()
        self.conv1 = nn.Conv2d(in_dims, k, 3, padding=1)
        self.relu =  nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d( k, k*2, 3, padding=1)
        self.conv3 = nn.Conv2d( k*2, k*4, 3, padding=1)
        self.conv4 = nn.Conv2d( k*4, k*2, 3, padding=1)
        self.conv5 = nn.Conv2d( k*2, 1, 3, padding=1)
        self.up = nn.ConvTranspose2d(k*2, k, 2, stride=2)
        
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data.normal_(0., 2.)
                
    def forward(self,x):
        x = self.relu(self.conv1(x))
        g = self.relu(self.conv2(self.pool(x)))
        g = self.relu(self.conv3(g))
        g = self.relu(self.conv4(g))
        
        g = self.relu(self.up(g))
        g = F.upsample(g, size=[x.size(2),x.size(3)],mode='bilinear')
        x = torch.cat([x, g], dim=1)
        return F.sigmoid(self.conv5(x)),None
    
class Basic(nn.Module):
    def __init__(self, in_dims, k=1):
        super(Basic, self).__init__()
        self.conv1 = nn.Conv2d(in_dims, k, 11, padding=5,groups=in_dims)
        self.relu =  nn.ReLU()
        self.conv2 = nn.Conv2d(k, 1, 1, padding=0)
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data.normal_(0., 2.)
    def forward(self,x):
        x = self.relu(self.conv1(x))
        return F.sigmoid(self.conv2(x)),None
    

class FCN(nn.Module):
    """
    A simple fully convolutional network for segmentation
    """
    def __init__(self, in_dims, k=1, n_layers=2):
        super(FCN, self).__init__()
        self.begin = nn.Conv2d(in_dims, in_dims*k, 1, padding=0)

        layer=[]
        for i in range(n_layers):
            layer.append(nn.Conv2d(in_dims*k, in_dims * k, 3, padding=1))
            layer.append(nn.BatchNorm2d(in_dims * k))
            layer.append(nn.ReLU())

        self.net = nn.Sequential(*layer[:-1])

        self.final = nn.Conv2d(in_dims * k, 1, 1, padding=0)
        
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data.normal_(0., 2.)
                
    def forward(self, x):
        x = self.begin(x)
        x = self.net(x)
        x = self.final(x)
        return F.sigmoid(x), None