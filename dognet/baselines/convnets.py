import torch
import torch.nn as nn
import torch.nn.functional as F

def disable_bias(net):
    for n in net.modules():
        if isinstance(n,torch.nn.Conv2d) or isinstance(n,torch.nn.ConvTranspose2d):
            if n.bias is not None:
                n.bias.requires_grad=False
                n.bias.zero_()
 

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data.normal_(0., 2.)   
    
    def get_reg_params(self):
        return self.parameters()
    
        
        
class Direct(Base):
    def __init__(self, in_dims):
        super(Direct, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_dims,5,15,padding=7)
        self.conv2 = torch.nn.Conv2d(5,2,1)
        self.relu =  nn.ReLU()
                       
    def forward(self,x):
        x = self.relu(self.conv1(x))
        xx = F.sigmoid(self.conv2(x))
        xxx = xx[:,0:1].mul(xx[:,1:])
        return xxx,None
    
class FCN(Base):
    def __init__(self, in_dims):
        super(FCN, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_dims,6,7,padding=3),
                           torch.nn.ReLU(),
                           torch.nn.Conv2d(6,6,7,padding=3),
                           torch.nn.ReLU(),
                           torch.nn.Conv2d(6,6,3,padding=1),
                           torch.nn.ReLU())
        self.conv2 = torch.nn.Conv2d(6,2,1)
         
    
    def forward(self,x):
        x = self.conv1(x)
        xx = F.sigmoid(self.conv2(x))
        xxx = xx[:,0:1].mul(xx[:,1:])
        return xxx,None
        
        
class U_net(Base):
    def __init__(self, in_dims, k=1):
        super(U_net, self).__init__()
        self.conv1 = nn.Conv2d(in_dims, k, 3, padding=1)
        self.relu =  nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d( k, k, 3, padding=1)
        self.conv3 = nn.Conv2d( k, k, 3, padding=1)
        self.conv5 = nn.Conv2d( 2*k, 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(k, k, 2, stride=2)
                        
    def forward(self,x):
        x = self.relu(self.conv1(x))
        g = self.relu(self.conv2(self.pool(x)))
        g = self.relu(self.conv3(g))
        
        g = self.relu(self.up(g))
        g = F.upsample(g, size=[x.size(2),x.size(3)],mode='bilinear',align_corners=True)
        x = torch.cat([x, g], dim=1)
        xx = F.sigmoid(self.conv5(x))
        xxx = xx[:,0:1].mul(xx[:,1:])
        return xxx,None
    

"""
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
                
    def get_reg_params(self):
        return self.parameters()
                
    def forward(self,x):
        x = self.relu(self.conv1(x))
        g = self.relu(self.conv2(self.pool(x)))
        g = self.relu(self.conv3(g))
        g = self.relu(self.conv4(g))
        
        g = self.relu(self.up(g))
        g = F.upsample(g, size=[x.size(2),x.size(3)],mode='bilinear')
        x = torch.cat([x, g], dim=1)
        return F.sigmoid(self.conv5(x)),None
    
class Direct(nn.Module):
    def __init__(self, in_dims, k=1, kernel=11):
        super(Direct, self).__init__()
        self.conv1 = nn.Conv2d(in_dims, k, kernel, padding=5,groups=in_dims)
        self.relu =  nn.ReLU()
        self.conv2 = nn.Conv2d(k, 1, 1, padding=0)
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data.normal_(0., 2.)
    def forward(self,x):
        x = self.relu(self.conv1(x))
        return F.sigmoid(self.conv2(x)),None
    
    def get_reg_params(self):
        return self.parameters()
    

class FCN(nn.Module):
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
                
    def get_reg_params(self):
        return self.parameters()
                
    def forward(self, x):
        x = self.begin(x)
        x = self.net(x)
        x = self.final(x)
        return F.sigmoid(x), None
"""