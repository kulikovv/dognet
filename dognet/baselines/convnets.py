import torch
import torch.nn as nn
import torch.nn.functional as F


class DownModule(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(DownModule, self).__init__()
        self.conv = nn.Conv2d(in_dims, out_dims, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1x1 = nn.Conv2d(out_dims, out_dims, 1, padding=0)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = F.dropout(self.conv(x), 0.1)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv1x1(x)
        return self.relu(x)


class UpModule(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(UpModule, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dims, out_dims, 2, stride=2)
        self.conv1x1 = nn.Conv2d(in_dims * 2, in_dims, 3, padding=1)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.smoothing = nn.Conv2d(out_dims, out_dims, 3, padding=1)

    def forward(self, x, y):
        x = F.upsample_bilinear(x, y.size()[2:])
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.relu(self.conv(x))
        return self.relu(self.smoothing(x))

class SUnet(nn.Module):
    """
    Extra small u-net
    """
    def __init__(self, in_dims, out_dims, k=1, y=8):
        super(SUnet, self).__init__()
        self.conv = nn.Conv2d(in_dims, y * k, 3, padding=1)
        self.d1 = DownModule(y * k, y * 2 * k)
        self.u1 = UpModule(y * 2 * k, y * k)
    def forward(self, x):
        x = self.conv(x)
        a = self.d1(x)    
        d = self.u1(a, a)
        return F.sigmoid(self.conv1x1(d)), None
    
class Unet(nn.Module):
    """
    Small u-net
    """
    def __init__(self, in_dims, out_dims, k=1, y=8):
        super(Unet, self).__init__()

        self.conv = nn.Conv2d(in_dims, y * k, 3, padding=1)
        self.d1 = DownModule(y * k, y * 2 * k)
        self.d2 = DownModule(y * 2 * k, y * 4 * k)

        self.u0 = UpModule(y * 4 * k, y * 2 * k)
        self.u1 = UpModule(y * 2 * k, y * k)

        self.conv1x1 = nn.Conv2d(y * k, out_dims, 1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        a = self.d1(x)
        b = self.d2(a)
        dd = self.u0(b, b)
        d = self.u1(dd, a)
        return F.sigmoid(self.conv1x1(d)), None
    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data.normal_(0., 2.)


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