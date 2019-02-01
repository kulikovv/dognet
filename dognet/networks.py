import torch.nn as nn
import torch.nn.functional as F
import torch

from .dogs import DoG2DIsotropic, DoG2DAnisotropic, DoG3DIsotropic

class Simple3DNetwork(nn.Module):
    def __init__(self, in_channels, filter_size=9, k=4,depth = 3, return_intermediate=False, learn_amplitude=False,
                 dog_class=DoG3DIsotropic):
        """
        Create a simple 3D DoGnet
        :param in_channels: input data number of channels
        :param filter_size: filter window size must be even (3,5,7,9,11 etc)
        :param k: number of filters for each image 
        :param d: depth of voxel volume
        :param return_intermediate: returns the output of DoG filters during the inference
        :param dog_class: the class of Difference of Gaussian used
        """
        super(Simple3DNetwork, self).__init__()
        self.conv1 = dog_class(filter_size, in_channels, k, depth , learn_amplitude=learn_amplitude)
        self.conv2 = nn.Conv2d(in_channels * k, 1, 1)
        self.return_intermediate = return_intermediate
        
    def weights_init(self):
        self.conv1.weights_init()
        self.conv2.weight.data.fill_(1.)
        self.conv2.bias.data.fill_(0.)
 
    def forward(self, x):
        y = self.conv1(x)
        x = self.conv2(y.squeeze(2))
        if self.return_intermediate:
            return F.sigmoid(x), y
        return F.sigmoid(x), None
    
    def get_reg_params(self):
        return self.conv2.parameters()
    

class SimpleNetwork(nn.Module):
    def __init__(self, in_channels, filter_size=9, k=4, return_intermediate=False, learn_amplitude=False,
                 dog_class=DoG2DIsotropic):
        """
        Create a simple DoGnet
        :param in_channels: input data number of channels
        :param filter_size: filter window size must be even (3,5,7,9,11 etc)
        :param k: number of filters for each image 
        :param return_intermediate: returns the output of DoG filters during the inference
        :param dog_class: the class of Difference of Gaussian used
        """
        super(SimpleNetwork, self).__init__()
        self.conv1 = dog_class(filter_size, in_channels,k, learn_amplitude=learn_amplitude)
        self.conv2 = nn.Conv2d(in_channels*k, 2, kernel_size=(1,1))
        
        self.return_intermediate = return_intermediate
        
    def weights_init(self):
        self.conv1.weights_init()
        self.conv2.weight.data.fill_(1.)
        self.conv2.bias.data.fill_(0.)
 
    def forward(self, x):
        y = self.conv1(x)
        #y = F.sigmoid(x)
        xx = F.sigmoid(self.conv2(y))
        xxx = xx[:,0:1].mul(xx[:,1:])
        if self.return_intermediate:
            return xxx, y
        return xxx, None
    
    def get_reg_params(self):
        return self.conv2.parameters()


class SimpleIsotropic(SimpleNetwork):
    def __init__(self, in_channels, filter_size=9, k=4, return_intermediate=False, learn_amplitude=False):
        super(SimpleIsotropic, self).__init__(in_channels, filter_size, k, return_intermediate, learn_amplitude,
                                              DoG2DIsotropic)


class SimpleAnisotropic(SimpleNetwork):
    def __init__(self, in_channels, filter_size=9, k=4, return_intermediate=False, learn_amplitude=False):
        super(SimpleAnisotropic, self).__init__(in_channels, filter_size, k, return_intermediate, learn_amplitude,
                                                DoG2DAnisotropic)


class DeepNetwork(nn.Module):
    def __init__(self, in_channels, filter_size=9, k=5, n_layers=2, learn_amplitude=False, dog_class=DoG2DIsotropic):
        """
        Create a Deep dognet
        :param in_channels: input data number of channels
        :param filter_size: filter window size must be even (3,5,7,9,11 etc)
        :param k: number of filters for each image 
        :param n_layers: number of repeats
        :param dog_class: the class of Difference of Gaussian used
        """
        super(DeepNetwork, self).__init__()
        
        layer = []

        for i in range(n_layers):
            layer.append(dog_class(filter_size, in_channels,k, learn_amplitude=learn_amplitude))
            layer.append(nn.Conv2d(in_channels*k, in_channels, 1))
            layer.append(nn.ReLU())
          

        self.net = nn.Sequential(*layer[:-2])
        self.final_convolution = nn.Conv2d(in_channels*k, 2, 1)
        
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(0.)
            elif isinstance(m, DoG2DIsotropic) or isinstance(m, DoG2DAnisotropic):
                m.weights_init()

    def forward(self, x):
        x = self.net(x)
        xx = F.sigmoid(self.final_convolution(x))
        xxx = xx[:,0:1].mul(xx[:,1:])
        return xxx, None
    
    def get_reg_params(self):
        return self.final_convolution.parameters()


class DeepIsotropic(DeepNetwork):
    def __init__(self, in_channels, filter_size=9, k=4, n_layers=2, learn_amplitude=False):
        super(DeepIsotropic, self).__init__(in_channels, filter_size, k, n_layers, learn_amplitude, DoG2DIsotropic)


class DeepAnisotropic(DeepNetwork):
    def __init__(self, in_channels, filter_size=9, k=4, n_layers=2, learn_amplitude=False):
        super(DeepAnisotropic, self).__init__(in_channels, filter_size, k, n_layers, learn_amplitude, DoG2DAnisotropic)


