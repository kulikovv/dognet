import torch.nn as nn
import torch.nn.functional as F

from dogs import DoG2DIsotropic, DoG2DAnisotropic


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
        self.conv1 = dog_class(filter_size, in_channels * k, learn_amplitude=learn_amplitude)
        self.conv2 = nn.Conv2d(in_channels * k, 1, 1)
        self.return_intermediate = return_intermediate

    def forward(self, x):
        y = self.conv1(x)
        #y = F.sigmoid(x)
        x = self.conv2(y)
        if self.return_intermediate:
            return F.sigmoid(x), y
        return F.sigmoid(x), None


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
            layer.append(dog_class(filter_size, in_channels * k, learn_amplitude=learn_amplitude))
            layer.append(nn.Conv2d(in_channels * k, in_channels*k/2, 1))
            layer.append(nn.ReLU())

        self.net = nn.Sequential(*layer[:-2])
        self.final_convolution = nn.Conv2d(in_channels*k, 1, 1)

    def forward(self, x):
        x = self.net(x)
        x = self.final_convolution(x)
        return F.sigmoid(x), None


class DeepIsotropic(DeepNetwork):
    def __init__(self, in_channels, filter_size=9, k=4, n_layers=2, learn_amplitude=False):
        super(DeepIsotropic, self).__init__(in_channels, filter_size, k, n_layers, learn_amplitude, DoG2DIsotropic)


class DeepAnisotropic(DeepNetwork):
    def __init__(self, in_channels, filter_size=9, k=4, n_layers=2, learn_amplitude=False):
        super(DeepAnisotropic, self).__init__(in_channels, filter_size, k, n_layers, learn_amplitude, DoG2DAnisotropic)


