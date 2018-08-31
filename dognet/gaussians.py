from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class Gaussian2DBase(nn.Module):
    def __init__(self, w, n_gaussian):
        super(Gaussian2DBase, self).__init__()
        assert 1 == w % 2, "'w' must be even 3,5,7,9,11 etc."
        self.xes = torch.FloatTensor(range(int(-w / 2),  int(w / 2) + 1)).unsqueeze(-1)
        self.xes = self.xes.repeat(self.xes.size(0), 1, n_gaussian)
        self.yes = self.xes.transpose(1, 0)

        self.xypod = Parameter(self.xes * self.yes, requires_grad=False)
        self.xes = Parameter(self.xes ** 2, requires_grad=False)
        self.yes = Parameter(self.yes ** 2, requires_grad=False)
        self.padding = int(w / 2)


class Gaussian3DBase(nn.Module):
    def __init__(self, w, n_gaussian, depth):
        super(Gaussian3DBase, self).__init__()
        assert 1 == w % 2, "'w' must be even 3,5,7,9,11 etc."
        assert 1 == depth % 2, "'depth' must be even 3,5,7,9,11 etc."

        self.xes = torch.FloatTensor(range(int(-w / 2),  int(w / 2) + 1)).unsqueeze(-1) ** 2
        self.xes = self.xes.repeat(depth, self.xes.size(0), 1, n_gaussian)
        self.yes = self.xes.transpose(1, 2)
        self.zes = torch.FloatTensor(range(int(-depth / 2),  int(depth / 2) + 1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1) ** 2

        self.xes = Parameter(self.xes, requires_grad=False)
        self.yes = Parameter(self.yes, requires_grad=False)
        self.zes = Parameter(self.zes.repeat(1, self.xes.size(1), self.xes.size(2), n_gaussian), requires_grad=False)


class Gaussian2DIsotropic(Gaussian2DBase):
    """
    Isotropic (circle shape) Gaussian filter 
    """

    def __init__(self, w, in_channels, n_gaussian=1, learn_amplitude=False):
        """
        Creates an isotropic Gaussian filter
        :param w: convolutional window size must be even 3,5,7,9,11 etc.
        :param n_gaussian: number of Gaussians produced
        :param learn_amplitude: is True - amplitude comes as a Parameter, otherwise the filter is normalized be 1/sigma
        """
        super(Gaussian2DIsotropic, self).__init__(w, in_channels*n_gaussian)
        self.s = Parameter(torch.randn(in_channels*n_gaussian).float(), requires_grad=True)
        self.in_channels = in_channels
        self.n_gaussian = n_gaussian
        self.amplitude = None
        if learn_amplitude:
            self.amplitude = Parameter(torch.randn(n_gaussian).float(), requires_grad=True)
    
    def weights_init(self,s):
        if self.n_gaussian>1:
            self.s.data = torch.linspace(0.5,1.5,self.n_gaussian).repeat(self.in_channels).to(self.s.device)+s
        else: 
            self.s.data.fill_(1.)
            
        if self.amplitude is not None:
            self.amplitude.data.fill_(1.)

    def get_filter(self, s=None, amplitude=None):
        """
        
        :param s: 
        :param amplitude: 
        :return: 
        """
        if s is None:
            s = self.s

        if amplitude is None:
            amplitude = self.amplitude

        s = s.abs()
        s = s.expand(self.xes.size(0), self.xes.size(1), s.size(0))
        if amplitude is None:
            amplitude = s
        else:
            amplitude = amplitude.expand(self.xes.size(0), self.xes.size(1), amplitude.size(0))

        filters = amplitude * (- ((self.xes + self.yes)* s ** 2)/2).exp()
        #filters = amplitude * (-(self.xes + self.yes)/(2* s ** 2)).exp()
        return filters.transpose(0, 2).unsqueeze(1).contiguous()

    def forward(self, x):
        filters = self.get_filter(self.s, self.amplitude)
        return F.conv2d(x, filters, padding=self.padding, groups=x.size(1))
        #r = F.conv2d(x, torch.sum(filters, dim=2,keepdim=True).contiguous(), padding=(0,self.padding),groups=x.size(1))/2.
        #r = F.conv2d(r,torch.sum(filters, dim=3,keepdim=True).contiguous(), padding=(self.padding,0), groups=x.size(1))/2.
        #return r

    def __check__(self, var):
        if var is not None:
            return str(self.s.data.cpu().numpy()) + ', '
        return ""

    def __repr__(self):

        return self.__class__.__name__ + ' (' \
               + self.__check__(self.s.data.cpu().numpy()) \
               + self.__check__(self.amplitude) + ")"


class Gaussian2DAnisotropic(Gaussian2DBase):
    """
    Gaussian for anisotropic case. Learns for each filter sx,sy (spread x,y), th (rotation angle), A (amplitude). 
    """

    def __init__(self, w, in_channels, n_gaussian, th=None, learn_amplitude=True):
        super(Gaussian2DAnisotropic, self).__init__(w, n_gaussian*in_channels)
        self.n_gaussian = n_gaussian
        self.in_channels = in_channels
        
        self.sx = Parameter(torch.randn(n_gaussian*in_channels).float(), requires_grad=True)
        self.sy = Parameter(torch.randn(n_gaussian*in_channels).float(), requires_grad=True)
        self.th = th

        if self.th is None:
            self.th = Parameter(torch.randn(n_gaussian*in_channels).float(), requires_grad=True)

        if learn_amplitude:
            self.amplitude = Parameter(torch.randn(n_gaussian*in_channels).float(), requires_grad=True)
        else:
            self.amplitude = None
            
    def weights_init(self,x,y):
        if self.n_gaussian>1:
            self.sx.data = torch.linspace(0.5,1.5,self.n_gaussian).repeat(self.in_channels).to(self.sx.device)+x
            self.sy.data = torch.linspace(0.5,1.5,self.n_gaussian).repeat(self.in_channels).to(self.sy.device)+y
        else: 
            self.sx.data.fill_(1.)
            self.sy.data.fill_(1.)
            
        if self.amplitude is not None:
            self.amplitude.data.fill_(1.)

    def get_filter(self, sx=None, sy=None, theta=None, amplitude=None):

        if theta is None:
            if self.th is not None:
                theta = self.th

        if sy is None:
            sy = self.sy

        if sx is None:
            sx = self.sx

        if amplitude is None:
            amplitude = self.amplitude

        sx = sx.abs()
        sy = sy.abs()

        a = theta.cos() ** 2* sx ** 2 / (2 ) + theta.sin() ** 2 * sy ** 2/ (2 )
        b = -(2 * theta).sin()* sx ** 2 / (4 ) + (2 * theta).sin()* sy ** 2 / (4 )
        c = theta.sin() ** 2 * sx ** 2 / (2) + theta.cos() ** 2* sy ** 2 / (2 )

        a = a.expand(self.xes.size(0), self.xes.size(1), a.size(0))
        b = b.expand(self.xes.size(0), self.xes.size(1), b.size(0))
        c = c.expand(self.xes.size(0), self.xes.size(1), c.size(0))

        if amplitude is None:
            amplitude = 2.*(sx + sy)

        filters =  amplitude*(- (a * self.xes + 2 * b * self.xypod + c * self.yes)).exp()
        return filters.transpose(0, 2).unsqueeze(1).contiguous()

    def forward(self, x):
        filters = self.get_filter(self.sx, self.sy, self.th, self.amplitude)
        return F.conv2d(x, filters, padding=self.padding, groups=x.size(1))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.sx) + ', ' \
               + str(self.sy) + ', ' \
               + str(self.th) + ', ' \
               + str(self.amplitude) + ')'


class Gaussian3DIsotropic(Gaussian3DBase):
    """
    Three dimensional Gaussian for volumes processing
    """

    def __init__(self, w, n_gaussian, depth, learn_amplitude=False):
        """
        Creates an isotropic Gaussian filter
        :param w: convolutional window size must be even 3,5,7,9,11 etc.
        :param n_gaussian: number of Gaussians produced
        :param learn_amplitude: is True - amplitude comes as a Parameter, otherwise the filter is normalized be 1/sigma
        """
        super(Gaussian3DIsotropic, self).__init__(w, n_gaussian, depth)
        self.s = Parameter(torch.randn(n_gaussian).float(), requires_grad=True)
        self.sz = Parameter(torch.randn(n_gaussian).float(), requires_grad=True)

        self.amplitude = None
        if learn_amplitude:
            self.amplitude = Parameter(torch.randn(n_gaussian).float(), requires_grad=True)
            
    def weights_init(self,s,z):
        self.s.data.fill_(s)
        self.sz.data.fill_(z)
        if self.amplitude is not None:
            self.amplitude.data.fill_(1.)

    def get_filter(self, s=None, sz=None, amplitude=None):
        """
        Get 3D Gaussian
        :param s: sigma for isotropic gaussian
        :param sz:
        :param amplitude: 
        :return: 
        """
        eps = 1e-8
        if s is None:
            s = self.s

        if sz is None:
            sz = self.sz

        if amplitude is None:
            amplitude = self.amplitude

        s = s.abs() + eps
        sz = sz.abs() + eps

        s = s.expand(self.xes.size(0), self.xes.size(1), self.xes.size(2), s.size(0))
        sz = sz.expand(self.xes.size(0), self.xes.size(1), self.xes.size(2), sz.size(0))

        if amplitude is not None:
            amplitude = amplitude.expand(self.xes.size(0), self.xes.size(1), self.xes.size(2), amplitude.size(0))
        else:
            ssz = (s + sz)
            amplitude = 2. / ssz

        filters = amplitude * (- (self.xes + self.yes) / (2 * s ** 2) - self.zes / (2 * sz ** 2)).exp()
        return filters.transpose(0, 3).unsqueeze(1).contiguous()

    def forward(self, x):
        filters = self.get_filter(self.s, self.amplitude)
        return F.conv3d(x, filters, padding=self.padding, groups=x.size(1))
