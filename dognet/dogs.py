import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.parameter import Parameter

from .gaussians import Gaussian2DAnisotropic,Gaussian2DIsotropic,Gaussian3DIsotropic


class DoG2DIsotropic(nn.Module):
    def __init__(self, w, in_channels, n_gaussian, learn_amplitude=False):
        super(DoG2DIsotropic, self).__init__()
        
        self.A = Gaussian2DIsotropic(w,in_channels, n_gaussian, learn_amplitude)
        self.B = Gaussian2DIsotropic(w,in_channels, n_gaussian, learn_amplitude)
        
    def weights_init(self):
        self.A.weights_init(0.)
        self.B.weights_init(0.01)

    def forward(self, x):
        filters = self.A.get_filter() - self.B.get_filter()
        return F.conv2d(x, filters, padding=self.A.padding, groups=x.size(1))
        #r = F.conv2d(x,torch.sum(filters, dim=2,keepdim=True).contiguous(), padding=(0,self.A.padding), groups=x.size(1))/2.
        #r = F.conv2d(r,torch.sum(filters, dim=3,keepdim=True).contiguous(), padding=(self.A.padding,0), groups=x.size(1))/2.
        #return r
        #r = F.conv2d(x, filters[:,:,self.A.padding:self.A.padding+1,:].contiguous(), padding=(0,self.A.padding), groups=x.size(1))+F.conv2d(x, filters[:,:,:,self.A.padding:self.A.padding+1].contiguous(), padding=(self.A.padding,0), groups=x.size(1))
        #return r
        #


class DoG2DAnisotropic(nn.Module):
    def __init__(self, w, in_channels, n_gaussian, learn_amplitude=True):
        super(DoG2DAnisotropic, self).__init__()
        self.theta = Parameter(torch.randn(n_gaussian*in_channels).float(), requires_grad=True)
        self.A = Gaussian2DAnisotropic(w, in_channels, n_gaussian, th=self.theta,learn_amplitude=learn_amplitude)
        self.B = Gaussian2DAnisotropic(w, in_channels, n_gaussian, th=self.theta,learn_amplitude=learn_amplitude)

    def weights_init(self):
        self.A.weights_init(0.,0.)
        self.B.weights_init(0.01,0.01)
        
    def forward(self, x):
        filters = self.A.get_filter() - self.B.get_filter()
        return F.conv2d(x, filters, padding=self.A.padding, groups=x.size(1))


class DoG3DIsotropic(nn.Module):
    def __init__(self, w,in_channels, n_gaussian, depth, learn_amplitude=False):
        super(DoG3DIsotropic, self).__init__()
        self.w = w
        self.A = Gaussian3DIsotropic(w, n_gaussian*in_channels,depth, learn_amplitude)
        self.B = Gaussian3DIsotropic(w, n_gaussian*in_channels,depth, learn_amplitude)

    def weights_init(self):
        self.A.weights_init(2.,1.)
        self.B.weights_init(1.,1.)
        
    def forward(self, x):
        filters = self.A.get_filter() - self.B.get_filter()
        filters = filters.transpose(2,4).transpose(3,4).contiguous()
        return F.conv3d(x, filters, padding=(0,int(self.w/2),int(self.w/2)), groups=x.size(1))

class Lap2D(nn.Module):
    """
    Isotropic (Lap2D shape) Lap2D filter 
    """
    def __init__(self, w, n_gaussian,learn_amplitude=False):
        super(Lap2D, self).__init__()
        self.xes = torch.FloatTensor(range(int(-w / 2)+1,  int(w / 2) + 1)).unsqueeze(-1)
        self.xes = self.xes.repeat(self.xes.size(0), 1, n_gaussian)
        self.yes = self.xes.transpose(1, 0)

        self.xypod = Parameter(self.xes * self.yes, requires_grad=False)
        self.xes = Parameter(self.xes.pow(2), requires_grad=False)
        self.yes = Parameter(self.yes.pow(2), requires_grad=False)
        self.padding = int(w / 2)
        self.s = Parameter(torch.randn(n_gaussian).float(), requires_grad=True)
        print("Current Lap2D")
        
    def weights_init(self):
        self.s.data.normal_(1.,0.3)

    def get_gaussian(self,s):
        #return  (- (self.xes + self.yes) / (2 * s.pow(2))).exp()/(2.4569*s)
        return  (- (self.xes + self.yes)*s.pow(2) / 2).exp()/(2.4569)*s
        
    def get_filter(self, s=None):
        """     
        :param s: 
        :param amplitude: 
        :return: 
        """
        if s is None:
            s = self.s
            
        eps=1e-3
        k=(self.s.pow(2)/eps) 
        filters = self.get_gaussian(self.s)\
                          - self.get_gaussian(self.s+eps) 
        
        return (k*filters).transpose(0, 2).unsqueeze(1).contiguous()

    def forward(self, x):
        filters = self.get_filter(self.s)
        return F.conv2d(x, filters, padding=self.padding, groups=x.size(1))  
"""
class Laplasian(nn.Module):
    def __init__(self, w, n_gaussian, padding=None, delta=0.1):
        super(Laplasian, self).__init__()
        self.s = Parameter(torch.randn(n_gaussian).float().cuda(), requires_grad=True)
        self.xes = torch.FloatTensor(range(-w / 2 + 1, w / 2 + 1))
        self.xes = Variable(self.xes.repeat(self.xes.size(0), 1), requires_grad=False).unsqueeze(-1).cuda()
        self.xes = self.xes.repeat(1, 1, n_gaussian)
        self.yes = self.xes.transpose(1, 0)
        self.sqr = -(self.xes ** 2 + self.yes ** 2)
        self.delta = delta
        self.padding = w / 2

    def forward(self, x):
        s = 2 * self.s.abs() ** 2
        s = s.expand(self.xes.size(0), self.xes.size(1), s.size(0))

        fiters = ((self.sqr / s).exp() - (self.sqr / (s + self.delta)).exp()).transpose(0, 2).unsqueeze(1).contiguous()
        # print(x.size(),fiters.size(),s.size())
        return F.conv2d(x, fiters, padding=self.padding, groups=x.size(1))
"""
