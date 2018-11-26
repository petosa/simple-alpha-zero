import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.append("..")
from model import Model


cfg = {
    'VGG2': [64, 'pool', 128, 'pool'],
    'VGG4': [64, 'pool', 128, 'pool', 256, 'pool'],
    'VGG6': [64, 'pool', 128, 'pool', 256, 'pool', 512, 'pool'],
    'VGG8': [64, 'pool', 128, 'pool', 256, 'pool', 512, 'pool', 512, 'pool'],
    'VGG11': [64, 'pool', 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
    'VGG13': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
    'VGG16': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool'],
    'VGG19': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 256, 'pool', 512, 512, 512, 512, 'pool', 512, 512, 512, 512, 'pool'],
}

class BigVGG(Model):
    
    def __init__(self, input_shape, output_shape):
        super(BigVGG, self).__init__(input_shape, output_shape)

        self.input_shape = input_shape
        self.output_shape = output_shape
        num_hidden_units = 128
        
        self.relu = nn.ReLU(inplace=True)
        self.vgg = self.VGG(cfg['VGG4'])
        self.p_head = torch.nn.Linear(num_hidden_units, np.prod(output_shape))
        self.v_head = torch.nn.Linear(num_hidden_units, 1)

    def forward(self, x):
        batch_size = len(x)
        this_output_shape = tuple([batch_size] + list(self.output_shape))
        x = x.permute(0,3,1,2) # NHWC -> NCHW

        print(x.shape)
        
        outx = self.vgg(x)
        flat = outx.view(batch_size, -1)
        
        p_logits = self.p_head(flat).view(this_output_shape)
        v = torch.tanh(self.v_head(flat))
        
        return p_logits, v

    def VGG(self, layarr):
        layers = []
        prevConvLayerSize = 3
        for lname in layarr:
            if lname == 'pool':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(prevConvLayerSize, lname, kernel_size=3, padding=1)
                batch = nn.BatchNorm2d(lname)
                layers += [conv, batch, self.relu]
                prevConvLayerSize = lname
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        vgg = nn.Sequential(*layers)
        return vgg

