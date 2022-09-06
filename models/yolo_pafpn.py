#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from .darknet import CSPDarknet, Darknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .multiattehead import CFADdecoder

def kaiming_init(module,a=0,mode = 'fan_out',nonlinearity = 'relu',bisa =0,distribution = 'normal'):
    assert distribution in ['uniform','normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight,a=a,mode=mode,nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_uniform_(module.weight,a=a,mode=mode,nonlinearity=nonlinearity)
    if hasattr(module,'bias') and module.bias is not None:
        nn.init.constant_(module.bias,bisa)
class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[128,256, 512],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
       
        #self.backbone = Darknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = [128,256, 512,1024]
        Conv = DWConv if depthwise else BaseConv



        self.lateral_conv0 = nn.Conv2d(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, 
        )
        self.BN1 = nn.GroupNorm(8,int(in_channels[2] * width))

        self.lateral_conv1 = nn.Conv2d(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, 
        )
        self.BN2 = nn.GroupNorm(8,int(in_channels[1] * width))

        self.lateral_conv2 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, 
        )
        self.BN3 = nn.GroupNorm(8,int(in_channels[0] * width))


        self.Afpn1 = CFADdecoder(self.in_channels[1],2,0.1)
        self.Apan2 = CFADdecoder(self.in_channels[2],2,0.1)
        self.Apan3 = CFADdecoder(self.in_channels[3],2,0.1)
    
     
        self.kaiming_reset()
    def kaiming_reset(self):
        kaiming_init(self.lateral_conv0,mode = 'fan_in')
        kaiming_init(self.lateral_conv1,mode = 'fan_in')
        kaiming_init(self.lateral_conv2,mode = 'fan_in')
        kaiming_init(self.promote_conv1,mode = 'fan_in')
        kaiming_init(self.promote_conv2,mode = 'fan_in')

        self.lateral_conv0.inited = True
        self.lateral_conv1.inited = True
        self.lateral_conv2.inited = True
        self.promote_conv1.inited = True
        self.promote_conv2.inited = True

    def forward(self, input):

        [x3, x2, x1, x0] = input
        bs= x0.size()[0]
        afpn_out0 = x0
        f1 = self.lateral_conv0(x0)
        f1 = self.BN1(f1).view(-1,bs,self.in_channels[2])
        x11 = x1.view(-1,bs,self.in_channels[2])
        afpn_out1 = self.Afpn1(x11,f1)
        afpn_out1 = afpn_out1.view(x1.size())

        f2 = self.lateral_conv1(afpn_out1)
        f2 = self.BN2(f2).view(-1,bs,self.in_channels[1])
        x22 = x2.view(-1,bs,self.in_channels[1])
        afpn_out2 = self.Afpn2(x22,f2)
        afpn_out2 = afpn_out2.view(x2.size())

        f3 = self.lateral_conv1(afpn_out2)
        f3 = self.BN3(f3).view(-1,bs,self.in_channels[0])
        x33 = x3.view(-1,bs,self.in_channels[0])
        afpn_out3 = self.Afpn3(x33,f3)
        afpn_out3 = afpn_out2.view(x3.size())


        outputs = (afpn_out3,afpn_out2, afpn_out1, afpn_out0)
        return outputs
