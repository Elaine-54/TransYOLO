#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .multiattehead import CFADdecoder,EFAdecoder
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
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = [128,256, 512]
        Conv = DWConv if depthwise else BaseConv

        # self.lateral_conv0 = BaseConv(
        #     int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        # )
        # self.lateral_conv1 = BaseConv(
        #     int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        # )
        # self.promote_conv1 = BaseConv(
        #     int(in_channels[0] * width), int(in_channels[1] * width), 1, 1, act=act
        # )
        # self.promote_conv2 = BaseConv(
        #     int(in_channels[1] * width), int(in_channels[2] * width), 1, 1, act=act
        # )
        self.lateral_conv0 = nn.Conv2d(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, 
        )
        self.BN1 = nn.GroupNorm(8,int(in_channels[1] * width))
        self.lateral_conv1 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, 
        )
        self.BN2 = nn.GroupNorm(8,int(in_channels[0] * width))
        self.promote_conv1 = nn.Conv2d(
            int(in_channels[0] * width), int(in_channels[1] * width), 1, 1, 
        )
        self.BN3 = nn.GroupNorm(8,int(in_channels[1] * width))
        self.promote_conv2 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[2] * width), 1, 1, 
        )
        self.BN4 = nn.GroupNorm(8,int(in_channels[2] * width))


        self.E10 = EFAdecoder(self.in_channels[2],4,0.1)
        self.E20 = EFAdecoder(self.in_channels[1],4,0.1)
        self.E30 = EFAdecoder(self.in_channels[0],4,0.1)
        self.E4_256_x = EFAdecoder(self.in_channels[1],4,0.1)
        self.E5_128_s = EFAdecoder(self.in_channels[0],4,0.1)
        self.E5_256_s = EFAdecoder(self.in_channels[1],4,0.1)



        self.Afpn1 = CFADdecoder(self.in_channels[1],1,0.1)
        self.Afpn2 = CFADdecoder(self.in_channels[0],1,0.1)
        self.Apan1 = CFADdecoder(self.in_channels[1],1,0.1)
        self.Apan2 = CFADdecoder(self.in_channels[2],1,0.1)

        self.kaiming_reset()
    def kaiming_reset(self):
        kaiming_init(self.lateral_conv0,mode = 'fan_in')
        kaiming_init(self.lateral_conv1,mode = 'fan_in')
        kaiming_init(self.promote_conv1,mode = 'fan_in')
        kaiming_init(self.promote_conv2,mode = 'fan_in')

        self.lateral_conv0.inited = True
        self.lateral_conv1.inited = True
        self.promote_conv1.inited = True
        self.promote_conv2.inited = True


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        
        afpn_out0 = self.E10(x0)
        x1= self.E20(x1)
        x2= self.E30(x2)
    

        f1 = self.lateral_conv0(x0)
        f1 = self.BN1(f1)
        afpn_out1 = self.Afpn1(x1,f1)
      
        f2 = afpn_out1
        #print(afpn_out1.size())
        f2_p = self.E4_256_x(f2)

        f2_x = self.lateral_conv1(f2)

        f2_x = self.BN2(f2_x)
       
        afpn_out2 = self.Afpn2(x2,f2_x)

        apan_out2 = afpn_out2
        p1 = self.E5_128_s(afpn_out2)
        p1 = self.promote_conv1(apan_out2)
        p1 = self.BN3(p1)

        apan_out1 = self.Apan1(f2_p,p1)
        p2 = self.E5_256_s(afpn_out1)
        p2 = self.promote_conv2(apan_out1)
        p2 = self.BN4(p2)
        apan_out0 = self.Apan2(afpn_out0,p2)

        outputs = (apan_out2, apan_out1, apan_out0)
        return outputs
