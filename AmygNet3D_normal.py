import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from Dataset import Crop
import torch.optim as optim
import math
from AdaDropout_dynamic import AdaDropout

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, do_act=True, if_drop=False, test_state=False):
       super(ConvBlock, self).__init__()

       self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

       self.bn = nn.BatchNorm3d(out_channels)
       self.do_act = do_act
       self.if_drop = if_drop
       if self.do_act:
         self.act = nn.ReLU()

       if self.if_drop:
          self.dropout = nn.Dropout3d(0)  #AdaDropout(out_channels, test_state)

    def forward(self, input):
       out = self.bn(self.conv(input))

       if self.do_act:
          out = self.act(out)
       if self.if_drop:
          out = self.dropout(out)

       return out

class HybridConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, do_act=True):
       super(HybridConvBlock, self).__init__()

       self.dilated_conv = ConvBlock(in_channels, out_channels, 3, dilation_rate, do_act)
       self.normal_conv = ConvBlock(in_channels, out_channels, 3, 1, do_act)

    def forward(self, input):
       x_dilated = self.dilated_conv(input)
       x_normal = self.normal_conv(input)
       _,_,a,b,c = x_dilated.size()
       x_normal = Crop(x_normal,[a,b,c])
       return x_normal


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, dilation_list, do_act=True, if_drop=False, test_state=False):
       super(ResBlock, self).__init__()

       self.conv_relu = HybridConvBlock(in_channels, out_channels_1, dilation_list[0], True) # Do activation
       self.conv = HybridConvBlock(out_channels_1, out_channels_2, dilation_list[1], False) # No activation
       self.reshape = ConvBlock(in_channels, out_channels_2, 1, 1, False) # No activation

       self.relu = nn.ReLU()

    def forward(self, x):
       identity = x
       out = self.conv_relu(x)
       out = self.conv(out)
       _,_,a,b,c = out.size()
       identity = Crop(identity,[a,b,c])
       identity = self.reshape(identity)
       out += identity
       out = self.relu(out)
       return out


class AmygNet3D(nn.Module):

    def __init__(self, num_classes, wrs_ratio, drop_rate, wrs_ratio_fc, drop_rate_fc, test_state=False):
        super(AmygNet3D,self).__init__()

        self.test_state = test_state

        self.firstConv = HybridConvBlock(1,30,1) # rate=2, do_act=True

        #dilated path
        self.g1 = ResBlock(30,30,40,[2,4])
        self.g2 = ResBlock(40,40,40,[2,8])
        self.g3 = ResBlock(40,40,50,[2,4])
        self.g4 = ResBlock(50,50,50,[2,1])

        #FC layers
        self.FC_1 = ConvBlock(50,150,1,1,True,True,self.test_state)
        self.FC_2 = ConvBlock(150,150,1,1,True,True,self.test_state)

        #Classification layer
        self.classification = ConvBlock(150,num_classes,1,1,False, False) # No activation, AdaDropout

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
               nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm3d):
               nn.init.constant_(m.bias.data, 0.0)
               nn.init.normal_(m.weight.data, 1.0, 0.02)

    def forward(self,x,args):
        out = self.firstConv(x)
        g1_out = self.g1(out)
        g2_out = self.g2(g1_out)
        g3_out = self.g3(g2_out)
        g4_out = self.g4(g3_out)


        if args.triple:
           att = AdaDropout(100, self.wrs_ratio, self.test_state)
           out = self.FC_1(att(out))
        else:
           out = self.FC_1(g4_out)
           out = self.FC_2(out)

        out = self.classification(out)
        return out
