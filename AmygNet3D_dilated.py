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
          self.dropout = nn.Dropout3d(0)

    def forward(self, input):
       out = self.bn(self.conv(input))

       if self.do_act:
          out = self.act(out)
       if self.if_drop:
          out = self.dropout(out)

       return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, kernel_size, dilation_list, do_act=True, if_drop=False, test_state=False):
       super(ResBlock, self).__init__()

       self.out_channels_2 = out_channels_2
       if dilation_list is None:
          self.conv_relu = ConvBlock(in_channels, out_channels_1, kernel_size, 1, True) # Do activation
          if out_channels_2 != 0:
             self.conv = ConvBlock(out_channels_1, out_channels_2, kernel_size, 1, False) # No activation
             self.conv1x1 = ConvBlock(in_channels, out_channels_2, 1, 1, False) # No activation
       else:
          self.conv_relu = ConvBlock(in_channels, out_channels_1, kernel_size, dilation_list[0], True)
          self.conv = ConvBlock(out_channels_1, out_channels_2, kernel_size, dilation_list[1], False)
          self.conv1x1 = ConvBlock(in_channels, out_channels_2, 1, 1, False) # No activation

       self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
       identity = x
       out = self.conv_relu(x)
       if self.out_channels_2 != 0:
          out = self.conv(out)
          _,_,a,b,c = out.size()
          identity = Crop(identity, [a,b,c])
          identity = self.conv1x1(identity)
          out += identity
          out = self.relu(out)
       return out


class AmygNet3D(nn.Module):

    def __init__(self, num_classes, wrs_ratio, drop_rate, wrs_ratio_fc, drop_rate_fc, test_state=False):
        super(AmygNet3D,self).__init__()

        self.test_state = test_state

        self.firstConv = ConvBlock(1,30,3,1)

        #dilated path
        self.dilated_conv1 = ResBlock(30,30,40,3,[2,4])
        self.dilated_conv2 = ResBlock(40,40,40,3,[2,8])
        self.dilated_conv3 = ResBlock(40,40,50,3,[2,4])
        self.dilated_conv4 = ResBlock(50,50,50,3,[2,1])

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

        #dilated_path
        x_dilated = self.firstConv(x)
        x_dilated = self.dilated_conv1(x_dilated)
        x_dilated = self.dilated_conv2(x_dilated)
        x_dilated = self.dilated_conv3(x_dilated)
        x_dilated = self.dilated_conv4(x_dilated)

        if args.triple:
           att = AdaDropout(100, self.wrs_ratio, self.test_state)
           x_FC_1 = self.FC_1(att(x_dilated))
        else:
           x_FC_1 = self.FC_1(x_dilated)
           x_FC_2 = self.FC_2(x_FC_1)

        output = self.classification(x_FC_2)
        return output

     
