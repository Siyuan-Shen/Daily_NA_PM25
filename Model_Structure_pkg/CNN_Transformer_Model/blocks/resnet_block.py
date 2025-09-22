import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv3d, BatchNorm3d, ReLU, MaxPool3d, AvgPool3d, Dropout3d
from Model_Structure_pkg.utils import *
from Training_pkg.utils import activation_function_table,define_activation_func
from Model_Structure_pkg.CNN_Module import BasicBlock, Bottleneck
from Training_pkg.utils import *

activation = activation_function_table()

def resnet_block_lookup_table(blocktype):
    if blocktype == 'BasicBlock':
        return BasicBlock
    elif blocktype == 'Bottleneck':
        return Bottleneck
    else:
        print(' Wrong Key Word! BasicBlock or Bottleneck only! ')
        return None

class resnet_block_embedding(nn.Module):

    def __init__(self, nchannel, # initial input channels
                    block, # block type, 'BasicBlock' or 'Bottleneck'
                    blocks_num, # number of blocks in each layer [n1, n2, n3, n4]   
                    output_channels, # output channels for each layer [c1, c2, c3, c4]
                    num_classes=1, # number of output classes
                    include_top=True, # whether to include the top layer
                 ):
        super(resnet_block_embedding, self).__init__()
        self.in_channel = output_channels[0] ## output channel for the first layer ,and inchannel for other blocks
        self.include_top = include_top

        self.actfunc = define_activation_func(activation)
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=3, stride=1,padding=1, bias=False,padding_mode=CNN_Transformer_CovLayer_padding_mode)
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ) # output 4x4
        self.apply_pooling_layer = True
        if CNN_Transformer_ResNet_Pooling_layer_type == 'MaxPooling2d':
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=2) # output 4x4
        elif CNN_Transformer_ResNet_Pooling_layer_type == 'AvgPooling2d':
            self.pooling = nn.AvgPool2d(kernel_size=3, stride=2) # output 4x4
        else:
            print('Pooling layer type not supported! Please use MaxPooling2d or AvgPooling2d.')
            self.apply_pooling_layer = False
        self.layer1 = self._make_layer(block, blocks_num[0], output_channels[0],stride=1)
        self.layer2 = self._make_layer(block, blocks_num[1], output_channels[1], stride=1)
        self.layer3 = self._make_layer(block, blocks_num[2], output_channels[2], stride=1)
        self.layer4 = self._make_layer(block, blocks_num[3], output_channels[3],stride=1)
        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(self.in_channel * block.expansion, num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, block_num, channel,  stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                activation=activation,
                               ))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    stride=1,
                                    activation=activation,
                                    ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        length = x.size(1)
        for i in range(length):
            temp_x = x[:,i,:,:,:]  # B, C, H, W
            temp_x = self.layer0(temp_x)
            if self.apply_pooling_layer:
                temp_x = F.pad(temp_x,pad=(1,1,1,1),mode=CNN_Transformer_Pooling_padding_mode,value=0)
                temp_x = self.pooling(temp_x)  # output 4x4
            temp_x = self.layer1(temp_x)
            temp_x = self.layer2(temp_x)
            temp_x = self.layer3(temp_x)
            temp_x = self.layer4(temp_x)

            if self.include_top:  
                temp_x = self.avgpool(temp_x)
                temp_x = torch.flatten(temp_x, 1)
                #temp_x = self.actfunc(temp_x)
            if i == 0:
                output_x = temp_x.unsqueeze(1)  # B, 1, C
            else:
                output_x = torch.cat((output_x, temp_x.unsqueeze(1)), dim=1)  # B, T, C 

        return output_x