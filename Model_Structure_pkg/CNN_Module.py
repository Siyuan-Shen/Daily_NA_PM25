import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model_Structure_pkg.utils import *
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


def initial_cnn_network(width,main_stream_nchannel,side_stream_nchannel):

    if TwoCombineModels_Settings:
        Model_A = initial_cnn_OneStage_network(width=width,main_stream_nchannel=main_stream_nchannel,side_stream_nchannel=side_stream_nchannel)
        Model_B = initial_cnn_OneStage_network(width=width,main_stream_nchannel=main_stream_nchannel,side_stream_nchannel=side_stream_nchannel)
        cnn_model = Combine_GeophysicalDivide_Two_Models(model_A=Model_A, model_B=Model_B)
    else:
        cnn_model = initial_cnn_OneStage_network(width=width,main_stream_nchannel=main_stream_nchannel,side_stream_nchannel=side_stream_nchannel)
    return cnn_model
 
def initial_cnn_OneStage_network(width,main_stream_nchannel,side_stream_nchannel):

    if ResNet_Settings:
        block = resnet_block_lookup_table(ResNet_Blocks)
        cnn_model = ResNet(nchannel=main_stream_nchannel,block=block,blocks_num=ResNet_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)#cnn_model = Net(nchannel=nchannel)
    elif ResNet_MLP_Settings:
        block = resnet_block_lookup_table(ResNet_MLP_Blocks)
        cnn_model = ResNet_MLP(nchannel=main_stream_nchannel,block=block,blocks_num=ResNet_MLP_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)#cnn_model = Net(nchannel=nchannel)
    elif ResNet_Classification_Settings:
        block = resnet_block_lookup_table(ResNet_Classification_Blocks)
        cnn_model = ResNet_Classfication(nchannel=main_stream_nchannel,block=block,blocks_num=ResNet_Classification_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)
    elif ResNet_MultiHeadNet_Settings:
        block = resnet_block_lookup_table(ResNet_MultiHeadNet_Blocks)
        cnn_model = MultiHead_ResNet(nchannel=main_stream_nchannel,block=block,blocks_num=ResNet_MultiHeadNet_blocks_num,include_top=True,groups=1,width_per_group=width)
    elif LateFusion_Settings:
        block = resnet_block_lookup_table(LateFusion_Blocks)
        cnn_model = LateFusion_ResNet(nchannel=main_stream_nchannel,nchannel_lf=side_stream_nchannel,block=block,blocks_num=LateFusion_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)
    elif MultiHeadLateFusion_Settings:
        block = resnet_block_lookup_table(MultiHeadLateFusion_Blocks)
        cnn_model = MultiHead_LateFusion_ResNet(nchannel=main_stream_nchannel,nchannel_lf=side_stream_nchannel,block=block,blocks_num=MultiHeadLateFusion_blocks_num,include_top=True,groups=1,width_per_group=width)
    return cnn_model

class BasicBlock(nn.Module):  
    
    expansion = 1  
    def __init__(self, in_channel, out_channel, stride=1,downsample=None,activation='tanh', **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False,padding_mode=CNN_CovLayer_padding_mode)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)  
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False, padding_mode=CNN_CovLayer_padding_mode)  # unsqueeze channels
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.actfunc = define_activation_func(activation)
        self.downsample = downsample
        
    def forward(self, x):
        
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actfunc(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity # out=F(X)+X
        out = self.actfunc(out)

        return out
    
class Bottleneck(nn.Module):  
    # Three convolutional layers, F(x) and X have different dimensions.
    """
    注意: 原论文中, 在虚线残差结构的主分支上, 第一个1x1卷积层的步距是2, 第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1, 第二个3x3卷积层步距是2,
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """
    
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, activation='tanh', width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        # 此处width=out_channel
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1, padding_mode=CNN_CovLayer_padding_mode)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False, padding_mode=CNN_CovLayer_padding_mode)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        global ReLU_ACF, Tanh_ACF, GeLU_ACF, Sigmoid_ACF
        self.actfunc = define_activation_func(activation)
        self.downsample = downsample

    def forward(self, x):
        in_size = x.size(0)
        identity = x
       
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actfunc(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.actfunc(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out=F(X)+X
        out = out + identity
        out = self.actfunc(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self,
                 nchannel, # initial input channel
                 block,  # block types
                 blocks_num,  
                 num_classes=1,  
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  

        self.groups = groups
        self.actfunc = define_activation_func(activation)
        self.width_per_group = width_per_group
        #self.conv1 = nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.in_channel)

        #self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer0 = nn.Sequential(self.conv1,self.bn1,self.tanh,self.maxpool)
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False,padding_mode=CNN_CovLayer_padding_mode) #output size:6x6
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=3, stride=1,padding=1, bias=False,padding_mode=CNN_CovLayer_padding_mode)
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ) # output 4x4

        self.apply_pooling_layer = True
        if Pooling_layer_type == 'MaxPooling2d':
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=2) # output 4x4
        elif Pooling_layer_type == 'AvgPooling2d':
            self.pooling = nn.AvgPool2d(kernel_size=3, stride=2) # output 4x4
        else:
            self.apply_pooling_layer = False
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(self.in_channel * block.expansion, num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

   
    def _make_layer(self, block, channel, block_num, stride=1):
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
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        if self.apply_pooling_layer:
            x = F.pad(x,pad=(1,1,1,1),mode=Pooling_padding_mode_3D,value=0)
            x = self.pooling(x)  # output 4x4
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            x = self.fc(x)

        return x

class Combine_GeophysicalDivide_Two_Models(nn.Module):
    def __init__(self,model_A,model_B,):
        super(Combine_GeophysicalDivide_Two_Models, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
    def forward(self,x_A,x_B):
        x_A = self.model_A(x_A)
        x_B = self.model_B(x_B)
        return x_A, x_B

class ResNet_MLP(nn.Module):
    
    def __init__(self,
                 nchannel, # initial input channel
                 block,  # block types
                 blocks_num,  
                 num_classes=1,  
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(ResNet_MLP, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  

        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        
        #self.conv1 = nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.in_channel)

        #self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer0 = nn.Sequential(self.conv1,self.bn1,self.tanh,self.maxpool)
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

        self.mlp_outlayer = nn.Sequential(nn.Linear(512 * block.expansion,512),
                                          self.actfunc,
                                          nn.BatchNorm1d(512),
                                          nn.Linear(512,128),
                                          self.actfunc,
                                          nn.Linear(128,num_classes))
   
    def _make_layer(self, block, channel, block_num, stride=1):
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
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            #x = self.fc(x)
            x = self.mlp_outlayer(x)

        return x
    

class ResNet_Classfication(nn.Module):

    def __init__(self,
                 nchannel, # initial input channel
                 block,  # block types
                 blocks_num,  
                 num_classes=1,  
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(ResNet_Classfication, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  

        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        self.left_bin    = ResNet_Classification_left_bin
        self.right_bin   = ResNet_Classification_right_bin
        self.bins_number = ResNet_Classification_bins_number
        #self.conv1 = nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.in_channel)

        #self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer0 = nn.Sequential(self.conv1,self.bn1,self.tanh,self.maxpool)
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.bins_fc = nn.Linear(512 * block.expansion, self.bins_number)
        self.softmax = nn.Softmax()
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

   
    def _make_layer(self, block, channel, block_num, stride=1):
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
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            classification_output = self.bins_fc(x)
            classification_output = self.softmax(classification_output)
        return classification_output


class MultiHead_ResNet(nn.Module):
    
    def __init__(self,
                 nchannel, # initial input channel
                 block,  # block types
                 blocks_num,   
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(MultiHead_ResNet, self).__init__()
        
        self.include_top = include_top
        self.in_channel = 64  
        self.in_channel_cls = 64  


        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        self.left_bin    = ResNet_MultiHeadNet_left_bin
        self.right_bin   = ResNet_MultiHeadNet_right_bin
        self.bins_number = ResNet_MultiHeadNet_bins_number
        self.bins        = torch.tensor(np.linspace(self.left_bin,self.right_bin,self.bins_number))

        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)


        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            self.fc = nn.Linear(512 * block.expansion, 1)
            self.bins_fc = nn.Linear(512 * block.expansion, self.bins_number)
        
        self.softmax = nn.Softmax()

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
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
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            regression_output = self.fc(x)
            classification_output = self.bins_fc(x)
            classification_output = self.softmax(classification_output)
        return regression_output, classification_output
    
class LateFusion_ResNet(nn.Module):

    def __init__(self,
                 nchannel, # initial input channel
                 nchannel_lf, # input channel for late fusion
                 block,  # block types
                 blocks_num,  
                 num_classes=1,  
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(LateFusion_ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  
        self.in_channel_lf = 16
        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        
       
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        self.layer0_lf = nn.Sequential(nn.Conv2d(nchannel_lf, self.in_channel_lf, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel_lf)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[3], stride=1)

        self.layer1_lf = self._make_layer_lf(block, 32, blocks_num[0])
        self.layer2_lf = self._make_layer_lf(block, 64, blocks_num[1], stride=1)
        self.layer3_lf = self._make_layer_lf(block, 64, blocks_num[2], stride=1)

        self.fuse_layer = self._make_layer_fused(block, 512, blocks_num[2], stride=1)
        
                
        
        

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
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
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def _make_layer_lf(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_lf != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_lf = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_lf,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _make_layer_fused(self, block, channel, block_num, stride=1):
        if stride != 1 or (self.in_channel_lf+self.in_channel) != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf+self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf+self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x,x_lf):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_lf = self.layer0_lf(x_lf)
        x_lf = self.layer1_lf(x_lf)
        x_lf = self.layer2_lf(x_lf)
        x_lf = self.layer3_lf(x_lf)
        
        
        x = torch.cat((x,x_lf),1)
        x = self.fuse_layer(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            x = self.fc(x)

        return x
    

        
'''
class MultiHead_LateFusion_ResNet(nn.Module):
    
    def __init__(self,
                 nchannel, # initial input channel
                 nchannel_lf, # input channel for late fusion
                 block,  # block types
                 blocks_num,   
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(MultiHead_LateFusion_ResNet, self).__init__()
        
        self.include_top = include_top
        self.in_channel = 64  
        self.in_channel_lf = 16
        self.groups = groups
        self.width_per_group = width_per_group
        self.actfunc = activation_func
        self.left_bin    = MultiHeadLateFusion_left_bin
        self.right_bin   = MultiHeadLateFusion_right_bin
        self.bins_number = MultiHeadLateFusion_bins_number
        self.bins        = torch.tensor(np.linspace(self.left_bin,self.right_bin,self.bins_number))

        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel)
        ,activation_func
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        self.layer0_lf = nn.Sequential(nn.Conv2d(nchannel_lf, self.in_channel_lf, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel_lf)
        ,activation_func
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[3], stride=1)

        self.layer1_lf = self._make_layer_lf(block, 32, blocks_num[0])
        self.layer2_lf = self._make_layer_lf(block, 64, blocks_num[1], stride=1)
        self.layer3_lf = self._make_layer_lf(block, 64, blocks_num[2], stride=1)

        self.fuse_layer = self._make_layer_fused(block, 512, blocks_num[2], stride=1)
        

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            self.fc = nn.Linear(512 * block.expansion, 1)
            self.bins_fc = nn.Linear(512 * block.expansion, self.bins_number)
        
        self.softmax = nn.Softmax()

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def _make_layer_lf(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_lf != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel_lf,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel_lf = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel_lf,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _make_layer_fused(self, block, channel, block_num, stride=1):
        if stride != 1 or (self.in_channel_lf+self.in_channel) != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf+self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel_lf+self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x,x_lf):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_lf = self.layer0_lf(x_lf)
        x_lf = self.layer1_lf(x_lf)
        x_lf = self.layer2_lf(x_lf)
        x_lf = self.layer3_lf(x_lf)
        
        
        x = torch.cat((x,x_lf),1)
        x = self.fuse_layer(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            regression_output = self.fc(x)
            classification_output = self.bins_fc(x)
            classification_output = self.softmax(classification_output)
        #final_output = 0.5*regression_output + 0.5*torch.matmul(classification_output,self.bins)
        return regression_output, classification_output
'''      
class MultiHead_LateFusion_ResNet(nn.Module):
    
    def __init__(self,
                 nchannel, # initial input channel
                 nchannel_lf, # input channel for late fusion
                 block,  # block types
                 blocks_num,   
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(MultiHead_LateFusion_ResNet, self).__init__()
        
        self.include_top = include_top
        self.in_channel = 64  
        self.in_channel_lf = 16
        self.in_channel_clsfy = 64  
        self.in_channel_lf_clsfy = 16  

        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        self.left_bin    = MultiHeadLateFusion_left_bin
        self.right_bin   = MultiHeadLateFusion_right_bin
        self.bins_number = MultiHeadLateFusion_bins_number
        self.bins        = torch.tensor(np.linspace(self.left_bin,self.right_bin,self.bins_number))

        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        self.layer0_lf = nn.Sequential(nn.Conv2d(nchannel_lf, self.in_channel_lf, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel_lf)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[3], stride=1)

        self.layer1_lf = self._make_layer_lf(block, 32, blocks_num[0])
        self.layer2_lf = self._make_layer_lf(block, 64, blocks_num[1], stride=1)
        self.layer3_lf = self._make_layer_lf(block, 64, blocks_num[2], stride=1)

        self.fuse_layer = self._make_layer_fused(block, 512, blocks_num[2], stride=1)


        self.layer0_clsfy = nn.Sequential(nn.Conv2d(nchannel, self.in_channel_clsfy, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel_clsfy)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        self.layer0_lf_clsfy = nn.Sequential(nn.Conv2d(nchannel_lf, self.in_channel_lf_clsfy, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel_lf_clsfy)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1_clsfy = self._make_layer_clsfy(block, 64, blocks_num[0])
        self.layer2_clsfy = self._make_layer_clsfy(block, 128, blocks_num[1], stride=1)
        self.layer3_clsfy = self._make_layer_clsfy(block, 256, blocks_num[3], stride=1)

        self.layer1_lf_clsfy = self._make_layer_lf_clsfy(block, 32, blocks_num[0])
        self.layer2_lf_clsfy = self._make_layer_lf_clsfy(block, 64, blocks_num[1], stride=1)
        self.layer3_lf_clsfy = self._make_layer_lf_clsfy(block, 64, blocks_num[2], stride=1)

        self.fuse_layer_clsfy = self._make_layer_fused_clsfy(block, 512, blocks_num[2], stride=1)
        


        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            self.fc = nn.Linear(512 * block.expansion, 1)
            self.bins_fc = nn.Linear(512 * block.expansion, self.bins_number)
        
        self.softmax = nn.Softmax()

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
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
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def _make_layer_lf(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_lf != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_lf = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_lf,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _make_layer_fused(self, block, channel, block_num, stride=1):
        if stride != 1 or (self.in_channel_lf+self.in_channel) != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf+self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf+self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    def _make_layer_clsfy(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_clsfy != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_clsfy, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_clsfy,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_clsfy = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_clsfy,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def _make_layer_lf_clsfy(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_lf_clsfy != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf_clsfy, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf_clsfy,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_lf_clsfy = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_lf_clsfy,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _make_layer_fused_clsfy(self, block, channel, block_num, stride=1):
        if stride != 1 or (self.in_channel_lf_clsfy+self.in_channel_clsfy) != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf_clsfy+self.in_channel_clsfy, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf_clsfy+self.in_channel_clsfy,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_clsfy = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_clsfy,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x,x_lf):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x_r = self.layer0(x)
        x_r = self.layer1(x_r)
        x_r = self.layer2(x_r)
        x_r = self.layer3(x_r)

        x_lf_r = self.layer0_lf(x_lf)
        x_lf_r = self.layer1_lf(x_lf_r)
        x_lf_r = self.layer2_lf(x_lf_r)
        x_lf_r = self.layer3_lf(x_lf_r)
        
        
        x_r = torch.cat((x_r,x_lf_r),1)
        x_r = self.fuse_layer(x_r)

        #######################################
        x_c = self.layer0_clsfy(x)
        x_c = self.layer1_clsfy(x_c)
        x_c = self.layer2_clsfy(x_c)
        x_c = self.layer3_clsfy(x_c)

        x_lf_c = self.layer0_lf_clsfy(x_lf)
        x_lf_c = self.layer1_lf_clsfy(x_lf_c)
        x_lf_c = self.layer2_lf_clsfy(x_lf_c)
        x_lf_c = self.layer3_lf_clsfy(x_lf_c)

        x_c = torch.cat((x_c,x_lf_c),1)
        x_c = self.fuse_layer_clsfy(x_c)

        if self.include_top:  
            x_r = self.avgpool(x_r)
            x_r = torch.flatten(x_r, 1)
            x_c = self.avgpool(x_c)
            x_c = torch.flatten(x_c, 1)
            #x = self.actfunc(x)
            regression_output = self.fc(x_r)
            classification_output = self.bins_fc(x_c)
            classification_output = self.softmax(classification_output)
        #final_output = 0.5*regression_output + 0.5*torch.matmul(classification_output,self.bins)
        return regression_output, classification_output
    

class Net(nn.Module):
    def __init__(self, nchannel):
        super(Net, self).__init__()

        self.conv = nn.Sequential(  # The first loop of ConvLay er, ReLU, Pooling
            nn.Conv2d(in_channels=nchannel,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(64,momentum=0.1),
            nn.Tanh(),

            # self.conv2 = nn.Sequential(ResidualBlocks(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=1),
            #                           nn.Tanh())

            # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(128,momentum=0.1),
            nn.Tanh(),

            # self.conv4 = nn.Sequential(ResidualBlocks(in_channel=64,out_channel=64,kernel_size=3,stride=1,padding=1),
            #                           nn.Tanh())

            # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(256,momentum=0.1),
            nn.Tanh(),
            # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(512,momentum=0.1),
            nn.Tanh()
        )

        # self.ful1 = nn.Sequential(nn.Linear(256  * 5 * 5, 64), nn.BatchNorm1d(64))
        self.ful = nn.Sequential(nn.Linear(512 * 5 * 5, 64),  # , nn.BatchNorm1d(64), nn.Tanh())
                                 nn.Linear(64, 16),  # ,nn.BatchNorm1d(16), nn.Tanh())  # ,nn.Softmax())
                                 nn.Linear(16, 2),  # ,nn.BatchNorm1d(2), nn.Tanh())
                                 nn.Linear(2, 1))

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv(x)
        out = out.view(in_size, -1)
        output = self.ful(out)
        return output


