import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv3d, BatchNorm3d, ReLU, MaxPool3d, AvgPool3d, Dropout3d
from Model_Structure_pkg.utils import *
from Training_pkg.utils import activation_function_table,define_activation_func, channel_names

def conv3x3x3(in_channels, out_channels, stride=1):
    """3x3x3 convolution with padding"""
    return Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,padding_mode=CNN3D_architeture_cfg)

activation = activation_function_table()

def resnet_block_lookup_table(blocktype):
    if blocktype == 'BasicBlock':
        return BasicBlock
    elif blocktype == 'Bottleneck':
        return Bottleneck
    else:
        print(' Wrong Key Word! BasicBlock or Bottleneck only! ')
        return None

def initial_3dcnn_net(main_stream_channel,wandb_config):
    main_stream_nchannel = len(main_stream_channel)
    block = resnet_block_lookup_table(ResCNN3D_Blocks)
    ResCNN3D_blocks_num = wandb_config['ResCNN3D_blocks_num']
    ResCNN3D_output_channels = wandb_config['ResCNN3D_output_channels']
    
    pooling_layer_switch = wandb_config['pooling_layer_switch']
    pooling_layer_type_3D = wandb_config['pooling_layer_type_3D']
    ResCNN3D_pooling_kernel_size = wandb_config['ResCNN3D_pooling_kernel_size']
    
    if MoE_Settings:
        MoE_num_experts = wandb_config['MoE_num_experts']
        MoE_gating_hidden_size = wandb_config['MoE_gating_hidden_size']
        MoE_selected_channels = wandb_config['MoE_selected_channels']
        selected_channels_index_for_gate = [main_stream_channel.index(ch) for ch in MoE_selected_channels]
        cnn3D_model = ResCNN3D_MoE(nchannel=main_stream_nchannel,num_experts=MoE_num_experts,
                                   selected_channels_index_for_gate=selected_channels_index_for_gate,
                                   blocks_num=ResCNN3D_blocks_num,
                                   output_channels=ResCNN3D_output_channels,
                                   num_classes=1,
                                   include_top=True,
                                   gating_hidden_dim=MoE_gating_hidden_size)
    elif MoCE_Settings:
        MoCE_num_experts = wandb_config['MoCE_num_experts']
        MoCE_gating_hidden_size = wandb_config['MoCE_gating_hidden_size']
        MoCE_selected_channels_index_for_gate = wandb_config['MoCE_selected_channels_index_for_gate']
        selected_channels_index_for_gate = [main_stream_channel.index(ch) for ch in MoCE_selected_channels_index_for_gate]    
        MoCE_base_model_channels = wandb_config['MoCE_base_model_channels']
        MoCE_side_blocks_num = wandb_config['MoCE_side_blocks_num']
        MoCE_side_output_channels = wandb_config['MoCE_side_output_channels']
        MoCE_side_pooling_kernel_switch = wandb_config['MoCE_side_pooling_kernel_switch']
        MoCE_side_pooling_layer_type_3D = wandb_config['MoCE_side_pooling_layer_type_3D']
        MoCE_side_pooling_kernel_size = wandb_config['MoCE_side_pooling_kernel_size']
        MoCE_side_experts_channels_list = wandb_config['MoCE_side_experts_channels_list']
        try:
            channels_to_add = wandb_config.get("channel_to_add", [])
        except AttributeError:
            channels_to_add = []
        MoCE_base_model_channels = MoCE_base_model_channels + channels_to_add ## the add of main_stream_channel is already done in the main function
        try:
            channels_to_exclude = wandb_config.get("channel_to_exclude", [])
        except AttributeError:
            channels_to_exclude = []
        for ichannel in range(len(channels_to_exclude)):
            if channels_to_exclude[ichannel] in MoCE_base_model_channels:
                MoCE_base_model_channels.remove(channels_to_exclude[ichannel])
            else:
                print('{} is not in the MoCE base model channel list.'.format(channels_to_exclude[ichannel]))
            for iexpert in range(len(MoCE_side_experts_channels_list)):
                if channels_to_exclude[ichannel] in MoCE_side_experts_channels_list[iexpert]:
                    MoCE_side_experts_channels_list[iexpert].remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the MoCE side expert {} channel list.'.format(channels_to_exclude[ichannel], iexpert))
        
        cnn3D_model = ResCNN3D_MoCE(num_experts=MoCE_num_experts,
                                    total_input_channels_list=main_stream_channel,
                                    selected_channels_index_for_gate=selected_channels_index_for_gate,
                                    CovLayer_padding_mode_3D=CovLayer_padding_mode_3D,
                                    Pooling_padding_mode_3D=Pooling_padding_mode_3D,
                                    base_model_channels=MoCE_base_model_channels,
                                    basemodel_blocks_num=ResCNN3D_blocks_num,
                                    basemodel_output_channels=ResCNN3D_output_channels,
                                    base_model_apply_pooling_layer=pooling_layer_switch,
                                    base_model_pooling_layer_type_3D=pooling_layer_type_3D,
                                    base_model_pooling_kernel_size=ResCNN3D_pooling_kernel_size,
                                    side_experts_channels_list=MoCE_side_experts_channels_list,
                                    side_model_blocks_num=MoCE_side_blocks_num,
                                    side_model_output_channels=MoCE_side_output_channels,
                                    side_model_apply_pooling_layer=MoCE_side_pooling_kernel_switch,
                                    side_model_pooling_layer_type_3D=MoCE_side_pooling_layer_type_3D,
                                    side_model_pooling_kernel_size=MoCE_side_pooling_kernel_size,
                                    num_classes=1,
                                    include_top=True,
                                    gating_hidden_dim=MoCE_gating_hidden_size,
                                    )
    else:
        cnn3D_model = ResCNN3D(nchannel=main_stream_nchannel,
                           block=block,
                           blocks_num=ResCNN3D_blocks_num,
                            output_channels=ResCNN3D_output_channels,
                            apply_pooling_layer=pooling_layer_switch,
                            pooling_layer_type_3D=pooling_layer_type_3D,
                            pooling_kernel_size=ResCNN3D_pooling_kernel_size,
                            CovLayer_padding_mode_3D=CovLayer_padding_mode_3D,
                            Pooling_padding_mode_3D=Pooling_padding_mode_3D,
                            num_classes=1,  # Assuming a single output for regression
                            include_top=True  # Include the top layer for classification/regression
    )
    return cnn3D_model

class BasicBlock(nn.Module):
    expansion = 1
    # expansion = 1 means that the output channels are equal to the input channels
    def __init__(self, in_channels, out_channels, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1),downsampele=None, activation='gelu'):
        super(BasicBlock, self).__init__()
        self.downsampele = downsampele
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = BatchNorm3d(out_channels)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = BatchNorm3d(out_channels)
        self.actfunc = define_activation_func(activation)
        
    def forward(self, x):
        residual = x
        if self.downsampele is not None:
            residual = self.downsampele(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actfunc(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.actfunc(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    # expansion = 4 means that the output channels are 4 times the input channels
    def __init__(self, in_channels, out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), downsampele=None):
        super(Bottleneck, self).__init__()
        self.downsampele = downsampele
        self.conv1 = Conv3d(in_channels, out_channels // 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = BatchNorm3d(out_channels // 4)
        self.conv2 = Conv3d(out_channels // 4, out_channels // 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = BatchNorm3d(out_channels // 4)
        self.conv3 = Conv3d(out_channels // 4, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn3 = BatchNorm3d(out_channels)
        
        self.actfunc = define_activation_func(activation)
    def forward(self, x):
        residual = x
        if self.downsampele is not None:
            residual = self.downsampele(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actfunc(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.actfunc(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + residual
        out = self.actfunc(out)
        return out
    
class ResCNN3D(nn.Module):

    def __init__(self, nchannel, # initial input channels
                    block, # block type, 'BasicBlock' or 'Bottleneck'
                    blocks_num, # number of blocks in each layer [n1, n2, n3, n4]   
                    output_channels, # output channels for each layer [c1, c2, c3, c4]
                    apply_pooling_layer,  # whether to apply pooling layer after the first conv layer
                    pooling_layer_type_3D,
                    pooling_kernel_size,
                    CovLayer_padding_mode_3D,
                    Pooling_padding_mode_3D,
                    num_classes=1, # number of output classes
                    include_top=True, # whether to include the top layer
                 ):
        super(ResCNN3D, self).__init__()
        self.in_channels = output_channels[0] ## output channel for the first layer ,and inchannel for other blocks
        self.include_top = include_top

        self.actfunc = define_activation_func(activation)
        self.apply_pooling_layer = apply_pooling_layer
        self.pooling_layer_type_3D = pooling_layer_type_3D
        self.CovLayer_padding_mode_3D = CovLayer_padding_mode_3D
        self.Pooling_padding_mode_3D = Pooling_padding_mode_3D
        self.pooling_kernel_size = pooling_kernel_size
        
        self.layer0 = nn.Sequential(
            Conv3d(nchannel, self.in_channels, kernel_size=(ResNet3D_depth,3,3), stride=(1,1,1), padding=(0,1,1), padding_mode=self.CovLayer_padding_mode_3D),
            BatchNorm3d(self.in_channels),
            self.actfunc,
        )
        if self.pooling_layer_type_3D == 'MaxPooling3d':
            self.pooling = nn.MaxPool3d(kernel_size=self.pooling_kernel_size, stride=(1,2,2)) # output 4x4
        elif self.pooling_layer_type_3D == 'AvgPooling3d':
            self.pooling = nn.AvgPool3d(kernel_size=self.pooling_kernel_size, stride=(1,2,2)) # output 4x4
        else:
            print('Pooling layer type not supported! Please use MaxPooling3d or AvgPooling3d.')
            self.apply_pooling_layer = False
        print('block type: {}, blocks_num: {}, output_channels: {}'.format(block, blocks_num, output_channels))
        self.layer1 = self._make_layer(block, blocks_num[0], output_channels[0],stride=1)
        self.layer2 = self._make_layer(block, blocks_num[1], output_channels[1], stride=1)
        self.layer3 = self._make_layer(block, blocks_num[2], output_channels[2], stride=1)
        self.layer4 = self._make_layer(block, blocks_num[3], output_channels[3],stride=1)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global average pooling
            self.fc = nn.Linear(self.in_channels * block.expansion, num_classes)  # Fully connected layer
        
        for m in self.modules():
            if isinstance(m, Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, block_num, output_channel, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != output_channel:
            downsample = nn.Sequential(
                Conv3d(self.in_channels, output_channel, kernel_size=(1,1,1), stride=stride),
                BatchNorm3d(output_channel * block.expansion),
            )
        
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channels, 
                                output_channel, 
                                stride=stride,
                                downsampele=downsample,
                                activation=activation))
            
            self.in_channels = output_channel * block.expansion
        
            for _ in range(1, block_num):
                layers.append(block(self.in_channels, output_channel,
                                    stride=1,
                                    downsampele=None,
                                    activation=activation))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
            
        x = self.layer0(x)
        if self.apply_pooling_layer:
            x = F.pad(x,pad=(1,1,1,1,0,0,),mode=self.Pooling_padding_mode_3D,value=0)
            x = self.pooling(x)
        #print('size of x after layer0: {}'.format(x.size()))
        x = self.layer1(x)
        #print('size of x after layer1: {}'.format(x.size()))
        x = self.layer2(x)
        #print('size of x after layer2: {}'.format(x.size()))
        x = self.layer3(x)
        #print('size of x after layer3: {}'.format(x.size()))
        x = self.layer4(x)
        #print('size of x after layer4: {}'.format(x.size()))

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x
    

class GatingNetwork3D_Subset(nn.Module):
    def __init__(self, in_channels_subset, num_experts, selected_channels, hidden_dim=64, activation='gelu'):
        """
        in_channels_subset : len(selected_channels)
        selected_channels  : list of channel indices to use for gating, e.g. [0, 1, 5, 10]
        """
        super().__init__()
        self.selected_channels = selected_channels
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_channels_subset, hidden_dim)
        self.actfunc = define_activation_func(activation)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # x: [B, C, D, H, W]
        x_sub = x[:, self.selected_channels, -1:, :, :]   # [B, C_sub, last day, H, W]
        pooled = self.avgpool(x_sub).view(x.size(0), -1)   # [B, C_sub]
        h = self.actfunc(self.fc1(pooled))
        logits = self.fc2(h)                        # [B, num_experts]
        gates = F.softmax(logits, dim=-1)           # [B, num_experts]
        print('gates[0,:]: ', gates[0,:])
        return gates, logits
    
class ResCNN3D_MoE(nn.Module):
    def __init__(self,
                 nchannel,
                 blocks_num,
                 output_channels, # output channels for each layer [c1, c2, c3, c4]
                 num_experts,
                 selected_channels_index_for_gate,
                 pooling_layer_type_3D,
                 pooling_kernel_size,
                 CovLayer_padding_mode_3D,
                 Pooling_padding_mode_3D,
                 num_classes=1,
                 include_top=True,
                 gating_hidden_dim=64,
                 ):
        super().__init__()
        # experts
        experts = []
        self.in_channels = output_channels[0] ## output channel for the first layer ,and inchannel for other blocks
        self.include_top = include_top
        
        for iexpert in range(num_experts):
            block = resnet_block_lookup_table(ResCNN3D_Blocks)
            expert_model = ResCNN3D(nchannel=nchannel,
                                    block=block,
                                    blocks_num=blocks_num,
                                    output_channels=output_channels,
                                    pooling_layer_type_3D=pooling_layer_type_3D,
                                    pooling_kernel_size=pooling_kernel_size,
                                    CovLayer_padding_mode_3D=CovLayer_padding_mode_3D,
                                    Pooling_padding_mode_3D=Pooling_padding_mode_3D,
                                    num_classes=num_classes,
                                    include_top=include_top)
            experts.append(expert_model)
        self.experts = nn.ModuleList(experts)
        
        # gating on subset of channels
        self.gating_network = GatingNetwork3D_Subset(
            in_channels_subset=len(selected_channels_index_for_gate),
            num_experts=num_experts,
            selected_channels=selected_channels_index_for_gate,
            hidden_dim=gating_hidden_dim,
            activation=activation
        )
        
    def forward(self, x):
        gates, logits = self.gating_network(x)      # only sees selected channels
        batch_size = x.size(0)

        expert_preds = []
        for expert in self.experts:
            y = expert(x)                           # experts still see all channels
            expert_preds.append(y.view(batch_size, 1))
        expert_outputs = torch.cat(expert_preds, dim=1)  # [B, num_experts]

        y_moe = torch.sum(gates * expert_outputs, dim=1, keepdim=True)
        return y_moe


class ResCNN3D_MoCE(nn.Module):
    def __init__(self,
                 num_experts,
                 total_input_channels_list,
                 selected_channels_index_for_gate,
                 CovLayer_padding_mode_3D,
                 Pooling_padding_mode_3D,
                 
                 base_model_channels,
                 basemodel_blocks_num,
                 basemodel_output_channels,
                 base_model_apply_pooling_layer,
                 base_model_pooling_layer_type_3D,
                 base_model_pooling_kernel_size,
                 
                 side_experts_channels_list,
                 side_model_blocks_num,
                 side_model_output_channels,
                 side_model_apply_pooling_layer,
                 side_model_pooling_layer_type_3D,
                 side_model_pooling_kernel_size,
                 
                 num_classes=1,
                 include_top=True,
                 gating_hidden_dim=64,
                 ):
        super().__init__()
        # experts
        experts = []
        self.total_input_channels_list = total_input_channels_list
        self.experts_channels_list = [base_model_channels] + side_experts_channels_list
        self.experts_channels_index = []
        self.selected_channels_index_for_gate = selected_channels_index_for_gate
        for iexpert in range(num_experts):
            channels = self.experts_channels_list[iexpert]
            channel_indices = [self.total_input_channels_list.index(ch) for ch in channels]
            self.experts_channels_index.append(channel_indices)
        # base model expert
        block = resnet_block_lookup_table(ResCNN3D_Blocks)
        nchannel = len(base_model_channels)
        base_expert_model = ResCNN3D(nchannel=nchannel,
                                    block=block,
                                    blocks_num=basemodel_blocks_num,
                                    output_channels=basemodel_output_channels,
                                    apply_pooling_layer=base_model_apply_pooling_layer,
                                    pooling_layer_type_3D=base_model_pooling_layer_type_3D,
                                    pooling_kernel_size=base_model_pooling_kernel_size,
                                    CovLayer_padding_mode_3D=CovLayer_padding_mode_3D,
                                    Pooling_padding_mode_3D=Pooling_padding_mode_3D,
                                    num_classes=num_classes,
                                    include_top=include_top)
        experts.append(base_expert_model)
        # side model experts
        for iexpert in range(num_experts - 1):
            expert_channels = side_experts_channels_list[iexpert]
            nchannel = len(expert_channels)
            side_expert_model = ResCNN3D(nchannel=nchannel,
                                        block=block,
                                        blocks_num=side_model_blocks_num,
                                        output_channels=side_model_output_channels,
                                        apply_pooling_layer=side_model_apply_pooling_layer,
                                        pooling_layer_type_3D=side_model_pooling_layer_type_3D,
                                        pooling_kernel_size=side_model_pooling_kernel_size,
                                        CovLayer_padding_mode_3D=CovLayer_padding_mode_3D,
                                        Pooling_padding_mode_3D=Pooling_padding_mode_3D,
                                        num_classes=num_classes,
                                        include_top=include_top)
            experts.append(side_expert_model)
        self.experts = nn.ModuleList(experts)
        
        # gating on subset of channels
        self.gating_network = GatingNetwork3D_Subset(
            in_channels_subset=len(selected_channels_index_for_gate),
            num_experts=num_experts,
            selected_channels=selected_channels_index_for_gate,
            hidden_dim=gating_hidden_dim,
            activation=activation
        )
    
    def _check_indices(self, x, idx, name):
        C = x.size(1)
        print('x size in _check_indices: ', x.size())
        print(f'Checking indices for {name}, total channels C={C}, indices: {len(idx)}')
        # idx can be list/np array/tensor
        if isinstance(idx, torch.Tensor):
            idx_cpu = idx.detach().cpu()
            mx, mn = int(idx_cpu.max()), int(idx_cpu.min())
        else:
            mx, mn = max(idx), min(idx)
        assert mn >= 0, f"{name}: has negative index (min={mn})"
        assert mx < C, f"{name}: index out of range (max={mx}, C={C})"
    def forward(self, x):   
        #self._check_indices(x, self.selected_channels_index_for_gate, "GatingNetwork3D_Subset")
        #for iexpert, expert in enumerate(self.experts):
        #    self._check_indices(x, self.experts_channels_index[iexpert], f"Expert {iexpert}")
        #    print(f'Expert {iexpert} indices: {self.experts_channels_index[iexpert]}')
        gates, logits = self.gating_network(x)      # only sees selected channels
        batch_size = x.size(0)

        expert_preds = []
        for iexpert, expert in enumerate(self.experts):
            x_expert = x[:, self.experts_channels_index[iexpert], :, :, :]  # select channels for this expert
            y = expert(x_expert)                           # experts only see selected channels
            expert_preds.append(y.view(batch_size, 1))
        expert_outputs = torch.cat(expert_preds, dim=1)  # [B, num_experts]

        y_moe = torch.sum(gates * expert_outputs, dim=1, keepdim=True)
        return y_moe
