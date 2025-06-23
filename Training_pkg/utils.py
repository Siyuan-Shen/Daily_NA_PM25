import toml
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Model_Structure_pkg.utils import *
cfg = toml.load('./config.toml')

####################################################################################
###                                 Paths Settings                               ###
####################################################################################
# Observation Path
obs_dir = cfg['Pathway']['learning_objective']

geophysical_species_data_dir    = obs_dir['geophysical_species_data_dir']
geophysical_biases_data_dir     = obs_dir['geophysical_biases_data_dir']
ground_observation_data_dir     = obs_dir['ground_observation_data_dir']
geophysical_species_data_infile = obs_dir['geophysical_species_data_infile']
geophysical_biases_data_infile  = obs_dir['geophysical_biases_data_infile']
ground_observation_data_infile  = obs_dir['ground_observation_data_infile']

################################################################################
# Training Path
training_dir_cfg = cfg['Pathway']['TrainingDataset']

CNN_Training_infiles = training_dir_cfg['CNN_Training_infiles']
CNN3D_Training_infiles = training_dir_cfg['CNN3D_Training_infiles']

####################################################################################
###                                Training Settings                             ###
####################################################################################
Training_Settings = cfg['Training-Settings']
# Identity
identity_cfg      = Training_Settings['identity']
version            = identity_cfg['version']
description       = identity_cfg['description']
author            = identity_cfg['author']
email             = identity_cfg['email']
runningdate       = identity_cfg['runningdate']

# learning objective
learning_objective_cfg = Training_Settings['learning-objective']
species = learning_objective_cfg['species']
bias = learning_objective_cfg['bias']
normalize_bias = learning_objective_cfg['normalize_bias']
normalize_species = learning_objective_cfg['normalize_species']
absolute_species = learning_objective_cfg['absolute_species']


# hyperparameters
hyperparameters_cfg = Training_Settings['hyper-parameters']
epoch = hyperparameters_cfg['epoch']
batchsize = hyperparameters_cfg['batchsize']
channel_names = hyperparameters_cfg['channel_names']

# Loss function
loss_function_cfg = Training_Settings['Loss-Functions']
Regression_loss_type = loss_function_cfg['Regression_loss_type']
Classification_loss_type = loss_function_cfg['Classification_loss_type']

GeoMSE_loss_function_cfg = loss_function_cfg['GeoMSE']
GeoMSE_Lamba1_Penalty1 = GeoMSE_loss_function_cfg['GeoMSE_Lamba1_Penalty1']
GeoMSE_Lamba1_Penalty2 = GeoMSE_loss_function_cfg['GeoMSE_Lamba1_Penalty2']
GeoMSE_Gamma = GeoMSE_loss_function_cfg['GeoMSE_Gamma']

MultiHead_Loss_function_cfg = loss_function_cfg['MultiHead_Loss']
ResNet_MultiHeadNet_regression_loss_coefficient = MultiHead_Loss_function_cfg['ResNet_MultiHeadNet_regression_loss_coefficient']
ResNet_MultiHeadNet_classfication_loss_coefficient = MultiHead_Loss_function_cfg['ResNet_MultiHeadNet_classfication_loss_coefficient']

## Optimizer
optimizer_cfg = Training_Settings['optimizer']
Adam_cfg= optimizer_cfg['Adam']
Adam_Settings = Adam_cfg['Settings']
Adam_beta0 = Adam_cfg['beta0']
Adam_beta1 = Adam_cfg['beta1']
Adam_eps = Adam_cfg['eps']

## Learning Rate
learning_rate_cfg = Training_Settings['learning_rate']

learning_rate0 = learning_rate_cfg['learning_rate0']

ExponentialLR_cfg = learning_rate_cfg['ExponentialLR']
ExponentialLR_Settings = ExponentialLR_cfg['Settings']
ExponentialLR_gamma = ExponentialLR_cfg['gamma']

CosineAnnealingLR_cfg = learning_rate_cfg['CosineAnnealingLR']
CosineAnnealingLR_Settings = CosineAnnealingLR_cfg['Settings']
CosineAnnealingLR_T_max = CosineAnnealingLR_cfg['T_max']
CosineAnnealingLR_eta_min = CosineAnnealingLR_cfg['eta_min']

CosineAnnealingRestartsLR_cfg = learning_rate_cfg['CosineAnnealingRestartsLR']
CosineAnnealingRestartsLR_Settings = CosineAnnealingRestartsLR_cfg['Settings']
CosineAnnealingRestartsLR_T_0 = CosineAnnealingRestartsLR_cfg['T_0']
CosineAnnealingRestartsLR_T_mult = CosineAnnealingRestartsLR_cfg['T_mult']
CosineAnnealingRestartsLR_eta_min = CosineAnnealingRestartsLR_cfg['eta_min']

## Activation Function
activation_function_cfg = Training_Settings['activation_func']
activation_func_name = activation_function_cfg['activation_func_name']
ReLU_ACF = activation_function_cfg['ReLU']['Settings']
Tanh_ACF = activation_function_cfg['Tanh']['Settings']
GeLU_ACF = activation_function_cfg['GeLU']['Settings']
Sigmoid_ACF = activation_function_cfg['Sigmoid']['Settings']
Mish_ACF = activation_function_cfg['Mish']['Settings']
ELU_ACF = activation_function_cfg['ELU']['Settings']


def Get_channel_names(channels_to_exclude:list):
    if ResNet_Settings or ResNet_MLP_Settings or ResNet_Classification_Settings or ResNet_MultiHeadNet_Settings:
        if len(channels_to_exclude) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = channel_names.copy()
            side_channel_names = []
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = channel_names.copy()
            side_channel_names = []
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
    elif LateFusion_Settings:
        if len(channels_to_exclude) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = LateFusion_initial_channels.copy()
            side_channel_names = LateFusion_LateFusion_channels.copy()
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = LateFusion_initial_channels.copy()
            side_channel_names = LateFusion_LateFusion_channels.copy()
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in side_channel_names:
                    side_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the side channel list.'.format(channels_to_exclude[ichannel]))
    elif MultiHeadLateFusion_Settings:
        if len(channels_to_exclude) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = MultiHeadLateFusion_initial_channels.copy()
            side_channel_names = MultiHeadLateFusion_LateFusion_channels.copy()
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = MultiHeadLateFusion_initial_channels.copy()
            side_channel_names = MultiHeadLateFusion_LateFusion_channels.copy()
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in side_channel_names:
                    side_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the side channel list.'.format(channels_to_exclude[ichannel]))

    return total_channel_names, main_stream_channel_names, side_channel_names

def activation_function_table():
    if ReLU_ACF == True:
        return 'relu' #nn.ReLU()
    elif Tanh_ACF == True:
        return 'tanh' #nn.Tanh()
    elif GeLU_ACF == True:
        return 'gelu' #nn.GELU()
    elif Sigmoid_ACF == True:
        return 'sigmoid' #nn.Sigmoid()
    elif Mish_ACF == True:
        return 'mish'
    elif ELU_ACF == True:
        return 'elu'

def define_activation_func(activation_func_name):
    if activation_func_name == 'relu':
        return nn.ReLU()
    elif activation_func_name == 'tanh':
        return nn.Tanh()
    elif activation_func_name == 'gelu':
        return nn.GELU()
    elif activation_func_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_func_name == 'mish':
        return nn.Mish()
    elif activation_func_name == 'elu':
        return nn.ELU()
    
def lr_strategy_lookup_table(optimizer):
    if ExponentialLR_Settings:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=ExponentialLR_gamma)
    elif CosineAnnealingLR_Settings:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=CosineAnnealingLR_T_max,eta_min=CosineAnnealingLR_eta_min)
    elif CosineAnnealingRestartsLR_Settings:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=CosineAnnealingRestartsLR_T_0,T_mult=CosineAnnealingRestartsLR_T_mult,eta_min=CosineAnnealingLR_eta_min)

def optimizer_lookup(model_parameters,learning_rate):
    if Adam_Settings:
        return torch.optim.Adam(params=model_parameters,betas=(Adam_beta0, Adam_beta1),eps=Adam_eps, lr=learning_rate)

