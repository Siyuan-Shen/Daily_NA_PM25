import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Training_pkg.utils import *

class SelfDesigned_LossFunction(nn.Module):
    def __init__(self,losstype,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SelfDesigned_LossFunction,self).__init__()
        self.Loss_Type = losstype
        self.reduction = reduction
        self.GeoMSE_Lamba1_Penalty1 = GeoMSE_Lamba1_Penalty1
        self.GeoMSE_Lamba1_Penalty2 = GeoMSE_Lamba1_Penalty2
        self.GeoMSE_Gamma  = GeoMSE_Gamma
    def forward(self,model_output,target,geophsical_species,geopysical_mean,geopysical_std):
        if self.Loss_Type == 'MSE':
            loss = F.mse_loss(model_output, target,reduction=self.reduction)
            #print('MSE Loss: {}'.format(loss))
            return loss
        
        elif self.Loss_Type == 'GeoMSE':
            geophsical_species = geophsical_species * geopysical_std + geopysical_mean
            MSE_loss = F.mse_loss(model_output, target)
            Penalty1 = self.GeoMSE_Lamba1_Penalty1 * torch.mean(torch.relu(-model_output - geophsical_species)) # To force the model output larger than -geophysical_species
            Penalty2 = self.GeoMSE_Lamba1_Penalty2 * torch.mean(torch.relu(model_output - self.GeoMSE_Gamma * geophsical_species)) # To force the model output larger than -geophysical_species
            loss = MSE_loss + Penalty1 + Penalty2
            #print('Total loss: {}, MSE Loss: {}, Penalty 1: {}, Penalty 2: {}'.format(loss, MSE_loss, Penalty1, Penalty2))
            return loss

        elif self.Loss_Type == 'CrossEntropyLoss':
            loss = F.cross_entropy(model_output, target)
            #print('CrossEntropyLoss: {}'.format(loss))
            return loss
        