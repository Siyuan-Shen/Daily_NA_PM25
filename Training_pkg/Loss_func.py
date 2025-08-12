import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Training_pkg.utils import *
from Model_Structure_pkg.utils import *

class SelfDesigned_LossFunction(nn.Module):
    def __init__(self,losstype,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SelfDesigned_LossFunction,self).__init__()
        self.Loss_Type = losstype
        self.reduction = reduction
        self.GeoMSE_Lamba1_Penalty1 = GeoMSE_Lamba1_Penalty1
        self.GeoMSE_Lamba1_Penalty2 = GeoMSE_Lamba1_Penalty2
        self.GeoMSE_Gamma  = GeoMSE_Gamma
    def forward(self,model_output,target,geophsical_species,geopysical_mean,geopysical_std,mask=None):
        if self.Loss_Type == 'MSE':
            if Apply_Transformer_architecture == True:
                """
                This is the mask
                Computes the mean squared error loss with a mask.
                
                predictions: (B, T, D)
                targets:     (B, T, D)
                mask:        (B, T) or (B, T, 1) with 1 for valid, 0 for invalid
                """
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                #print('predictions:', model_output.view(-1),'targets:', target.view(-1), 'predictions shape:', model_output.shape, 'targets shape:', target.shape)
                masked_model_output = model_output[torch.where(mask == 1)].view(-1)
                masked_target = target[torch.where(mask == 1)].view(-1)

                masked_loss = (masked_model_output - masked_target) ** 2
                
                #print('squared_error:', squared_error.view(-1),'mask:', mask.view(-1), 'masked_loss:', masked_loss.view(-1))
                #print('sum of masked_loss:', masked_loss.sum(),'masked_loss.view(-1): ', masked_loss.view(-1), 'sum of mask:', mask.sum())
                #loss = masked_loss.sum() / mask.sum().clamp(min=1e-8)  # avoid divide by zero
                loss = torch.sum(masked_loss.view(-1)) / mask.sum().clamp(min=1e-8)
                #print('MSE Loss: {}'.format(loss))
                return loss
            else:
                loss = F.mse_loss(model_output, target)
                #print('MSE Loss: {}'.format(loss))
                return loss
        
        elif self.Loss_Type == 'GeoMSE':
            geophsical_species = geophsical_species * geopysical_std + geopysical_mean
            MSE_loss = F.mse_loss(model_output, target)
            Penalty1 = self.GeoMSE_Lamba1_Penalty1 * torch.mean(torch.relu(-model_output - geophsical_species)) # To force the model output larger than -geophysical_species
            Penalty2 = self.GeoMSE_Lamba1_Penalty2 * torch.mean(torch.relu(model_output - self.GeoMSE_Gamma * geophsical_species)) # To force the model output larger than -geophysical_species
            loss = MSE_loss + Penalty1 + Penalty2
            print('Total loss: {}, MSE Loss: {}, Penalty 1: {}, Penalty 2: {}'.format(loss, MSE_loss, Penalty1, Penalty2))
            return loss

        elif self.Loss_Type == 'CrossEntropyLoss':
            loss = F.cross_entropy(model_output, target)
            print('CrossEntropyLoss: {}'.format(loss))
            return loss
        
