import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from Training_pkg.utils import *

import time

def save_daily_datesbased_model(model,evaluation_type, typeName, begindates,enddates, version, species, nchannel, special_name, ifold, **args):
    '''
    Evaluation type is not only applied to the evaluation, also the estimation model. 
    For estimation models, the ifold is set to 0.
    '''
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)
    d_model = args.get('d_model', 64)
    n_head = args.get('n_head', 8)
    ffn_hidden = args.get('ffn_hidden', 256)
    num_layers = args.get('num_layers', 6)
    max_len = args.get('max_len', 1000)
    CNN_nchannel = args.get('CNN_nchannel', 4)
    Transformer_nchannel = args.get('Transformer_nchannel', 3)

    
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/{}/'.format(species, version,evaluation_type)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'
        model_outfile = outdir +  '{}_{}_{}_{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, typeName, species, width,height, begindates,enddates,nchannel,special_name, ifold)
        torch.save(model, model_outfile)
    elif Apply_3D_CNN_architecture:
        if MoE_Settings:
            Model_structure_type = '3DCNN_MoE_{}Experts_Model'.format(MoE_num_experts)
        elif MoCE_Settings:
            Model_structure_type = '3DCNN_MoCE_{}Experts_Model'.format(MoCE_num_experts)
        else:
            Model_structure_type = 'CNN3DModel'
        model_outfile = outdir +  '{}_{}_{}_{}x{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, typeName, species, depth, width,height, begindates,enddates,nchannel,special_name, ifold)
        torch.save(model, model_outfile)
    
    elif Apply_Transformer_architecture:
        Model_structure_type = 'TransformerModel'
        model_outfile = outdir +  '{}_{}_{}_{}dmodel_{}heads_{}ffnHidden_{}numlayers_{}lens_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, typeName, species, d_model, n_head, ffn_hidden, num_layers, max_len, begindates,enddates,nchannel,special_name, ifold)
        torch.save(model, model_outfile)
    elif Apply_CNN_Transformer_architecture:
        Model_structure_type = 'CNNTransformerModel'
        model_outfile = outdir + f'{Model_structure_type}_{evaluation_type}_{typeName}_{species}_{width}x{height}_{d_model}dmodel_{n_head}heads_{ffn_hidden}ffnHidden_{num_layers}numlayers_{max_len}lens_{begindates}-{enddates}_{CNN_nchannel}CNNChannels_{Transformer_nchannel}TransformerChannels_No{ifold}.pt'
        torch.save(model, model_outfile)
    return

def load_daily_datesbased_model(evaluation_type, typeName, begindates,enddates, version, species, nchannel, special_name, ifold,**args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)
    d_model = args.get('d_model', 64)
    n_head = args.get('n_head', 8)
    ffn_hidden = args.get('ffn_hidden', 256)
    num_layers = args.get('num_layers', 6)
    max_len = args.get('max_len', 1000)
    CNN_nchannel = args.get('CNN_nchannel', 4)
    Transformer_nchannel = args.get('Transformer_nchannel', 3)

    indir = model_outdir + '{}/{}/Results/results-Trained_Models/{}/'.format(species, version,evaluation_type)
    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'
        model_infile = indir + '{}_{}_{}_{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, typeName, species, width,height, begindates,enddates,nchannel,special_name, ifold)
        
        if not os.path.isfile(model_infile):
            raise ValueError('The {} file does not exist!'.format(model_infile))
        
        model = torch.load(model_infile)
    elif Apply_3D_CNN_architecture:
        if MoE_Settings:
            Model_structure_type = '3DCNN_MoE_{}Experts_Model'.format(MoE_num_experts)
        elif MoCE_Settings:
            Model_structure_type = '3DCNN_MoCE_{}Experts_Model'.format(MoCE_num_experts)
        else:
            Model_structure_type = 'CNN3DModel'
        model_infile = indir + '{}_{}_{}_{}x{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, typeName, species, depth, width,height, begindates,enddates,nchannel,special_name, ifold)
        
        if not os.path.isfile(model_infile):
            raise ValueError('The {} file does not exist!'.format(model_infile))
        
        model = torch.load(model_infile)

    elif Apply_Transformer_architecture:
        Model_structure_type = 'TransformerModel'
        model_infile = indir + '{}_{}_{}_{}dmodel_{}heads_{}ffnHidden_{}numlayers_{}lens_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, typeName, species, d_model, n_head, ffn_hidden, num_layers, max_len, begindates,enddates,nchannel,special_name, ifold)
        
        if not os.path.isfile(model_infile):
            raise ValueError('The {} file does not exist!'.format(model_infile))
        
        model = torch.load(model_infile)
    
    elif Apply_CNN_Transformer_architecture:
        Model_structure_type = 'CNNTransformerModel'
        model_infile = indir + f'{Model_structure_type}_{typeName}_{species}_{width}x{height}_{d_model}dmodel_{n_head}heads_{ffn_hidden}ffnHidden_{num_layers}numlayers_{max_len}lens_{begindates}-{enddates}_{CNN_nchannel}CNNChannels_{Transformer_nchannel}TransformerChannels_No{ifold}.pt'

        if not os.path.isfile(model_infile):
            raise ValueError('The {} file does not exist!'.format(model_infile))
        
        model = torch.load(model_infile)

    return model


def _process_concatenate_site_data(data, temp_data, isite):
    site = str(isite)
    data[site]['data'] = np.concatenate((data[site]['data'], temp_data[site]['data']), axis=1)

def load_aggregate_TrainingDatasets(Training_Channels, sites_number):
    '''
    Here I compared the differences of time consuming between the old and new 
    version of loading and aggregating the training datasets.

    >>> data = load_aggregate_TrainingDatasets(channel_names,1684)
    The time of loading and aggregating the training datasets is 11.369443893432617 seconds
    >>> data = old_load_aggregate_TrainingDatasets(channel_names,1684)
    The time of old loading and aggregating the training datasets is 26.89457082748413 seconds 

    def old_load_aggregate_TrainingDatasets(Training_Channels, sites_number):
        start_time = time.time()
        
        for i, channel_name in enumerate(Training_Channels):
            infile = CNN_Training_infiles.format(channel_name)
            if os.path.exists(infile):
                temp_data = np.load(infile, allow_pickle=True).item()
            else:
                raise ValueError('The {} file does not exist!'.format(infile))
            
            if i == 0:
                data = temp_data
            else:
                for isite in range(sites_number):
                    site = str(isite)
                    data[site]['data'] = np.concatenate((data[site]['data'], temp_data[site]['data']), axis=1)
        end_time = time.time()
        print('The time of old loading and aggregating the training datasets is {} seconds'.format(end_time-start_time))
        return data'
    '''

    
    for i, channel_name in enumerate(Training_Channels):
        infile = CNN_Training_infiles.format(channel_name)
        if os.path.exists(infile):
            temp_data = np.load(infile, allow_pickle=True).item()
        else:
            raise ValueError('Training file - The {} file does not exist!'.format(infile))
        
        if i == 0:
            data = temp_data
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_concatenate_site_data, data, temp_data, isite) for isite in range(sites_number)]
                for future in futures:
                    future.result()
  
    return data

