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
    
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/{}/'.format(species, version,evaluation_type)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'
        model_outfile = outdir +  '{}_{}_{}_{}_{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, evaluation_type, typeName, species, width,height, begindates,enddates,nchannel,special_name, ifold)
        torch.save(model, model_outfile)
    elif Apply_3D_CNN_architecture:
        Model_structure_type = '3DCNNModel'
        model_outfile = outdir +  '{}_{}_{}_{}_{}x{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, evaluation_type, typeName, species, depth, width,height, begindates,enddates,nchannel,special_name, ifold)
        torch.save(model, model_outfile)
    return

def load_daily_datesbased_model(evaluation_type, typeName, begindates,enddates, version, species, nchannel, special_name, ifold,**args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)
    indir = model_outdir + '{}/{}/Results/results-Trained_Models/{}/'.format(species, version,evaluation_type)
    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'
        model_infile = indir + '{}_{}_{}_{}_{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, evaluation_type, typeName, species, width,height, begindates,enddates,nchannel,special_name, ifold)
        
        if not os.path.isfile(model_infile):
            raise ValueError('The {} file does not exist!'.format(model_infile))
        
        model = torch.load(model_infile)
    elif Apply_3D_CNN_architecture:
        Model_structure_type = '3DCNNModel'
        model_infile = indir + '{}_{}_{}_{}_{}x{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, evaluation_type, typeName, species, depth, width,height, begindates,enddates,nchannel,special_name, ifold)
        
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

