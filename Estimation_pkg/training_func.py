import numpy as np
import gc
import os
import torch
import torch.nn as nn
import time
from Estimation_pkg.utils import *
from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import Split_Datasets_based_site_index,randomly_select_training_testing_indices,Get_final_output
from Evaluation_pkg.iostream import *
from Evaluation_pkg.Statistics_Calculation_func import calculate_statistics
from Model_Structure_pkg.CNN_Module import initial_cnn_network
from Model_Structure_pkg.ResCNN3D_Module import initial_3dcnn_net
from Model_Structure_pkg.Transformer_Model import Transformer_max_len, Transformer_spin_up_len
from Model_Structure_pkg.utils import *

from Training_pkg.utils import *
from Training_pkg.utils import epoch as config_epoch, batchsize as config_batchsize, learning_rate0 as config_learning_rate0
from Training_pkg.TensorData_func import Dataset_Val, Dataset
from Training_pkg.TrainingModule import CNN_train, cnn_predict, CNN3D_train, cnn_predict_3D,Transformer_train
from Training_pkg.data_func import CNNInputDatasets, CNN3DInputDatasets, TransformerInputDatasets
from Training_pkg.iostream import load_daily_datesbased_model
from multiprocessing import Manager
import torch.multiprocessing as mp

def Train_Model_forEstimation(total_channel_names, main_stream_channel_names,
                             side_stream_channel_names,sweep_id=None,):
    typeName = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=False, species=species)
    Evaluation_type = 'Estimation'
    world_size = torch.cuda.device_count()
    sweep_mode = False
    temp_sweep_config = None
    entity = None
    project = None
    name = None

    manager = Manager()
    run_id_container = manager.dict() 
    ### Load the datasets
    if Apply_CNN_architecture:
        ### Initialize the CNN datasets
        Model_structure_type = 'CNNModel'
        print('Init_CNN_Datasets starting...')
        start_time = time.time()
        Init_CNN_Datasets = CNNInputDatasets(species=species, total_channel_names=total_channel_names,bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,datapoints_threshold=observation_datapoints_threshold)
        print('Init_CNN_Datasets finished, time elapsed: ', time.time() - start_time)
        total_sites_number = Init_CNN_Datasets.total_sites_number
        true_input_mean, true_input_std = Init_CNN_Datasets.true_input_mean, Init_CNN_Datasets.true_input_std
        TrainingDatasets_mean, TrainingDatasets_std = Init_CNN_Datasets.TrainingDatasets_mean, Init_CNN_Datasets.TrainingDatasets_std
        width, height = Init_CNN_Datasets.width, Init_CNN_Datasets.height
        sites_lat, sites_lon = Init_CNN_Datasets.sites_lat, Init_CNN_Datasets.sites_lon
    elif Apply_3D_CNN_architecture:
        Model_structure_type = '3DCNNModel'
        print('Init_CNN_Datasets starting...')
        start_time = time.time()
        Init_CNN_Datasets = CNN3DInputDatasets(species=species, total_channel_names=total_channel_names,bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,datapoints_threshold=observation_datapoints_threshold)
        print('Init_CNN_Datasets finished, time elapsed: ', time.time() - start_time)
        total_sites_number = Init_CNN_Datasets.total_sites_number

        true_input_mean, true_input_std = Init_CNN_Datasets.true_input_mean, Init_CNN_Datasets.true_input_std
        TrainingDatasets_mean, TrainingDatasets_std = Init_CNN_Datasets.TrainingDatasets_mean, Init_CNN_Datasets.TrainingDatasets_std
        depth, width, height = Init_CNN_Datasets.depth,Init_CNN_Datasets.width, Init_CNN_Datasets.height
        sites_lat, sites_lon = Init_CNN_Datasets.sites_lat, Init_CNN_Datasets.sites_lon
    elif Apply_Transformer_architecture:
        Model_structure_type = 'TransformerModel'
        print('Init_Transformer_Datasets starting...')
        start_time = time.time()
        Init_Transformer_Datasets = TransformerInputDatasets(species=species, total_channel_names=total_channel_names,bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,datapoints_threshold=observation_datapoints_threshold)
        print('Init_Transformer_Datasets finished, time elapsed: ', time.time() - start_time)

        total_sites_number = Init_Transformer_Datasets.total_sites_number

        true_input_mean, true_input_std = Init_Transformer_Datasets.true_input_mean, Init_Transformer_Datasets.true_input_std
        TrainingDatasets_mean, TrainingDatasets_std = Init_Transformer_Datasets.TrainingDatasets_mean, Init_Transformer_Datasets.TrainingDatasets_std
        sites_lat, sites_lon = Init_Transformer_Datasets.sites_lat, Init_Transformer_Datasets.sites_lon
    

    for imodel in range(len(Estimation_begindates)):
        ### Get the training and targets in desired range
        if Apply_CNN_architecture or Apply_3D_CNN_architecture:
            # Get the initial true_input and training datasets for the current model (within the desired time range)
            print('1...',' Start Date: ', Estimation_begindates[imodel], ' End Date: ', Estimation_enddates[imodel])
            desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=Estimation_begindates[imodel],
                                                                                            end_date=Estimation_enddates[imodel]) # initial datasets
            # Normalize the training datasets
            print('2...', 'Desired Training Datasets: ', desired_trainingdatasets.keys())
            normalized_TrainingDatasets  = Init_CNN_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
            del desired_trainingdatasets
            gc.collect()
            # Concatenate the training datasets and true input for the current model for training and tetsing purposes
            
        elif Apply_Transformer_architecture:
            # Get the initial true_input and training datasets for the current model (within the desired time range)
            print('1...',' Start Date: ', Estimation_begindates[imodel], ' End Date: ', Estimation_enddates[imodel])
            desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_Transformer_Datasets.get_desired_range_inputdatasets(start_date=Estimation_begindates[imodel],
                                                                                            end_date=Estimation_enddates[imodel],max_len=Transformer_max_len,spinup_len=Transformer_spin_up_len)
            # Normalize the training datasets
            print('2...', 'Desired Training Datasets: ', desired_trainingdatasets.keys())
            normalized_TrainingDatasets  = Init_Transformer_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
            del desired_trainingdatasets
            gc.collect()

        if Apply_3D_CNN_architecture or Apply_CNN_architecture:
            print('3...')
            cctnd_trainingdatasets, cctnd_true_input,cctnd_ground_observation_data,cctnd_geophysical_species_data, cctnd_sites_index, cctnd_dates = Init_CNN_Datasets.concatenate_trainingdatasets(desired_true_input=desired_true_input, 
                                                                                                                                desired_normalized_trainingdatasets=normalized_TrainingDatasets,
                                                                                                                                desired_ground_observation_data=desired_ground_observation_data,
                                                                                                                                desired_geophysical_species_data=desired_geophysical_species_data)
        elif Apply_Transformer_architecture:
            # Concatenate the training datasets and true input for the current model for training and testing purposes
            print('3...')
            cctnd_trainingdatasets, cctnd_true_input,cctnd_ground_observation_data,cctnd_geophysical_species_data, cctnd_sites_index, cctnd_dates = Init_Transformer_Datasets.concatenate_trainingdatasets(desired_true_input=desired_true_input, 
                                                                                                                                desired_normalized_trainingdatasets=normalized_TrainingDatasets,
                                                                                                                                desired_ground_observation_data=desired_ground_observation_data,
                                                                                                                                desired_geophysical_species_data=desired_geophysical_species_data)            
        
        X_test, y_test = cctnd_trainingdatasets[0:10000,:], cctnd_true_input[0:10000,:] ## Just to get the shape of the input data for the model to feed into the function. No need to get the validation datasets for estimation.

        if Apply_CNN_architecture:
            if world_size > 1:
                mp.spawn(CNN_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,cctnd_trainingdatasets, cctnd_true_input,\
                                    X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                Evaluation_type,typeName,Estimation_begindates[imodel],\
                                Estimation_enddates[imodel],0),nprocs=world_size)
            else:
                CNN_train(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,cctnd_trainingdatasets, cctnd_true_input,\
                                    X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                Evaluation_type,typeName,Estimation_begindates[imodel],\
                                Estimation_enddates[imodel],0)
        elif Apply_3D_CNN_architecture:

            if world_size > 1:
                mp.spawn(CNN3D_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,cctnd_trainingdatasets, cctnd_true_input,\
                                    X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                    Evaluation_type,typeName,Estimation_begindates[imodel],\
                                    Estimation_enddates[imodel],0),nprocs=world_size)
            else:
                CNN3D_train(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,cctnd_trainingdatasets, cctnd_true_input,\
                                    X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                    Evaluation_type,typeName,Estimation_begindates[imodel],\
                                    Estimation_enddates[imodel],0)
        elif Apply_Transformer_architecture:
            if world_size > 1:
                mp.spawn(Transformer_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,cctnd_trainingdatasets, cctnd_true_input,\
                                    X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std, \
                                    Evaluation_type,typeName,Estimation_begindates[imodel],\
                                    Estimation_enddates[imodel],0),nprocs=world_size)
            else:
                Transformer_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,cctnd_trainingdatasets, cctnd_true_input,\
                                    X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std, \
                                    Evaluation_type,typeName,Estimation_begindates[imodel],\
                                    Estimation_enddates[imodel],0)

    if Apply_3D_CNN_architecture or Apply_CNN_architecture:
        del Init_CNN_Datasets
    elif Apply_Transformer_architecture:
        del Init_Transformer_Datasets
    
    del Init_CNN_Datasets
    gc.collect()
                
    return