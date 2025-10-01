import numpy as np
import torch
import torch.nn as nn
import os
import gc
from sklearn.model_selection import RepeatedKFold
import random
import csv
import shap
import wandb
import time

import torch.multiprocessing as mp

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import Split_Datasets_based_site_index,randomly_select_training_testing_indices,Get_final_output
from Evaluation_pkg.iostream import *
from Evaluation_pkg.Statistics_Calculation_func import calculate_statistics
from Model_Structure_pkg.CNN_Module import initial_cnn_network
from Model_Structure_pkg.ResCNN3D_Module import initial_3dcnn_net
from Model_Structure_pkg.utils import *

from Training_pkg.utils import *
from Training_pkg.utils import epoch as config_epoch, batchsize as config_batchsize, learning_rate0 as config_learning_rate0
from Training_pkg.TensorData_func import Dataset_Val, Dataset
from Training_pkg.TrainingModule import CNN_train, cnn_predict, CNN3D_train, cnn_predict_3D,Transformer_train,transformer_predict, CNN_Transformer_train, cnn_transformer_predict
from Training_pkg.data_func import CNNInputDatasets, CNN3DInputDatasets,TransformerInputDatasets, CNN_Transformer_InputDatasets
from Training_pkg.iostream import load_daily_datesbased_model
from wandb_config import init_get_sweep_config
from multiprocessing import Manager

####################################################################################
###                                Hyperparameters Search                         ###
####################################################################################
# This module is used to help search the optimized hyperparameters for the CNN model
# Here the cross-validation is not applied to accelerate the training and validation process
# This is only applied at the initial stage, and the final products should be validated by the cross-validation
# The hyperparameters includes:
# 1. Learning rate
# 2. Batch size
# 3. Total epochs
# 4. CNN model architecture
# 5. Loss function
# 6. Optimizer
# 7. Scheduler
# 8. Regularization


def Hyperparameters_Search_Training_Testing_Validation(total_channel_names,main_stream_channel_names,side_stream_channel_names,sweep_id=None):
    
    typeName = Get_typeName(bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species, log_species=False, species=species)
    world_size = torch.cuda.device_count()

    if HSV_Spatial_splitting_Switch:
            Evaluation_type = 'Hyperparameters_Search_Validation_Spatial_Splitting'

    if HSV_Apply_wandb_sweep_Switch:
        sweep_mode = True
        temp_sweep_config = init_get_sweep_config()
        entity = temp_sweep_config.get("entity", "ACAG-NorthAmericaDailyPM25")
        project = temp_sweep_config.get("project", version)
        name = temp_sweep_config.get("name", None)
        if Apply_Transformer_architecture or Apply_CNN_Transformer_architecture:
            d_model, n_head, ffn_hidden, num_layers, max_len,spin_up_len = temp_sweep_config.get("d_model", 64), temp_sweep_config.get("n_head", 8), temp_sweep_config.get("ffn_hidden", 256), temp_sweep_config.get("num_layers", 6), temp_sweep_config.get("max_len", 1000), temp_sweep_config.get("spin_up_len", 100)
        
    else:
        sweep_mode = False
        temp_sweep_config = None
        entity = None
        project = None
        name = None
        if Apply_Transformer_architecture:
            d_model, n_head, ffn_hidden, num_layers, max_len,spin_up_len = Transformer_d_model, Transformer_n_head, Transformer_ffn_hidden, Transformer_num_layers, Transformer_max_len, Transformer_spin_up_len
        if Apply_CNN_Transformer_architecture:
            d_model, n_head, ffn_hidden, num_layers, max_len,spin_up_len = CNN_Transformer_d_model, CNN_Transformer_n_head, CNN_Transformer_ffn_hidden, CNN_Transformer_num_layers, CNN_Transformer_max_len, CNN_Transformer_spin_up_len
    #####################################################################
    if Apply_CNN_architecture:
        
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
        print('Init_3DCNN_Datasets starting...')
        start_time = time.time()
        Init_CNN_Datasets = CNN3DInputDatasets(species=species, total_channel_names=total_channel_names,bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,datapoints_threshold=observation_datapoints_threshold)
        print('Init_3DCNN_Datasets finished, time elapsed: ', time.time() - start_time)

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

    elif Apply_CNN_Transformer_architecture:
        Model_structure_type = 'CNNTransformerModel'
        print('Init_CNN_Transformer_Datasets starting...')
        start_time = time.time()
        Init_CNN_Datasets = CNN_Transformer_InputDatasets(species=species, cnn_channel_names=CNN_Embedding_channel_names, transformer_channel_names=Transformer_Embedding_channel_names, bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species, datapoints_threshold=observation_datapoints_threshold)
        print('Init_CNN_Datasets finished, time elapsed: ', time.time() - start_time)
        total_sites_number = Init_CNN_Datasets.total_sites_number
        true_input_mean, true_input_std = Init_CNN_Datasets.true_input_mean, Init_CNN_Datasets.true_input_std
        CNN_trainingdatasets_mean, CNN_trainingdatasets_std = Init_CNN_Datasets.CNN_trainingdatasets_mean, Init_CNN_Datasets.CNN_trainingdatasets_std
        Transformer_trainingdatasets_mean, Transformer_trainingdatasets_std = Init_CNN_Datasets.Transformer_trainingdatasets_mean, Init_CNN_Datasets.Transformer_trainingdatasets_std
        width, height = Init_CNN_Datasets.width, Init_CNN_Datasets.height
        sites_lat, sites_lon = Init_CNN_Datasets.sites_lat, Init_CNN_Datasets.sites_lon


    # Start the hyperparameters search validation
    Statistics_list = ['test_R2','train_R2','geo_R2','RMSE','NRMSE','slope','PWA']
    if not Use_recorded_data_to_show_validation_results:
            final_data_recording = np.array([],dtype=float)
            obs_data_recording = np.array([],dtype=float)
            geo_data_recording = np.array([],dtype=float)
            sites_recording = np.array([],dtype=int)
            dates_recording = np.array([],dtype=int)

            training_final_data_recording = np.array([],dtype=float)
            training_obs_data_recording = np.array([],dtype=float)
            training_sites_recording = np.array([],dtype=int)
            training_dates_recording = np.array([],dtype=int)
            manager = Manager()
            run_id_container = manager.dict() 

            if HSV_Spatial_splitting_Switch:
                
                for imodel in range(len(HSV_Spatial_splitting_begindates)):
                    if Apply_CNN_architecture or Apply_3D_CNN_architecture:
                        # Get the initial true_input and training datasets for the current model (within the desired time range)
                        print('1...')
                        desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=HSV_Spatial_splitting_begindates[imodel],
                                                                                                        end_date=HSV_Spatial_splitting_enddates[imodel]) # initial datasets
                        # Normalize the training datasets
                        print('2...')
                        normalized_TrainingDatasets  = Init_CNN_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
                        
                        del desired_trainingdatasets
                        gc.collect()
                        # Concatenate the training datasets and true input for the current model for training and tetsing purposes
                        print('3...')
                        cctnd_trainingdatasets, cctnd_true_input,cctnd_ground_observation_data,cctnd_geophysical_species_data, cctnd_sites_index, cctnd_dates = Init_CNN_Datasets.concatenate_trainingdatasets(desired_true_input=desired_true_input, 
                                                                                                                                                desired_normalized_trainingdatasets=normalized_TrainingDatasets,
                                                                                                                                                desired_ground_observation_data=desired_ground_observation_data,
                                                                                                                                                desired_geophysical_species_data=desired_geophysical_species_data)
                    
                    elif Apply_Transformer_architecture:
                        # Get the initial true_input and training datasets for the current model (within the desired time range)
                        print('1...')
                        desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_Transformer_Datasets.get_desired_range_inputdatasets(start_date=HSV_Spatial_splitting_begindates[imodel],
                                                                                                        end_date=HSV_Spatial_splitting_enddates[imodel],max_len=max_len,spinup_len=spin_up_len)
                        # Normalize the training datasets
                        print('2...')
                        normalized_TrainingDatasets  = Init_Transformer_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
                        del desired_trainingdatasets
                        gc.collect()
                        # Concatenate the training datasets and true input for the current model for training and tetsing purposes
                        print('3...')
                        cctnd_trainingdatasets, cctnd_true_input,cctnd_ground_observation_data,cctnd_geophysical_species_data, cctnd_sites_index, cctnd_dates = Init_Transformer_Datasets.concatenate_trainingdatasets(desired_true_input=desired_true_input, 
                                                                                                                                                desired_normalized_trainingdatasets=normalized_TrainingDatasets,
                                                                                                                                                desired_ground_observation_data=desired_ground_observation_data,
                                                                                                                                                desired_geophysical_species_data=desired_geophysical_species_data)
                    elif Apply_CNN_Transformer_architecture:
                        # Get the initial true_input and training datasets for the current model (within the desired time range)
                        print('1...')
                        desired_CNN_trainingdatasets, desired_Transformer_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=HSV_Spatial_splitting_begindates[imodel],
                                                                                                        end_date=HSV_Spatial_splitting_enddates[imodel],max_len=max_len,spinup_len=spin_up_len)
                        # Normalize the training datasets
                        print('2...')
                        normalized_CNN_TrainingDatasets,normalized_Transformer_TrainingDatasets  = Init_CNN_Datasets.normalize_trainingdatasets(desired_CNN_trainingdatasets=desired_CNN_trainingdatasets, desired_Transformer_trainingdatasets=desired_Transformer_trainingdatasets)
                        del desired_CNN_trainingdatasets, desired_Transformer_trainingdatasets
                        gc.collect()
                        # Concatenate the training datasets and true input for the current model for training and tetsing purposes
                        print('3...')
                        cctnd_CNN_trainingdatasets, cctnd_Transformer_trainingdatasets, cctnd_true_input,cctnd_ground_observation_data,cctnd_geophysical_species_data, cctnd_sites_index, cctnd_dates = Init_CNN_Datasets.concatenate_trainingdatasets(desired_true_input=desired_true_input, 
                                                                                                                                                desired_normalized_CNN_trainingdatasets=normalized_CNN_TrainingDatasets,
                                                                                                                                                desired_normalized_Transformer_trainingdatasets=normalized_Transformer_TrainingDatasets,
                                                                                                                                                desired_ground_observation_data=desired_ground_observation_data,
                                                                                                                                                desired_geophysical_species_data=desired_geophysical_species_data)
                    print('4...')
                    training_selected_sites, testing_selected_sites = randomly_select_training_testing_indices(sites_index=np.arange(total_sites_number), training_portion=HSV_Spatial_splitting_training_portion)
                    print('training_selected_sites: ',training_selected_sites.shape)
                    print('testing_selected_sites: ',testing_selected_sites.shape) 

                    if Apply_CNN_Transformer_architecture:
                        print('cctnd_CNN_trainingdatasets[0]: ', cctnd_CNN_trainingdatasets[0])
                        print('cctnd_Transformer_trainingdatasets[0]: ', cctnd_Transformer_trainingdatasets[0])
                        X_train_CNN, y_train, X_test_CNN, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datasets_based_site_index(train_site_index=training_selected_sites,
                                                                                                                                        test_site_index=testing_selected_sites,
                                                                                                                                        total_trainingdatasets=cctnd_CNN_trainingdatasets,
                                                                                                                                        total_true_input=cctnd_true_input,
                                                                                                                                        total_sites_index=cctnd_sites_index,
                                                                                                                                        total_dates=cctnd_dates)
                        X_train_Transformer, y_train, X_test_Transformer, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datasets_based_site_index(train_site_index=training_selected_sites,
                                                                                                                                        test_site_index=testing_selected_sites,
                                                                                                                                        total_trainingdatasets=cctnd_Transformer_trainingdatasets,
                                                                                                                                        total_true_input=cctnd_true_input,
                                                                                                                                        total_sites_index=cctnd_sites_index,
                                                                                                                                        total_dates=cctnd_dates)
                        print('cctnd_ground_observation_data[test_datasets_index]: ', cctnd_ground_observation_data[test_datasets_index])
                        print('cctnd_geophysical_species_data[test_datasets_index]: ', cctnd_geophysical_species_data[test_datasets_index])
                        print('test_datasets_index: ', test_datasets_index)
                        
                        del cctnd_CNN_trainingdatasets, cctnd_Transformer_trainingdatasets, cctnd_true_input
                        gc.collect()
                    else:
                        print('cctnd_trainingdatasets[0]: ', cctnd_trainingdatasets[0])
                        X_train, y_train, X_test, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datasets_based_site_index(train_site_index=training_selected_sites,
                                                                                                                                            test_site_index=testing_selected_sites,
                                                                                                                                            total_trainingdatasets=cctnd_trainingdatasets,
                                                                                                                                            total_true_input=cctnd_true_input,
                                                                                                                                            total_sites_index=cctnd_sites_index,
                                                                                                                                            total_dates=cctnd_dates)
                        print('cctnd_ground_observation_data[test_datasets_index]: ', cctnd_ground_observation_data[test_datasets_index])
                        print('cctnd_geophysical_species_data[test_datasets_index]: ', cctnd_geophysical_species_data[test_datasets_index])
                        print('test_datasets_index: ', test_datasets_index)
                        
                        del cctnd_trainingdatasets, cctnd_true_input
                        gc.collect()
                    
                    if Apply_CNN_architecture:
                           
                        if world_size > 1:
                            mp.spawn(CNN_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                  X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                                Evaluation_type,typeName,HSV_Spatial_splitting_begindates[imodel],\
                                                HSV_Spatial_splitting_enddates[imodel],0),nprocs=world_size)
                        else:
                            CNN_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                  X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                                Evaluation_type,typeName,HSV_Spatial_splitting_begindates[imodel],\
                                                HSV_Spatial_splitting_enddates[imodel],0)
                        try:
                            channels_to_exclude = temp_sweep_config.get("channel_to_exclude", [])
                        except AttributeError:
                            channels_to_exclude = []

                        excluded_total_channel_names, main_stream_channel_names, side_stream_channel_names = Get_channel_names(channels_to_exclude=channels_to_exclude)
                        index_of_main_stream_channels_of_initial = [total_channel_names.index(channel) for channel in main_stream_channel_names]
                        X_train = X_train[:,index_of_main_stream_channels_of_initial,:,:]
                        X_test  = X_test[:,index_of_main_stream_channels_of_initial,:,:]

                        # Since in hyperparameter searching we do not apply multiple tests, we only see the final testing accuracy, so no loop here in 
                        # different time ranges. 
                        Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=HSV_Spatial_splitting_begindates[imodel],
                                                                    enddates=HSV_Spatial_splitting_enddates[imodel], version=version,species=species,
                                                                    nchannel=len(main_stream_channel_names),special_name=description,ifold=0,width=width,height=height)
                        validation_output = cnn_predict(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                        mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                        training_output = cnn_predict(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                        mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                    
                    if Apply_3D_CNN_architecture:
                        
                        if world_size > 1:
                            mp.spawn(CNN3D_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                  X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                                Evaluation_type,typeName,HSV_Spatial_splitting_begindates[imodel],\
                                                HSV_Spatial_splitting_enddates[imodel],0),nprocs=world_size)
                        else:
                            CNN3D_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                  X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                                Evaluation_type,typeName,HSV_Spatial_splitting_begindates[imodel],\
                                                HSV_Spatial_splitting_enddates[imodel],0)
                        try:
                            channels_to_exclude = temp_sweep_config.get("channel_to_exclude", [])
                        except AttributeError:
                            channels_to_exclude = []

                        excluded_total_channel_names, main_stream_channel_names, side_stream_channel_names = Get_channel_names(channels_to_exclude=channels_to_exclude)
                        index_of_main_stream_channels_of_initial = [total_channel_names.index(channel) for channel in main_stream_channel_names]
                        X_train = X_train[:,index_of_main_stream_channels_of_initial,:,:,:]
                        X_test  = X_test[:,index_of_main_stream_channels_of_initial,:,:,:]

                        # Since in hyperparameter searching we do not apply multiple tests, we only see the final testing accuracy, so no loop here in 
                        # different time ranges. 
                        Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=HSV_Spatial_splitting_begindates[imodel],
                                                                    enddates=HSV_Spatial_splitting_enddates[imodel], version=version,species=species,
                                                                    nchannel=len(main_stream_channel_names),special_name=description,ifold=0,width=width,height=height,depth=depth)
                        
                        validation_output = cnn_predict_3D(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                        mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                        training_output = cnn_predict_3D(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                        mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                    
                    if Apply_Transformer_architecture:
                        if world_size > 1:
                            mp.spawn(Transformer_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                  X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std, \
                                                Evaluation_type,typeName,HSV_Spatial_splitting_begindates[imodel],\
                                                HSV_Spatial_splitting_enddates[imodel],0),nprocs=world_size)
                        else:
                            Transformer_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                  X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std, \
                                                Evaluation_type,typeName,HSV_Spatial_splitting_begindates[imodel],\
                                                HSV_Spatial_splitting_enddates[imodel],0)
                        try:
                            channels_to_exclude = temp_sweep_config.get("channel_to_exclude", [])
                        except AttributeError:
                            channels_to_exclude = []
                        
                        excluded_total_channel_names, main_stream_channel_names, side_stream_channel_names = Get_channel_names(channels_to_exclude=channels_to_exclude)
                        index_of_main_stream_channels_of_initial = [total_channel_names.index(channel) for channel in main_stream_channel_names]
                        X_train = X_train[:,:,index_of_main_stream_channels_of_initial]
                        X_test  = X_test[:,:,index_of_main_stream_channels_of_initial]

                        # Since in hyperparameter searching we do not apply multiple tests, we only see the final testing accuracy, so no loop here in
                        # different time ranges.
                        Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=HSV_Spatial_splitting_begindates[imodel],
                                                                    enddates=HSV_Spatial_splitting_enddates[imodel], version=version,species=species,
                                                                    nchannel=len(main_stream_channel_names),special_name=description,ifold=0,d_model=d_model,
                                                                    n_head=n_head,ffn_hidden=ffn_hidden,
                                                                    num_layers=num_layers,max_len=max_len+spin_up_len)
                        validation_output = transformer_predict(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                        mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                        training_output = transformer_predict(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                        mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                        
                        validation_output = np.squeeze(validation_output)
                        training_output = np.squeeze(training_output)
                    
                    if Apply_CNN_Transformer_architecture:
                        if world_size > 1:
                            mp.spawn(CNN_Transformer_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,CNN_Embedding_channel_names,Transformer_Embedding_channel_names,
                                                  X_train_CNN, X_test_CNN,X_train_Transformer, X_test_Transformer,
                                                  y_train, y_test,Transformer_trainingdatasets_mean, Transformer_trainingdatasets_std, width,height,
                                                  Evaluation_type,typeName,HSV_Spatial_splitting_begindates[imodel],HSV_Spatial_splitting_enddates[imodel],0),nprocs=world_size)
                        else:
                            CNN_Transformer_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,CNN_Embedding_channel_names,Transformer_Embedding_channel_names,
                                                  X_train_CNN, X_test_CNN,X_train_Transformer, X_test_Transformer,
                                                  y_train, y_test,Transformer_trainingdatasets_mean, Transformer_trainingdatasets_std, width,height,
                                                  Evaluation_type,typeName,HSV_Spatial_splitting_begindates[imodel],HSV_Spatial_splitting_enddates[imodel],0)
                        
                        try:
                            CNN_channels_to_exclude = temp_sweep_config.get("CNN_channel_to_exclude", [])
                            Transformer_channel_to_exclude = temp_sweep_config.get("Transformer_channel_to_exclude", [])
                        except AttributeError:
                            CNN_channels_to_exclude = []
                            Transformer_channel_to_exclude = []
                        excluded_CNN_channel_names, main_stream_CNN_channel_names, side_stream_CNN_channel_names = Get_channel_names(channels_to_exclude=CNN_channels_to_exclude, init_channels=CNN_Embedding_channel_names)
                        excluded_Transformer_channel_names, main_stream_Transformer_channel_names, side_stream_Transformer_channel_names = Get_channel_names(channels_to_exclude=Transformer_channel_to_exclude, init_channels=Transformer_Embedding_channel_names)
                        index_of_main_stream_CNN_channels_of_initial = [CNN_Embedding_channel_names.index(channel) for channel in main_stream_CNN_channel_names]
                        index_of_main_stream_Transformer_channels_of_initial = [Transformer_Embedding_channel_names.index(channel) for channel in main_stream_Transformer_channel_names]
                        X_train_CNN = X_train_CNN[:,:,index_of_main_stream_CNN_channels_of_initial,:,:]
                        X_test_CNN  = X_test_CNN[:,:,index_of_main_stream_CNN_channels_of_initial,:,:]
                        X_train_Transformer = X_train_Transformer[:,:,index_of_main_stream_Transformer_channels_of_initial]
                        X_test_Transformer  = X_test_Transformer[:,:,index_of_main_stream_Transformer_channels_of_initial]

                        # Since in hyperparameter searching we do not apply multiple tests, we only see the final testing accuracy, so no loop here in
                        # different time ranges.
                        Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=HSV_Spatial_splitting_begindates[imodel],
                                                                    enddates=HSV_Spatial_splitting_enddates[imodel], version=version,species=species,
                                                                    nchannel=len(main_stream_CNN_channel_names)+len(main_stream_Transformer_channel_names),
                                                                    special_name=description,ifold=0,d_model=d_model,n_head=n_head,ffn_hidden=ffn_hidden,
                                                                    num_layers=num_layers,max_len=max_len+spin_up_len,width=width,height=height,CNN_nchannel=len(main_stream_CNN_channel_names),
                                                                    Transformer_nchannel=len(main_stream_Transformer_channel_names))
                        validation_output = cnn_transformer_predict(CNN_inputarray=X_test_CNN, Transformer_inputarray=X_test_Transformer, model=Daily_Model, batchsize=3000)
                        training_output = cnn_transformer_predict(CNN_inputarray=X_train_CNN, Transformer_inputarray=X_train_Transformer, model=Daily_Model, batchsize=3000)
                        validation_output = np.squeeze(validation_output)
                        training_output = np.squeeze(training_output)                                                        
                    del Daily_Model
                    gc.collect()
                    

                    # Get the final output for the validation datasets
                    final_output = Get_final_output(Validation_Prediction=validation_output, validation_geophysical_species=cctnd_geophysical_species_data[test_datasets_index],
                                                    bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,
                                                    log_species=False, mean=true_input_mean, std=true_input_std)
                    training_final_output = Get_final_output(Validation_Prediction=training_output, validation_geophysical_species=cctnd_geophysical_species_data[train_datasets_index],
                                                    bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,
                                                    log_species=False, mean=true_input_mean, std=true_input_std)
                    
                    print('final_output.shape: ', final_output.shape)
                    print('training_final_output.shape: ', training_final_output.shape)
                    print('cctnd_ground_observation_data[test_datasets_index].shape: ', cctnd_ground_observation_data[test_datasets_index].shape)
                    print('cctnd_geophysical_species_data[test_datasets_index].shape: ', cctnd_geophysical_species_data[test_datasets_index].shape)
                    print('sites_test.shape: ', sites_test.shape)
                    print('dates_test.shape: ', dates_test.shape)
                    
                    if Apply_Transformer_architecture or Apply_CNN_Transformer_architecture:
                        sites_test = np.tile(sites_test[:,np.newaxis], (1, max_len+spin_up_len)).flatten()
                        sites_train = np.tile(sites_train[:,np.newaxis], (1, max_len+spin_up_len)).flatten()
                    final_data_recording = np.concatenate((final_data_recording, final_output), axis=0)
                    obs_data_recording = np.concatenate((obs_data_recording, cctnd_ground_observation_data[test_datasets_index].flatten()), axis=0)
                    geo_data_recording = np.concatenate((geo_data_recording, cctnd_geophysical_species_data[test_datasets_index].flatten()), axis=0)
                    sites_recording = np.concatenate((sites_recording, sites_test), axis=0)
                    dates_recording = np.concatenate((dates_recording, dates_test.flatten()), axis=0)

                    training_final_data_recording = np.concatenate((training_final_data_recording, training_final_output), axis=0)
                    training_obs_data_recording = np.concatenate((training_obs_data_recording, cctnd_ground_observation_data[train_datasets_index].flatten()), axis=0)
                    training_sites_recording = np.concatenate((training_sites_recording, sites_train), axis=0)
                    training_dates_recording = np.concatenate((training_dates_recording, dates_train.flatten()), axis=0)


                                                                                                                                                                                                                                       
    Daily_statistics_recording, Monthly_statistics_recording, Annual_statistics_recording = calculate_statistics(test_begindates=HSV_Spatial_splitting_begindates[0],
                                                                                                                test_enddates=HSV_Spatial_splitting_enddates[-1],final_data_recording=final_data_recording,
                                                                                                                obs_data_recording=obs_data_recording,geo_data_recording=geo_data_recording,
                                                                                                                sites_recording=sites_recording,dates_recording=dates_recording,
                                                                                                                training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,
                                                                                                                training_sites_recording=training_sites_recording,
                                                                                                                training_dates_recording=training_dates_recording,
                                                                                                                Statistics_list=Statistics_list,)
    

   
    print('Start to save the validation results to csv file... for {}'.format(Model_structure_type))     
    
    if Apply_CNN_architecture:
        csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                          main_stream_channel_names=main_stream_channel_names,test_begindate=HSV_Spatial_splitting_begindates[0],
                                          test_enddate=HSV_Spatial_splitting_enddates[-1],
                                          width=width,height=height,entity=entity,project=project,sweep_id=sweep_id,name=name)
    elif Apply_3D_CNN_architecture:
        csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                          main_stream_channel_names=main_stream_channel_names,test_begindate=HSV_Spatial_splitting_begindates[0],
                                            test_enddate=HSV_Spatial_splitting_enddates[-1],
                                          width=width,height=height,depth=depth,entity=entity,project=project,sweep_id=sweep_id,name=name)
    elif Apply_Transformer_architecture:
        csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                          main_stream_channel_names=main_stream_channel_names,test_begindate=HSV_Spatial_splitting_begindates[0],
                                            test_enddate=HSV_Spatial_splitting_enddates[-1],
                                          d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, num_layers=num_layers, max_len=max_len+spin_up_len, entity=entity,project=project,sweep_id=sweep_id,name=name)
    
    elif Apply_CNN_Transformer_architecture:
        csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                          main_stream_channel_names=main_stream_CNN_channel_names,
                                          test_begindate=HSV_Spatial_splitting_begindates[0],
                                            test_enddate=HSV_Spatial_splitting_enddates[-1],
                                          d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, num_layers=num_layers, max_len=max_len+spin_up_len,
                                          width=width,height=height, entity=entity,project=project,sweep_id=sweep_id,name=name,CNN_nchannel=len(main_stream_CNN_channel_names),Transformer_nchannel=len(main_stream_Transformer_channel_names))
    print('csvfile_outfile: ', csvfile_outfile)
    output_csv(outfile=csvfile_outfile,status='w',Area='North America',
                test_begindate=HSV_Spatial_splitting_begindates[0],test_enddate=HSV_Spatial_splitting_enddates[-1],
                Daily_statistics_recording=Daily_statistics_recording,
                Monthly_statistics_recording=Monthly_statistics_recording,
                Annual_statistics_recording=Annual_statistics_recording,)
    
    print('Start to log the validation results to wandb... for {}'.format(Model_structure_type))
    if not Use_recorded_data_to_show_validation_results:
        run_id = run_id_container.get("run_id", None)
        run_name = run_id_container.get("run_name", None)
        print('run_id: ', run_id)
        print('run_name: ', run_name)
        manager.shutdown()  # Shutdown the manager to release resources
        
        os.environ["WANDB_DEBUG"] = "true"
        
        wandb.init( entity="ACAG-NorthAmericaDailyPM25",
                id=run_id,
                    name=run_name,
                    # Set the wandb project where this run will be logged.
                    project=version,
                    # Track hyperparameters and run metadata.
                group=sweep_id if sweep_mode else None,
                mode='online',
                resume="allow"
                )  # <-- Prevent hangs on init)

        print("Wandb init succeeded:", wandb.run.id)


        wandb.log({'test_R2': Daily_statistics_recording['All_points']['test_R2'],
                        'train_R2': Daily_statistics_recording['All_points']['train_R2'],
                        'geo_R2': Daily_statistics_recording['All_points']['geo_R2'],
                        'RMSE': Daily_statistics_recording['All_points']['RMSE'],
                        'NRMSE': Daily_statistics_recording['All_points']['NRMSE'],
                        'slope': Daily_statistics_recording['All_points']['slope'],
                        })
        print('logged information to wandb: ','\n'.join(['test_R2: {}'.format(Daily_statistics_recording['All_points']['test_R2']),
                                                        'train_R2: {}'.format(Daily_statistics_recording['All_points']['train_R2']),
                                                        'geo_R2: {}'.format(Daily_statistics_recording['All_points']['geo_R2']),
                                                        'RMSE: {}'.format(Daily_statistics_recording['All_points']['RMSE']),
                                                        'NRMSE: {}'.format(Daily_statistics_recording['All_points']['NRMSE']),
                                                        'slope: {}'.format(Daily_statistics_recording['All_points']['slope'])]))
        
        wandb.finish()

        del final_data_recording, obs_data_recording, geo_data_recording, sites_recording, dates_recording
        del training_final_data_recording, training_obs_data_recording, training_sites_recording, training_dates_recording
        del Daily_statistics_recording, Monthly_statistics_recording, Annual_statistics_recording
        gc.collect()
            
    return