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
from Evaluation_pkg.data_func import Split_Datasets_based_site_index,randomly_select_training_testing_indices,Get_final_output,Split_Datasets_randomly
from Evaluation_pkg.iostream import *
from Evaluation_pkg.Statistics_Calculation_func import calculate_statistics
from Model_Structure_pkg.CNN_Module import initial_cnn_network
from Model_Structure_pkg.ResCNN3D_Module import initial_3dcnn_net
from Model_Structure_pkg.utils import *

from Training_pkg.utils import *
from Training_pkg.utils import epoch as config_epoch, batchsize as config_batchsize, learning_rate0 as config_learning_rate0
from Training_pkg.TensorData_func import Dataset_Val, Dataset
from Training_pkg.TrainingModule import CNN_train, cnn_predict, CNN3D_train, cnn_predict_3D,Transformer_train,transformer_predict,CNN_Transformer_train,cnn_transformer_predict
from Training_pkg.data_func import CNNInputDatasets, CNN3DInputDatasets,TransformerInputDatasets, CNN_Transformer_InputDatasets
from Training_pkg.iostream import load_daily_datesbased_model

from Visualization_pkg.Assemble_Func import plot_longterm_Annual_Monthly_Daily_Scatter_plots,plot_timeseries_statistics_plots
from wandb_config import wandb_run_config, wandb_initialize, init_get_sweep_config
from multiprocessing import Manager


def random_crossvalidation(total_channel_names, main_stream_channel_names,
                             side_stream_channel_names,sweep_id=None,):
    world_size = torch.cuda.device_count()
    print(f"Number of available GPUs: {world_size}")
    typeName = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=False, species=species)

    #####################################################################
    # Start the hyperparameters search validation
    Statistics_list = ['test_R2','train_R2','geo_R2','RMSE','NRMSE','slope','PWA']
    seed       = 19980130
    rkf = RepeatedKFold(n_splits=Random_CV_folds, n_repeats=1, random_state=seed)
    manager = Manager()
    run_id_container = manager.dict() 
    
    #### Initialize the wandb for sweep mode
    if Random_CV_Apply_wandb_sweep_Switch:
        sweep_mode = True
        temp_sweep_config = init_get_sweep_config()
        entity = temp_sweep_config.get("entity", "ACAG-NorthAmericaDailyPM25")
        project = temp_sweep_config.get("project", version)
        name = temp_sweep_config.get("name", None)
        if Apply_Transformer_architecture:
            d_model, n_head, ffn_hidden, num_layers, max_len,spin_up_len = temp_sweep_config.get("d_model", 64), temp_sweep_config.get("n_head", 8), temp_sweep_config.get("ffn_hidden", 256), temp_sweep_config.get("num_layers", 6), temp_sweep_config.get("max_len", 1000), temp_sweep_config.get("spin_up_len", 100)
    else:
        sweep_mode = False
        temp_sweep_config = None
        entity = None
        project = None
        name = None
        if Apply_Transformer_architecture:
            d_model, n_head, ffn_hidden, num_layers, max_len,spin_up_len = Transformer_d_model, Transformer_n_head, Transformer_ffn_hidden, Transformer_num_layers, Transformer_max_len, Transformer_spin_up_len

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
    
    if Apply_CNN_architecture:
        args = {'width': width, 'height': height}
    elif Apply_3D_CNN_architecture:
        args = {'width': width, 'height': height, 'depth': depth}
    elif Apply_Transformer_architecture:
        args = {'d_model': d_model, 'n_head': n_head, 'ffn_hidden': ffn_hidden, 'num_layers': num_layers, 'max_len': max_len+spin_up_len}
    elif Apply_CNN_Transformer_architecture:
        args = {'d_model': d_model, 'n_head': n_head, 'ffn_hidden': ffn_hidden, 'num_layers': num_layers, 'max_len': max_len+spin_up_len,
                'width': width, 'height': height, 'CNN_nchannel': len(CNN_Embedding_channel_names), 'Transformer_nchannel': len(Transformer_Embedding_channel_names)}

    ### Start Training, Validation, and Recording
    if not Use_recorded_data_to_show_validation_results_Random_CV:

            ### Initialize the arrays for recording
            final_data_recording = np.array([],dtype=float)
            obs_data_recording = np.array([],dtype=float)
            geo_data_recording = np.array([],dtype=float)
            sites_recording = np.array([],dtype=int)
            dates_recording = np.array([],dtype=int)

            training_final_data_recording = np.array([],dtype=float)
            training_obs_data_recording = np.array([],dtype=float)
            training_sites_recording = np.array([],dtype=int)
            training_dates_recording = np.array([],dtype=int)

            ### Start the training process in each specified time range
            for imodel in range(len(Random_CV_training_begindates)):

                ### Get the training and targets in desired range
                if Apply_CNN_architecture or Apply_3D_CNN_architecture:
                    # Get the initial true_input and training datasets for the current model (within the desired time range)
                    print('1...',' Start Date: ', Random_CV_training_begindates[imodel], ' End Date: ', Random_CV_training_enddates[imodel])
                    desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=Random_CV_training_begindates[imodel],
                                                                                                    end_date=Random_CV_training_enddates[imodel]) # initial datasets
                    # Normalize the training datasets
                    print('2...', 'Desired Training Datasets: ', desired_trainingdatasets.keys())
                    normalized_TrainingDatasets  = Init_CNN_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
                    del desired_trainingdatasets
                    gc.collect()
                    # Concatenate the training datasets and true input for the current model for training and tetsing purposes
                    
                elif Apply_Transformer_architecture:
                    # Get the initial true_input and training datasets for the current model (within the desired time range)
                    print('1...',' Start Date: ', Random_CV_training_begindates[imodel], ' End Date: ', Random_CV_training_enddates[imodel])
                    desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_Transformer_Datasets.get_desired_range_inputdatasets(start_date=Random_CV_training_begindates[imodel],
                                                                                                    end_date=Random_CV_training_enddates[imodel],max_len=max_len,spinup_len=spin_up_len)
                    # Normalize the training datasets
                    print('2...', 'Desired Training Datasets: ', desired_trainingdatasets.keys())
                    normalized_TrainingDatasets  = Init_Transformer_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
                    del desired_trainingdatasets
                    gc.collect()
                    
                elif Apply_CNN_Transformer_architecture:
                    # Get the initial true_input and training datasets for the current model (within the desired time range)
                    print('1...')
                    desired_CNN_trainingdatasets, desired_Transformer_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=Random_CV_training_begindates[imodel],
                                                                                                    end_date=Random_CV_training_enddates[imodel],max_len=max_len,spinup_len=spin_up_len)
                    # Normalize the training datasets
                    print('2...')
                    normalized_CNN_TrainingDatasets,normalized_Transformer_TrainingDatasets  = Init_CNN_Datasets.normalize_trainingdatasets(desired_CNN_trainingdatasets=desired_CNN_trainingdatasets, desired_Transformer_trainingdatasets=desired_Transformer_trainingdatasets)
                    del desired_CNN_trainingdatasets, desired_Transformer_trainingdatasets
                    gc.collect()
                                        
            ### Concatenate the datasets
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
                if Apply_CNN_Transformer_architecture:
                    print('3...')
                    cctnd_CNN_trainingdatasets, cctnd_Transformer_trainingdatasets, cctnd_true_input,cctnd_ground_observation_data,cctnd_geophysical_species_data, cctnd_sites_index, cctnd_dates = Init_CNN_Datasets.concatenate_trainingdatasets(desired_true_input=desired_true_input, 
                                                                                                                                        desired_normalized_CNN_trainingdatasets=normalized_CNN_TrainingDatasets,
                                                                                                                                        desired_normalized_Transformer_trainingdatasets=normalized_Transformer_TrainingDatasets,
                                                                                                                                        desired_ground_observation_data=desired_ground_observation_data,
                                                                                                                                        desired_geophysical_species_data=desired_geophysical_species_data)
                
                total_data_points = len(cctnd_true_input)
                print('Total data points for training and testing: ', total_data_points)
                ### Start folds loop
                for ifold, (training_selected_indices,testing_selected_indices) in enumerate(rkf.split(total_data_points)):
                    print('training_selected_sites: ',training_selected_indices.shape)
                    print('testing_selected_sites: ',testing_selected_indices.shape)

                    print('4...')
                    ### Split the datesets based on the indices of training and testing indices
                    if Apply_3D_CNN_architecture or Apply_CNN_architecture or Apply_Transformer_architecture:
                        X_train, y_train, X_test, y_test, dates_train, dates_test, sites_train, sites_test, = Split_Datasets_randomly(train_index=training_selected_indices,
                                                                                                                                        test_index=testing_selected_indices,
                                                                                                                                        total_trainingdatasets=cctnd_trainingdatasets,
                                                                                                                                        total_true_input=cctnd_true_input,
                                                                                                                                        total_sites_index=cctnd_sites_index,
                                                                                                                                        total_dates=cctnd_dates)
                    elif Apply_CNN_Transformer_architecture:
                        X_train_CNN, y_train, X_test_CNN, y_test, dates_train, dates_test, sites_train, sites_test = Split_Datasets_randomly(train_index=training_selected_indices,
                                                                                                                                    test_index=testing_selected_indices,
                                                                                                                                    total_trainingdatasets=cctnd_CNN_trainingdatasets,
                                                                                                                                    total_true_input=cctnd_true_input,
                                                                                                                                    total_sites_index=cctnd_sites_index,
                                                                                                                                    total_dates=cctnd_dates)
                        X_train_Transformer, y_train, X_test_Transformer, y_test, dates_train, dates_test, sites_train, sites_test = Split_Datasets_randomly(train_index=training_selected_indices,
                                                                                                                                    test_index=testing_selected_indices,
                                                                                                                                    total_trainingdatasets=cctnd_Transformer_trainingdatasets,
                                                                                                                                    total_true_input=cctnd_true_input,
                                                                                                                                    total_sites_index=cctnd_sites_index,
                                                                                                                                    total_dates=cctnd_dates)
    return
