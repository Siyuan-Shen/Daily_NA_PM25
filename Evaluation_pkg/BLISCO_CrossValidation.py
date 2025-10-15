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
from Training_pkg.TrainingModule import CNN_train, cnn_predict, CNN3D_train, cnn_predict_3D,Transformer_train,transformer_predict,CNN_Transformer_train,cnn_transformer_predict
from Training_pkg.data_func import CNNInputDatasets, CNN3DInputDatasets,TransformerInputDatasets, CNN_Transformer_InputDatasets
from Training_pkg.iostream import load_daily_datesbased_model

from Visualization_pkg.Assemble_Func import plot_longterm_Annual_Monthly_Daily_Scatter_plots,plot_timeseries_statistics_plots
from wandb_config import wandb_run_config, wandb_initialize, init_get_sweep_config
from multiprocessing import Manager


def BLISCO_cross_validation(buffer_radius,total_channel_names, main_stream_channel_names,
                            side_stream_channel_names, sweep_id=None):
    world_size = torch.cuda.device_count()
    print(f"Number of available GPUs: {world_size}")
    typeName = Get_typeName(bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species, log_species=False, species=species)
    Evaluation_type = 'BLISCO_CrossValidation_{}-km_{}-folds_{}-seeds'.format(buffer_radius,BLISCO_CV_folds,BLISCO_CV_seeds_number)

    #####################################################################
    
    #####################################################################
    # Start the hyperparameters search validation
    Statistics_list = ['test_R2','train_R2','geo_R2','RMSE','NRMSE','slope','PWA']
    seed       = 19980130
    
    manager = Manager()
    run_id_container = manager.dict() 
    
    #### Initialize the wandb for sweep mode, no sweep mode for BLISCO CV ####
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
        
    rkf = RepeatedKFold(n_splits=BLISCO_CV_folds, n_repeats=1, random_state=seed)
    
    
    if Apply_CNN_architecture:
        args = {'width': width, 'height': height}
    elif Apply_3D_CNN_architecture:
        args = {'width': width, 'height': height, 'depth': depth}
    elif Apply_Transformer_architecture:
        args = {'d_model': d_model, 'n_head': n_head, 'ffn_hidden': ffn_hidden, 'num_layers': num_layers, 'max_len': max_len+spin_up_len}
    elif Apply_CNN_Transformer_architecture:
        args = {'d_model': d_model, 'n_head': n_head, 'ffn_hidden': ffn_hidden, 'num_layers': num_layers, 'max_len': max_len+spin_up_len,
                'width': width, 'height': height, 'CNN_nchannel': len(CNN_Embedding_channel_names), 'Transformer_nchannel': len(Transformer_Embedding_channel_names)}

    if not Use_recorded_data_to_show_validation_results_BLISCO_CV:
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
        for imodel in range(len(BLISCO_CV_training_begindates)):
            
            ### Get the training and targets in desired range
            if Apply_CNN_architecture or Apply_3D_CNN_architecture:
                # Get the initial true_input and training datasets for the current model (within the desired time range)
                print('1...',' Start Date: ', BLISCO_CV_training_begindates[imodel], ' End Date: ', BLISCO_CV_training_enddates[imodel])
                desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=BLISCO_CV_training_begindates[imodel],
                                                                                                end_date=BLISCO_CV_training_enddates[imodel]) # initial datasets
                # Normalize the training datasets
                print('2...', 'Desired Training Datasets: ', desired_trainingdatasets.keys())
                normalized_TrainingDatasets  = Init_CNN_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
                del desired_trainingdatasets
                gc.collect()
                # Concatenate the training datasets and true input for the current model for training and tetsing purposes
                
            elif Apply_Transformer_architecture:
                # Get the initial true_input and training datasets for the current model (within the desired time range)
                print('1...',' Start Date: ', BLISCO_CV_training_begindates[imodel], ' End Date: ', BLISCO_CV_training_enddates[imodel])
                desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_Transformer_Datasets.get_desired_range_inputdatasets(start_date=BLISCO_CV_training_begindates[imodel],
                                                                                                end_date=BLISCO_CV_training_enddates[imodel],max_len=max_len,spinup_len=spin_up_len)
                # Normalize the training datasets
                print('2...', 'Desired Training Datasets: ', desired_trainingdatasets.keys())
                normalized_TrainingDatasets  = Init_Transformer_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
                del desired_trainingdatasets
                gc.collect()
                
            elif Apply_CNN_Transformer_architecture:
                # Get the initial true_input and training datasets for the current model (within the desired time range)
                print('1...')
                desired_CNN_trainingdatasets, desired_Transformer_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=BLISCO_CV_training_begindates[imodel],
                                                                                                end_date=BLISCO_CV_training_enddates[imodel],max_len=max_len,spinup_len=spin_up_len)
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
            
            ###########################################################################################################################
            ### Get the training and testing indices for each fold based on the BLISCO method
            
            ### First get the sites of each site, and find the sites that are not all nan during this range
            ### Then derive the training and testing indices for each fold based on the BLISCO method
            init_sites_index = np.arange(total_sites_number)
            for isite in range(total_sites_number):
                temp_site_indices = np.where(cctnd_sites_index == isite)[0]
                if len(temp_site_indices) == 0:
                    init_sites_index[isite] = -1
                else:
                    if len(np.where(~np.isnan(cctnd_ground_observation_data[np.where(cctnd_sites_index == isite)]))[0]) == 0:
                        init_sites_index[isite] = -1
            valid_sites_index = np.where(init_sites_index != -1)[0]
            indices_of_valid_index = np.arange(len(valid_sites_index))
            print('Total sites: {}, valid sites: {}'.format(total_sites_number,len(valid_sites_index)))
            if len(valid_sites_index) < BLISCO_CV_folds:
                raise ValueError('The number of valid sites is smaller than the number of folds, please decrease the number of folds or the observation_datapoints_threshold!')
            valid_site_lat = sites_lat[valid_sites_index]
            valid_site_lon = sites_lon[valid_sites_index]
            index_for_BLISCO = np.zeros((BLISCO_CV_folds,len(valid_site_lat),len(BLISCO_CV_training_begindates)),dtype=np.int32)
            
            nearest_distances = np.array([],dtype=np.float32)
            for isite in range(len(valid_site_lat)):
                ## get the nearest distance for each site to the rest of the sites.
                valid_site_distances = calculate_distance_forArray(site_lat=valid_site_lat[isite],site_lon=valid_site_lon
                                                                [isite],SATLAT_MAP=valid_site_lat,SATLON_MAP=valid_site_lon)
                nearest_distances = np.append(nearest_distances,np.min(valid_site_distances[np.where(valid_site_distances>0.01)]))
            print('The average nearest distance between sites is: ', np.mean(nearest_distances))
            ## Get the index for Self-Isolated sites (have distances larger than buffer radius naturally) 
            ## and sites for BLeCO.
            Self_Isolated_sites_index = np.where(nearest_distances>=buffer_radius)[0]
            Sites_forBLeCO_index      = np.where(nearest_distances<buffer_radius)[0]
            self_isolated_fold_count = 0
            length_of_Self_Isolated_sites_index = len(Self_Isolated_sites_index)

            ## If there are Self-Isolated sites, then we need to split the Self-Isolated sites into different folds.
            if len(Self_Isolated_sites_index) > 0:
                if len(Self_Isolated_sites_index) < BLISCO_CV_folds:
                    ## Here we append -999 to the Self_Isolated_sites_index to make the length of Self_Isolated_sites_index
                    ## equal to BLISCO_CV_folds.
                    for i in range(BLISCO_CV_folds - length_of_Self_Isolated_sites_index):
                        Self_Isolated_sites_index = np.append(Self_Isolated_sites_index,-999)
                    for train_index, test_index in rkf.split(Self_Isolated_sites_index):
                        ## If the test_index is -999, then we set all self-Isolated sites for training index for this fold.
                        if Self_Isolated_sites_index[test_index] == -999:
                            temp_train_index = np.where(Self_Isolated_sites_index!=-999)
                            print(test_index,temp_train_index,Self_Isolated_sites_index[temp_train_index])
                            index_for_BLISCO[self_isolated_fold_count,indices_of_valid_index[Self_Isolated_sites_index[temp_train_index]],imodel] = -1.0
                            self_isolated_fold_count += 1
                        
                        ## If this site is selected as the test site, then we set the index to 1.0. And we set the
                        ## rest of the sites (not filled sites, e.g., not equal to -999.0) as the training sites.
                        else:
                            temp_train_index = np.where(Self_Isolated_sites_index[train_index]!=-999)
                            print(test_index,train_index[temp_train_index],Self_Isolated_sites_index[train_index[temp_train_index]])
                            Self_Isolated_sites_index = Self_Isolated_sites_index.astype(int)
                            index_for_BLISCO[self_isolated_fold_count,indices_of_valid_index[Self_Isolated_sites_index[test_index]],imodel]  = 1.0
                            index_for_BLISCO[self_isolated_fold_count,indices_of_valid_index[Self_Isolated_sites_index[train_index[temp_train_index]]],imodel] = -1.0
                            self_isolated_fold_count += 1
                ## If the number of Self-Isolated sites is larger than the number of BLCO_kfold, then we can split the
                    ## Self-Isolated sites into different folds directly, just like normal sppatial cross-validation.
                else:
                     
                    for train_index, test_index in rkf.split(Self_Isolated_sites_index):
                        Self_Isolated_sites_index = Self_Isolated_sites_index.astype(int)
                        index_for_BLISCO[self_isolated_fold_count,indices_of_valid_index[Self_Isolated_sites_index[test_index]],imodel]  = 1.0
                        index_for_BLISCO[self_isolated_fold_count,indices_of_valid_index[Self_Isolated_sites_index[train_index]],imodel] = -1.0
                        self_isolated_fold_count += 1
                
            if len(Sites_forBLeCO_index) > 0:
                    ## If there are sites for BLeCO, then we need to split the sites for BLeCO into different folds using
                    ## the function derive_Test_Training_index_4Each_BLCO_fold.
                    Only_BLeCO_index = derive_Test_Training_index_4Each_BLISCO_fold(kfolds=BLISCO_CV_folds,number_of_SeedClusters=BLISCO_CV_seeds_number,site_lat=valid_site_lat[Sites_forBLeCO_index],site_lon=valid_site_lon[Sites_forBLeCO_index],
                                                                        BLISCO_Buffer_Size=buffer_radius)
                    for ifold in range(BLISCO_CV_folds):
                        
                        index_for_BLISCO[ifold,indices_of_valid_index[Sites_forBLeCO_index[np.where(Only_BLeCO_index[ifold,:]==1.0)]],imodel] = 1.0
                        index_for_BLISCO[ifold,indices_of_valid_index[Sites_forBLeCO_index[np.where(Only_BLeCO_index[ifold,:]==-1.0)]],imodel] = -1.0
            ###########################################################################################################################
            
            for ifold in range(BLISCO_CV_folds):
                
                #### The test and training indices for this fold should be derived based on the initial sites index rather than the valid sites index.
                #### This is because concataned datasets are based on the initial sites index.
                
                test_index = valid_sites_index[np.where(index_for_BLISCO[ifold,:,imodel] == 1.0)[0]]
                train_index = valid_sites_index[np.where(index_for_BLISCO[ifold,:,imodel] == -1.0)[0]]
                excluded_index = valid_sites_index[np.where(index_for_BLISCO[ifold,:,imodel] == 0.0)[0]]
                print('Buffer Size: {} km,No.{}-fold, test_index #: {}, train_index #: {}, total # of sites: {}'.format(buffer_radius,ifold+1,len(test_index),len(train_index),len(valid_site_lat)))
                print('4...')
                ### Split the datesets based on the indices of training and testing indices
                if Apply_3D_CNN_architecture or Apply_CNN_architecture or Apply_Transformer_architecture:
                    X_train, y_train, X_test, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datasets_based_site_index(train_site_index=train_index,
                                                                                                                                    test_site_index=test_index,
                                                                                                                                    total_trainingdatasets=cctnd_trainingdatasets,
                                                                                                                                    total_true_input=cctnd_true_input,
                                                                                                                                    total_sites_index=cctnd_sites_index,
                                                                                                                                    total_dates=cctnd_dates)
                elif Apply_CNN_Transformer_architecture:
                    X_train_CNN, y_train, X_test_CNN, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datasets_based_site_index(train_site_index=train_index,
                                                                                                                                test_site_index=test_index,
                                                                                                                                total_trainingdatasets=cctnd_CNN_trainingdatasets,
                                                                                                                                total_true_input=cctnd_true_input,
                                                                                                                                total_sites_index=cctnd_sites_index,
                                                                                                                                total_dates=cctnd_dates)
                    X_train_Transformer, y_train, X_test_Transformer, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datasets_based_site_index(train_site_index=train_index,
                                                                                                                                test_site_index=test_index,
                                                                                                                                total_trainingdatasets=cctnd_Transformer_trainingdatasets,
                                                                                                                                total_true_input=cctnd_true_input,
                                                                                                                                total_sites_index=cctnd_sites_index,
                                                                                                                                total_dates=cctnd_dates)
                ## Start Training
                ## 2D CNN Training
                if Apply_CNN_architecture:
                    if world_size > 1:
                        mp.spawn(CNN_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                            X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                        Evaluation_type,typeName,BLISCO_CV_training_begindates[imodel],\
                                        BLISCO_CV_training_enddates[imodel],ifold),nprocs=world_size)
                    else:
                        CNN_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                            X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                        Evaluation_type,typeName,BLISCO_CV_training_begindates[imodel],\
                                        BLISCO_CV_training_enddates[imodel],ifold)

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
                    Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=BLISCO_CV_training_begindates[imodel],
                                                                enddates=BLISCO_CV_training_enddates[imodel], version=version,species=species,
                                                                nchannel=len(main_stream_channel_names),special_name=description,ifold=ifold,width=width,height=height)
                    validation_output = cnn_predict(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                    mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                    training_output = cnn_predict(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                    mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)


                # 3D CNN Training
                elif Apply_3D_CNN_architecture:

                    if world_size > 1:
                        mp.spawn(CNN3D_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                            X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                            Evaluation_type,typeName,BLISCO_CV_training_begindates[imodel],\
                                            BLISCO_CV_training_enddates[imodel],ifold),nprocs=world_size)
                    else:
                        CNN3D_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                            X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                            Evaluation_type,typeName,BLISCO_CV_training_begindates[imodel],\
                                            BLISCO_CV_training_enddates[imodel],ifold)
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
                    Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=BLISCO_CV_training_begindates[imodel],
                                                                enddates=BLISCO_CV_training_enddates[imodel], version=version,species=species,
                                                                nchannel=len(main_stream_channel_names),special_name=description,ifold=ifold,width=width,height=height,depth=depth)
                    
                    validation_output = cnn_predict_3D(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                    mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                    training_output = cnn_predict_3D(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                        mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                # Transformer Training
                elif Apply_Transformer_architecture:
                    if world_size > 1:
                        mp.spawn(Transformer_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                            X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std, \
                                            Evaluation_type,typeName,BLISCO_CV_training_begindates[imodel],\
                                            BLISCO_CV_training_enddates[imodel],ifold),nprocs=world_size)
                    else:
                        Transformer_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                            X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std, \
                                            Evaluation_type,typeName,BLISCO_CV_training_begindates[imodel],\
                                            BLISCO_CV_training_enddates[imodel],ifold)
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
                    Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=BLISCO_CV_training_begindates[imodel],
                                                                enddates=BLISCO_CV_training_enddates[imodel], version=version,species=species,
                                                                nchannel=len(main_stream_channel_names),special_name=description,ifold=ifold,d_model=d_model,
                                                                n_head=n_head,ffn_hidden=ffn_hidden,
                                                                num_layers=num_layers,max_len=max_len+spin_up_len)
                    validation_output = transformer_predict(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                    mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                    training_output = transformer_predict(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                    mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                    
                    validation_output = np.squeeze(validation_output)
                    training_output = np.squeeze(training_output)
                elif Apply_CNN_Transformer_architecture:
                    if world_size > 1:
                        mp.spawn(CNN_Transformer_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,CNN_Embedding_channel_names,Transformer_Embedding_channel_names,
                                            X_train_CNN, X_test_CNN,X_train_Transformer, X_test_Transformer,
                                            y_train, y_test,Transformer_trainingdatasets_mean, Transformer_trainingdatasets_std, width,height,
                                            Evaluation_type,typeName,BLISCO_CV_training_begindates[imodel],BLISCO_CV_training_enddates[imodel],ifold),nprocs=world_size)
                    else:
                        CNN_Transformer_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,CNN_Embedding_channel_names,Transformer_Embedding_channel_names,
                                            X_train_CNN, X_test_CNN,X_train_Transformer, X_test_Transformer,
                                            y_train, y_test,Transformer_trainingdatasets_mean, Transformer_trainingdatasets_std, width,height,
                                            Evaluation_type,typeName,BLISCO_CV_training_begindates[imodel],BLISCO_CV_training_enddates[imodel],ifold)
                    
                    try:
                        CNN_channels_to_exclude = temp_sweep_config.get("CNN_channel_to_exclude", [])
                        Transformer_channel_to_exclude = temp_sweep_config.get("Transformer_channel_to_exclude", [])
                    except AttributeError:
                        CNN_channels_to_exclude = []
                        Transformer_channel_to_exclude = []
                    excluded_CNN_channel_names, main_stream_CNN_channel_names, side_stream_CNN_channel_names = Get_channel_names(channels_to_exclude=CNN_channels_to_exclude, initial_channel_names=CNN_Embedding_channel_names)
                    excluded_Transformer_channel_names, main_stream_Transformer_channel_names, side_stream_Transformer_channel_names = Get_channel_names(channels_to_exclude=Transformer_channel_to_exclude, initial_channel_names=Transformer_Embedding_channel_names)
                    index_of_main_stream_CNN_channels_of_initial = [CNN_Embedding_channel_names.index(channel) for channel in main_stream_CNN_channel_names]
                    index_of_main_stream_Transformer_channels_of_initial = [Transformer_Embedding_channel_names.index(channel) for channel in main_stream_Transformer_channel_names]
                    X_train_CNN = X_train_CNN[:,:,index_of_main_stream_CNN_channels_of_initial,:,:]
                    X_test_CNN  = X_test_CNN[:,:,index_of_main_stream_CNN_channels_of_initial,:,:]
                    X_train_Transformer = X_train_Transformer[:,:,index_of_main_stream_Transformer_channels_of_initial]
                    X_test_Transformer  = X_test_Transformer[:,:,index_of_main_stream_Transformer_channels_of_initial]

                    # Since in hyperparameter searching we do not apply multiple tests, we only see the final testing accuracy, so no loop here in
                    # different time ranges.
                    Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=BLISCO_CV_training_begindates[imodel],
                                                                enddates=BLISCO_CV_training_enddates[imodel], version=version,species=species,
                                                                nchannel=len(main_stream_CNN_channel_names)+len(main_stream_Transformer_channel_names),
                                                                special_name=description,ifold=0,d_model=d_model,n_head=n_head,ffn_hidden=ffn_hidden,
                                                                num_layers=num_layers,max_len=max_len+spin_up_len,width=width,height=height,CNN_nchannel=len(main_stream_CNN_channel_names),
                                                                Transformer_nchannel=len(main_stream_Transformer_channel_names))
                    validation_output = cnn_transformer_predict(CNN_inputarray=X_test_CNN, Transformer_inputarray=X_test_Transformer, model=Daily_Model, batchsize=3000)
                    training_output = cnn_transformer_predict(CNN_inputarray=X_train_CNN, Transformer_inputarray=X_train_Transformer, model=Daily_Model, batchsize=3000)
                    validation_output = np.squeeze(validation_output)
                    training_output = np.squeeze(training_output) 
            
                del Daily_Model, X_train, y_train, X_test, y_test
                gc.collect()
                # Get the final output for the validation datasets
                final_output = Get_final_output(Validation_Prediction=validation_output, validation_geophysical_species=cctnd_geophysical_species_data[test_datasets_index],
                                                bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,
                                                log_species=False, mean=true_input_mean, std=true_input_std)
                training_final_output = Get_final_output(Validation_Prediction=training_output, validation_geophysical_species=cctnd_geophysical_species_data[train_datasets_index],
                                                bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,
                                                log_species=False, mean=true_input_mean, std=true_input_std)
                # Calculate the statistics for the validation datasets


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
        
        save_data_recording(final_data_recording=final_data_recording, obs_data_recording=obs_data_recording, geo_data_recording=geo_data_recording,
                                sites_recording=sites_recording, dates_recording=dates_recording,
                                training_final_data_recording=training_final_data_recording, training_obs_data_recording=training_obs_data_recording,
                                training_sites_recording=training_sites_recording, training_dates_recording=training_dates_recording,
                                sites_lat_array=sites_lat, sites_lon_array=sites_lon,
                                species=species,version=version,begindates=BLISCO_CV_training_begindates[0],
                                enddates=BLISCO_CV_training_enddates[-1],typeName=typeName,nchannel=len(main_stream_channel_names),
                                evaluation_type=Evaluation_type,project=project,entity=entity,sweep_id=sweep_id,name=name,**args)
    
    
    final_data_recording, obs_data_recording, geo_data_recording, sites_recording, dates_recording, training_final_data_recording, training_obs_data_recording, training_sites_recording, training_dates_recording, sites_lat_array, sites_lon_array = load_data_recording(species=species,version=version,begindates=BLISCO_CV_training_begindates[0],
                                                                                                                                                                                                                                         enddates=BLISCO_CV_training_enddates[-1],typeName=typeName,nchannel=len(main_stream_channel_names),
                                                                                                                                                                                                                                         evaluation_type=Evaluation_type,special_name=description,project=project,entity=entity,sweep_id=sweep_id,**args)
    
    ### Calculate statistics and record them to the whole time range
    Daily_statistics_recording, Monthly_statistics_recording, Annual_statistics_recording = calculate_statistics(test_begindates=BLISCO_CV_validation_begindates[0],
                                                                                                                test_enddates=BLISCO_CV_validation_enddates[-1],final_data_recording=final_data_recording,
                                                                                                                obs_data_recording=obs_data_recording,geo_data_recording=geo_data_recording,
                                                                                                                sites_recording=sites_recording,dates_recording=dates_recording,
                                                                                                                training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,
                                                                                                                training_sites_recording=training_sites_recording,
                                                                                                                training_dates_recording=training_dates_recording,
                                                                                                                Statistics_list=Statistics_list,)

    csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                          main_stream_channel_names=main_stream_channel_names,test_begindate=BLISCO_CV_validation_begindates[0],
                                          test_enddate=BLISCO_CV_validation_enddates[-1],project=project,entity=entity,sweep_id=sweep_id,
                                          **args,)
    output_csv(outfile=csvfile_outfile,status='w',Area='North America',
                test_begindate=BLISCO_CV_validation_begindates[0],test_enddate=BLISCO_CV_validation_enddates[-1],
                Daily_statistics_recording=Daily_statistics_recording,
                Monthly_statistics_recording=Monthly_statistics_recording,
                Annual_statistics_recording=Annual_statistics_recording,)    
       
    ####   Calculate the statistics and recording to each time period that is interested
    for idate in range(len(BLISCO_CV_validation_addtional_regions)):
        test_begindate =  BLISCO_CV_validation_begindates[idate]
        test_enddate = BLISCO_CV_validation_enddates[idate]          
        Daily_statistics_recording, Monthly_statistics_recording, Annual_statistics_recording = calculate_statistics(test_begindates=test_begindate,
                                                                                                                test_enddates=test_enddate,final_data_recording=final_data_recording,
                                                                                                                obs_data_recording=obs_data_recording,geo_data_recording=geo_data_recording,
                                                                                                                sites_recording=sites_recording,dates_recording=dates_recording,
                                                                                                                training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,
                                                                                                                training_sites_recording=training_sites_recording,
                                                                                                                training_dates_recording=training_dates_recording,
                                                                                                                Statistics_list=Statistics_list,)
        print('Start to save the validation results to csv file... for {}'.format(Model_structure_type))     
        ## Output the statistics to csv files
        if Apply_CNN_architecture:
            csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                            main_stream_channel_names=main_stream_channel_names,test_begindate=test_begindate,test_enddate=test_enddate,
                                            entity=entity,project=project,sweep_id=sweep_id,name=name,**args)
        elif Apply_3D_CNN_architecture:
            csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                            main_stream_channel_names=main_stream_channel_names,test_begindate=test_begindate,test_enddate=test_enddate,
                                            entity=entity,project=project,sweep_id=sweep_id,name=name,**args)
        elif Apply_Transformer_architecture:
            csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                            main_stream_channel_names=main_stream_channel_names,test_begindate=test_begindate,test_enddate=test_enddate,
                                            entity=entity,project=project,sweep_id=sweep_id,name=name,**args)
        elif Apply_CNN_Transformer_architecture:
            csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                            main_stream_channel_names=main_stream_channel_names,test_begindate=test_begindate,test_enddate=test_enddate,
                                            entity=entity,project=project,sweep_id=sweep_id,name=name,**args)

        output_csv(outfile=csvfile_outfile,status='w',Area='North America',
                    test_begindate=test_begindate,test_enddate=test_enddate,
                    Daily_statistics_recording=Daily_statistics_recording,
                    Monthly_statistics_recording=Monthly_statistics_recording,
                    Annual_statistics_recording=Annual_statistics_recording,)       
    #calculate the correlation coefficient
    correlation_coefficient = np.corrcoef(obs_data_recording, geo_data_recording)[0, 1]
    print(f'Correlation Coefficient between Ground-based PM2.5 and Geophysical PM2.5: {correlation_coefficient:.4f}')

    
    if Apply_3D_CNN_architecture or Apply_CNN_architecture:
        del Init_CNN_Datasets
    elif Apply_Transformer_architecture:
        del Init_Transformer_Datasets
    del final_data_recording, obs_data_recording, geo_data_recording, sites_recording, dates_recording
    del training_final_data_recording, training_obs_data_recording, training_sites_recording, training_dates_recording
    gc.collect()   

                
def derive_Test_Training_index_4Each_BLISCO_fold(kfolds, number_of_SeedClusters, site_lat, site_lon, BLISCO_Buffer_Size):
    frac_testing  = 1.0/kfolds
    frac_training = 1.0 - frac_testing
    rkf = RepeatedKFold(n_splits=kfolds, n_repeats=1, random_state=19980130)
    number_of_test_sites = np.zeros((kfolds),dtype=np.int32)
    test_fold = 0
    for train_index, test_index in rkf.split(site_lat):
        number_of_test_sites[test_fold] = len(test_index)
        test_fold+=1
    # if # == -1   -> this site is for training for this fold, 
    # elif # == +1 -> this site is for testing for this fold.
    # elif # == 0  -> this site is exlcuded from training for this fold.
    index_for_BLISCO = np.zeros((kfolds,len(site_lat)),dtype=np.int64) 

    # calculate local monitor density
    usite_density = np.zeros(len(site_lat),dtype=np.float64)

    for isite in range(len(site_lat)):
        temp_Distances = calculate_distance_forArray(site_lat=site_lat[isite],site_lon=site_lon[isite],SATLAT_MAP=site_lat,SATLON_MAP=site_lon)
        temp_Density   = len(np.where(temp_Distances < 200.0)[0])
        usite_density[isite] = temp_Density

    ispot = np.zeros((len(site_lat))) # record sites that are still available for selecting as test datasets.
    BLISCO_criteria_radius = np.zeros((kfolds)) # this array is used to record the minimal criterial radius from sites to cluster seeds to select testing sites 
    # find stations that are still not withheld from selecting as the test sites.

    for ifold in range(kfolds):
        sites_unwithheld4testing = np.where(ispot == 0)[0].astype(int)
        sites_withheld4testing   = np.where(ispot > 0)[0].astype(int)

        # evenly divide stations by density, get the sites density limits by percentile
        density_percentile = np.percentile(usite_density[sites_unwithheld4testing], np.linspace(0,100,kfolds+1),interpolation='midpoint' )
        
        cluster_seeds_index = np.random.choice(range(len(sites_unwithheld4testing)), min(len(sites_unwithheld4testing),number_of_SeedClusters), replace=False)

        # --- print('sites_unwithheld4testing shape: {}, sites_unwithheld4testing 0:10 - {}, cluster_seeds_index[0:10]: {}'.format(sites_unwithheld4testing.shape,sites_unwithheld4testing[0:10],cluster_seeds_index[0:10]))
        
        # find distances between selected stations and other stations
        sites_unwithheld4testing_Distance = np.zeros((number_of_SeedClusters,len(sites_unwithheld4testing)))
        for icluster in range(len(cluster_seeds_index)):
            print('icluster: {}, \ncluster_seeds_index shape: {}, \nsites_unwithheld4testing shape:{}, \n site_lat shape:{}; site lon shape: {}, \nsites_unwithheld4testing_Distance shape:{}'.format(icluster,cluster_seeds_index.shape,sites_unwithheld4testing.shape,site_lat.shape,site_lon.shape,sites_unwithheld4testing_Distance.shape))
            temp_distance = calculate_distance_forArray(site_lat=site_lat[sites_unwithheld4testing[cluster_seeds_index[icluster]]],
                                                                                        site_lon=site_lon[sites_unwithheld4testing[cluster_seeds_index[icluster]]],
                                                                                        SATLAT_MAP=site_lat[sites_unwithheld4testing],SATLON_MAP=site_lon[sites_unwithheld4testing])
            sites_unwithheld4testing_Distance[icluster,:]= temp_distance
        # find the minimal distance of each sites to all seed clusters.

        Minimal_Distance2clusters = np.min(sites_unwithheld4testing_Distance,axis=0)
        Minimal_Distance2clusters_Sorted = np.sort(Minimal_Distance2clusters)
        
        ### calculate radius within which enough stations are located to fulfill this fold's quota.
        
        criterial_index = min(int(number_of_test_sites[ifold])-1,len(Minimal_Distance2clusters_Sorted)-1)
        BLISCO_criteria_radius[ifold] = Minimal_Distance2clusters_Sorted[criterial_index]
        # store testing stations for this fold, find all sites with distances smaller than the criterial radius
        if criterial_index < number_of_SeedClusters:
            #print('ifold: ',ifold,cluster_seeds_index[0:criterial_index+1],criterial_index+1, len(Minimal_Distance2clusters_Sorted),)
            ispot[sites_unwithheld4testing[cluster_seeds_index[0:criterial_index+1]]]= ifold + 1
        else:    
            ispot[sites_unwithheld4testing[np.where(Minimal_Distance2clusters <= BLISCO_criteria_radius[ifold] )]] = ifold + 1

        ifold_test_site_index       = np.where(ispot == (ifold+1))[0]
        ifold_init_train_site_index = np.where(ispot != (ifold+1))[0]

        ifold_train_site_index = GetBufferTrainingIndex(test_index=ifold_test_site_index,train_index=ifold_init_train_site_index,buffer=BLISCO_Buffer_Size,sitelat=site_lat,sitelon=site_lon)
        index_for_BLISCO[ifold,ifold_test_site_index]  = np.full((len(ifold_test_site_index)) , 1.0)
        index_for_BLISCO[ifold,ifold_train_site_index] = np.full((len(ifold_train_site_index)),-1.0)

    return index_for_BLISCO