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
from Evaluation_pkg.data_func import Split_Datesets_based_on_dates, Split_Datasets_based_site_index,randomly_select_training_testing_indices,Get_final_output,Split_Datasets_randomly
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
from config import cfg
from Net_Architecture_config import cfg as net_architecture_cfg

def Temporal_Buffer_Out_CrossValidation(buffer_days,TBO_CV_max_test_days,total_channel_names, main_stream_channel_names,
                            side_stream_channel_names, sweep_id=None):
    world_size = torch.cuda.device_count()
    print(f"Number of available GPUs: {world_size}")
    typeName = Get_typeName(bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species, log_species=False, species=species)
    Evaluation_type = 'TBO_CrossValidation_{}-days_{}-MaxTestDays_{}-folds'.format(buffer_days,TBO_CV_max_test_days,TBO_CV_folds)

    #####################################################################
    
    #####################################################################
    # Start the hyperparameters search validation
    Statistics_list = ['test_R2','train_R2','geo_R2','RMSE','NRMSE','slope','PWA']
    seed       = 19980130
    
    manager = Manager()
    run_id_container = manager.dict() 
    
    #### Initialize the wandb for sweep mode, no sweep mode for TBO CV ####
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
        if MoE_Settings:
            Model_structure_type = '3DCNN_MoE_{}Experts_Model'.format(MoE_num_experts)
        elif MoCE_Settings:
            Model_structure_type = '3DCNN_MoCE_{}Experts_Model'.format(MoCE_num_experts)
        else:
            Model_structure_type = 'CNN3DModel'
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

    rkf = RepeatedKFold(n_splits=TBO_CV_folds, n_repeats=1, random_state=seed)


    if Apply_CNN_architecture:
        args = {'width': width, 'height': height}
    elif Apply_3D_CNN_architecture:
        args = {'width': width, 'height': height, 'depth': depth}
    elif Apply_Transformer_architecture:
        args = {'d_model': d_model, 'n_head': n_head, 'ffn_hidden': ffn_hidden, 'num_layers': num_layers, 'max_len': max_len+spin_up_len}
    elif Apply_CNN_Transformer_architecture:
        args = {'d_model': d_model, 'n_head': n_head, 'ffn_hidden': ffn_hidden, 'num_layers': num_layers, 'max_len': max_len+spin_up_len,
                'width': width, 'height': height, 'CNN_nchannel': len(CNN_Embedding_channel_names), 'Transformer_nchannel': len(Transformer_Embedding_channel_names)}

    if not Use_recorded_data_to_show_validation_results_TBO_CV:
        if not sweep_mode:
                cfg_outdir = Config_outdir + '{}/{}/Results/results-{}/configuration-files/'.format(species, version, Evaluation_type)
                os.makedirs(cfg_outdir, exist_ok=True)
                save_configuration_output(cfg_outdir=cfg_outdir, cfg=cfg, outdir=cfg_outdir, net_architecture_cfg=net_architecture_cfg, evaluation_type=Evaluation_type, typeName=typeName,
                                  nchannel=len(main_stream_channel_names), **args)
                
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
        for imodel in range(len(TBO_CV_training_begindates)):
            dates_range, int_dates_range = create_date_range(TBO_CV_training_begindates[imodel], TBO_CV_training_enddates[imodel])
            ### Get the training and targets in desired range
            if Apply_CNN_architecture or Apply_3D_CNN_architecture:
                # Get the initial true_input and training datasets for the current model (within the desired time range)
                print('1...',' Start Date: ', TBO_CV_training_begindates[imodel], ' End Date: ', TBO_CV_training_enddates[imodel])
                desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=TBO_CV_training_begindates[imodel],
                                                                                                end_date=TBO_CV_training_enddates[imodel]) # initial datasets
                # Normalize the training datasets
                print('2...', 'Desired Training Datasets: ', desired_trainingdatasets.keys())
                normalized_TrainingDatasets  = Init_CNN_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
                del desired_trainingdatasets
                gc.collect()
                # Concatenate the training datasets and true input for the current model for training and tetsing purposes
                
            elif Apply_Transformer_architecture:
                # Get the initial true_input and training datasets for the current model (within the desired time range)
                print('1...',' Start Date: ', TBO_CV_training_begindates[imodel], ' End Date: ', TBO_CV_training_enddates[imodel])
                desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_Transformer_Datasets.get_desired_range_inputdatasets(start_date=TBO_CV_training_begindates[imodel],
                                                                                                end_date=TBO_CV_training_enddates[imodel],max_len=max_len,spinup_len=spin_up_len)
                # Normalize the training datasets
                print('2...', 'Desired Training Datasets: ', desired_trainingdatasets.keys())
                normalized_TrainingDatasets  = Init_Transformer_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
                del desired_trainingdatasets
                gc.collect()
                
            elif Apply_CNN_Transformer_architecture:
                # Get the initial true_input and training datasets for the current model (within the desired time range)
                print('1...')
                desired_CNN_trainingdatasets, desired_Transformer_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=TBO_CV_training_begindates[imodel],
                                                                                                end_date=TBO_CV_training_enddates[imodel],max_len=max_len,spinup_len=spin_up_len)
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
            ### Get the training and testing indices for each fold based on the TBO method
            _, int_dates = create_date_range(TBO_CV_training_begindates[imodel], TBO_CV_training_enddates[imodel])
            start_date = pd.to_datetime(str(TBO_CV_training_begindates[imodel]), format="%Y%m%d")
            end_date   = pd.to_datetime(str(TBO_CV_training_enddates[imodel]), format="%Y%m%d")
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
            index_for_TBO = derive_Test_Training_index_4Each_TBO_fold(
                    kfolds=TBO_CV_folds, dates_range=dates, max_test_days=TBO_CV_max_test_days, TBO_buffer_days=buffer_days)
            print('index_for_TBO: ', index_for_TBO, 'shape: ', index_for_TBO.shape)

            for ifold in range(TBO_CV_folds):
                #### The test and training indices for this fold should be derived based on the initial sites index rather than the valid sites index.
                #### This is because concataned datasets are based on the initial sites index.
                
                test_index = int_dates[np.where(index_for_TBO[ifold,:] == 1.0)[0]]
                train_index = int_dates[np.where(index_for_TBO[ifold,:] == -1.0)[0]]
                excluded_index = int_dates[np.where(index_for_TBO[ifold,:] == 0.0)[0]]
                print('Buffer Size: {} days,No.{}-fold, test_index #: {}, train_index #: {}, total # of dates: {}'.format(buffer_days,ifold+1,len(test_index),len(train_index),len(int_dates)))
                print('4...')
                ### Split the datesets based on the indices of training and testing indices
                if Apply_3D_CNN_architecture or Apply_CNN_architecture or Apply_Transformer_architecture:
                    X_train, y_train, X_test, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datesets_based_on_dates(train_dates=train_index,
                                                                                                                                    test_dates=test_index,
                                                                                                                                    total_trainingdatasets=cctnd_trainingdatasets,
                                                                                                                                    total_true_input=cctnd_true_input,
                                                                                                                                    total_sites_index=cctnd_sites_index,
                                                                                                                                    total_dates=cctnd_dates)
                elif Apply_CNN_Transformer_architecture:
                    X_train_CNN, y_train, X_test_CNN, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datesets_based_on_dates(train_dates=train_index,
                                                                                                                                test_dates=test_index,
                                                                                                                                total_trainingdatasets=cctnd_CNN_trainingdatasets,
                                                                                                                                total_true_input=cctnd_true_input,
                                                                                                                                total_sites_index=cctnd_sites_index,
                                                                                                                                total_dates=cctnd_dates)
                    X_train_Transformer, y_train, X_test_Transformer, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datesets_based_on_dates(train_dates=train_index,
                                                                                                                                test_dates=test_index,
                                                                                                                                total_trainingdatasets=cctnd_Transformer_trainingdatasets,
                                                                                                                                total_true_input=cctnd_true_input,
                                                                                                                                total_sites_index=cctnd_sites_index,
                                                                                                                                total_dates=cctnd_dates)
                ## Start Training
                ## 2D CNN Training
                if Apply_CNN_architecture:
                    if not Use_saved_models_to_reproduce_validation_results_TBO_CV:
                        if world_size > 1:
                            mp.spawn(CNN_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                            Evaluation_type,typeName,TBO_CV_training_begindates[imodel],\
                                            TBO_CV_training_enddates[imodel],ifold),nprocs=world_size)
                        else:
                            CNN_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                            Evaluation_type,typeName,TBO_CV_training_begindates[imodel],\
                                            TBO_CV_training_enddates[imodel],ifold)

                    index_of_main_stream_channels_of_initial = [total_channel_names.index(channel) for channel in main_stream_channel_names]
                    X_train = X_train[:,index_of_main_stream_channels_of_initial,:,:]
                    X_test  = X_test[:,index_of_main_stream_channels_of_initial,:,:]
                    # Since in hyperparameter searching we do not apply multiple tests, we only see the final testing accuracy, so no loop here in 
                    # different time ranges. 
                    Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=TBO_CV_training_begindates[imodel],
                                                                enddates=TBO_CV_training_enddates[imodel], version=version,species=species,
                                                                nchannel=len(main_stream_channel_names),special_name=description,ifold=ifold,width=width,height=height)
                    validation_output = cnn_predict(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                    mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                    training_output = cnn_predict(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                    mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)


                # 3D CNN Training
                elif Apply_3D_CNN_architecture:
                    if not Use_saved_models_to_reproduce_validation_results_TBO_CV:
                        if world_size > 1:
                            mp.spawn(CNN3D_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                                Evaluation_type,typeName,TBO_CV_training_begindates[imodel],\
                                                TBO_CV_training_enddates[imodel],ifold),nprocs=world_size)
                        else:
                            CNN3D_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                                Evaluation_type,typeName,TBO_CV_training_begindates[imodel],\
                                                TBO_CV_training_enddates[imodel],ifold)
                    index_of_main_stream_channels_of_initial = [total_channel_names.index(channel) for channel in main_stream_channel_names]
                    X_train = X_train[:,index_of_main_stream_channels_of_initial,:,:,:]
                    X_test  = X_test[:,index_of_main_stream_channels_of_initial,:,:,:]
                    # Since in hyperparameter searching we do not apply multiple tests, we only see the final testing accuracy, so no loop here in 
                    # different time ranges. 
                    Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=TBO_CV_training_begindates[imodel],
                                                                enddates=TBO_CV_training_enddates[imodel], version=version,species=species,
                                                                nchannel=len(main_stream_channel_names),special_name=description,ifold=ifold,width=width,height=height,depth=depth)
                    
                    validation_output = cnn_predict_3D(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                    mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                    training_output = cnn_predict_3D(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                        mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                # Transformer Training
                elif Apply_Transformer_architecture:
                    if not Use_saved_models_to_reproduce_validation_results_TBO_CV:
                        if world_size > 1:
                            mp.spawn(Transformer_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std, \
                                                Evaluation_type,typeName,TBO_CV_training_begindates[imodel],\
                                                TBO_CV_training_enddates[imodel],ifold),nprocs=world_size)
                        else:
                            Transformer_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std, \
                                                Evaluation_type,typeName,TBO_CV_training_begindates[imodel],\
                                                TBO_CV_training_enddates[imodel],ifold)
                    index_of_main_stream_channels_of_initial = [total_channel_names.index(channel) for channel in main_stream_channel_names]
                    X_train = X_train[:,:,index_of_main_stream_channels_of_initial]
                    X_test  = X_test[:,:,index_of_main_stream_channels_of_initial]

                    # Since in hyperparameter searching we do not apply multiple tests, we only see the final testing accuracy, so no loop here in
                    # different time ranges.
                    Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=TBO_CV_training_begindates[imodel],
                                                                enddates=TBO_CV_training_enddates[imodel], version=version,species=species,
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
                    if not Use_saved_models_to_reproduce_validation_results_TBO_CV:
                        if world_size > 1:
                            mp.spawn(CNN_Transformer_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,CNN_Embedding_channel_names,Transformer_Embedding_channel_names,
                                                X_train_CNN, X_test_CNN,X_train_Transformer, X_test_Transformer,
                                                y_train, y_test,Transformer_trainingdatasets_mean, Transformer_trainingdatasets_std, width,height,
                                                Evaluation_type,typeName,TBO_CV_training_begindates[imodel],TBO_CV_training_enddates[imodel],ifold),nprocs=world_size)
                        else:
                            CNN_Transformer_train(0,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,CNN_Embedding_channel_names,Transformer_Embedding_channel_names,
                                                X_train_CNN, X_test_CNN,X_train_Transformer, X_test_Transformer,
                                                y_train, y_test,Transformer_trainingdatasets_mean, Transformer_trainingdatasets_std, width,height,
                                                Evaluation_type,typeName,TBO_CV_training_begindates[imodel],TBO_CV_training_enddates[imodel],ifold)
                    
                    excluded_Transformer_channel_names, main_stream_Transformer_channel_names, side_stream_Transformer_channel_names = Get_channel_names(channels_to_exclude=Transformer_channel_to_exclude, initial_channel_names=Transformer_Embedding_channel_names)
                    index_of_main_stream_CNN_channels_of_initial = [CNN_Embedding_channel_names.index(channel) for channel in main_stream_CNN_channel_names]
                    index_of_main_stream_Transformer_channels_of_initial = [Transformer_Embedding_channel_names.index(channel) for channel in main_stream_Transformer_channel_names]
                    X_train_CNN = X_train_CNN[:,:,index_of_main_stream_CNN_channels_of_initial,:,:]
                    X_test_CNN  = X_test_CNN[:,:,index_of_main_stream_CNN_channels_of_initial,:,:]
                    X_train_Transformer = X_train_Transformer[:,:,index_of_main_stream_Transformer_channels_of_initial]
                    X_test_Transformer  = X_test_Transformer[:,:,index_of_main_stream_Transformer_channels_of_initial]

                    # Since in hyperparameter searching we do not apply multiple tests, we only see the final testing accuracy, so no loop here in
                    # different time ranges.
                    Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=TBO_CV_training_begindates[imodel],
                                                                enddates=TBO_CV_training_enddates[imodel], version=version,species=species,
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
                                species=species,version=version,begindates=TBO_CV_training_begindates[0],
                                enddates=TBO_CV_training_enddates[-1],typeName=typeName,nchannel=len(main_stream_channel_names),
                                evaluation_type=Evaluation_type,project=project,entity=entity,sweep_id=sweep_id,name=name,**args)


    final_data_recording, obs_data_recording, geo_data_recording, sites_recording, dates_recording, training_final_data_recording, training_obs_data_recording, training_sites_recording, training_dates_recording, sites_lat_array, sites_lon_array = load_data_recording(species=species,version=version,begindates=TBO_CV_training_begindates[0],
                                                                                                                                                                                                                                     enddates=TBO_CV_training_enddates[-1],typeName=typeName,nchannel=len(main_stream_channel_names),
                                                                                                                                                                                                                                     evaluation_type=Evaluation_type,special_name=description,project=project,entity=entity,sweep_id=sweep_id,**args)

    ### Calculate statistics and record them to the whole time range
    Daily_statistics_recording, Monthly_statistics_recording, Annual_statistics_recording = calculate_statistics(test_begindates=TBO_CV_validation_begindates[0],
                                                                                                                test_enddates=TBO_CV_validation_enddates[-1],final_data_recording=final_data_recording,
                                                                                                                obs_data_recording=obs_data_recording,geo_data_recording=geo_data_recording,
                                                                                                                sites_recording=sites_recording,dates_recording=dates_recording,
                                                                                                                training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,
                                                                                                                training_sites_recording=training_sites_recording,
                                                                                                                training_dates_recording=training_dates_recording,
                                                                                                                Statistics_list=Statistics_list,)

    csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                          main_stream_channel_names=main_stream_channel_names,test_begindate=TBO_CV_validation_begindates[0],
                                          test_enddate=TBO_CV_validation_enddates[-1],project=project,entity=entity,sweep_id=sweep_id,
                                          **args,)
    output_csv(outfile=csvfile_outfile,status='w',Area='North America',
                test_begindate=TBO_CV_validation_begindates[0],test_enddate=TBO_CV_validation_enddates[-1],
                Daily_statistics_recording=Daily_statistics_recording,
                Monthly_statistics_recording=Monthly_statistics_recording,
                Annual_statistics_recording=Annual_statistics_recording,)    
       
    ####   Calculate the statistics and recording to each time period that is interested
    for idate in range(len(TBO_CV_validation_begindates)):
        test_begindate =  TBO_CV_validation_begindates[idate]
        test_enddate = TBO_CV_validation_enddates[idate]          
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

                

import pandas as pd
from collections import defaultdict
from math import ceil

def derive_Test_Training_index_4Each_TBO_fold(
    kfolds,
    dates_range,
    max_test_days,
    TBO_buffer_days
):
    """
    Temporal buffered CV aligned to input dates:
      - Each unique date is TEST exactly once (across all folds).
      - Each fold's TEST is multiple disjoint ranges, each <= max_test_days.
      - Ranges for the same fold are separated by >= TBO_buffer_days.
      - Training excludes +/- TBO_buffer_days around that fold's test ranges.

    Increasing max_test_days increases the length of disjoint test slots,
    provided feasibility: (kfolds - 1) * max_test_days >= TBO_buffer_days.
    """

    all_dates = pd.to_datetime(pd.Series(dates_range)) #Convert everything to pandas datetimes.
    unique_days = pd.DatetimeIndex(sorted(all_dates.unique())) #sorted list of distinct days
    n_unique = len(unique_days)
    n_total = len(all_dates)

    # Map unique day -> original positions (handle duplicates)
    uniq_pos = {d: i for i, d in enumerate(unique_days)} # map from a date → its index in unique_day
    orig_by_unique = defaultdict(list) # list of original indices whose day corresponds to unique day u
    for i, d in enumerate(all_dates):
        orig_by_unique[uniq_pos[d]].append(i)

    # ---- Feasibility check: ensure we can keep within-fold gaps >= buffer ----
    if (kfolds - 1) * max(1, int(max_test_days)) < TBO_buffer_days:
        raise ValueError(
            "Infeasible: (kfolds - 1) * max_test_days must be >= TBO_buffer_days "
            f"(got (kfolds-1)={kfolds-1}, max_test_days={max_test_days}, buffer={TBO_buffer_days}). "
            "Increase max_test_days or kfolds, or reduce TBO_buffer_days."
        )

    # ---- Build consecutive blocks covering ALL dates, with length <= max_test_days ----
    # This ensures that as max_test_days increases, blocks (test slots) get longer.
    block_days = int(max_test_days)  # monotonic: equals the max days
    blocks = []  # (u_start, u_end) on unique day index
    i = 0
    
    # s: start index of current block (set from i).
    # sdt: the start date of this block.
    # j: grows forward as long as the date span from sdt is < block_days.
    # e: end index of the block once growth stops.
    # Append (s, e); then set i = e + 1 to continue with the next block.
    while i < n_unique:
        s = i
        sdt = unique_days[s]
        j = s
        # grow up to block_days (by actual calendar days)
        while j + 1 < n_unique and (unique_days[j + 1] - sdt).days < block_days:
            j += 1
        e = j
        blocks.append((s, e))
        i = e + 1  # next block starts immediately (full coverage)
    # A sequence of disjoint blocks that cover every unique day once; each block length is ≤ max_test_days.
    # The blocks records the start and end indices for each test block in terms of unique days.
    
    # ---- Assign blocks to folds with "cooldown" >= buffer between blocks of same fold ----
    available_from = [pd.Timestamp.min] * kfolds   # next date this fold may start a block
    fold_sizes = [0] * kfolds                      # total test days assigned to fold
    fold_blocks = [[] for _ in range(kfolds)]      # list of (s,e) per fold
    
    # we iterate through all time blocks (each a range of consecutive days),
    # we need to assign each block to one of the kfolds.
    for (s, e) in blocks:
        start_dt = unique_days[s]
        end_dt   = unique_days[e]
        blen     = e - s + 1

        # folds eligible now (cooldown elapsed)
        eligible = [f for f in range(kfolds) if start_dt >= available_from[f]]
        if not eligible:
            # If no fold eligible yet (edge case with irregular dates), pick soonest available
            f = int(np.argmin(available_from))
        else:
            # Among eligible folds, balance by current test size
            # From the list eligible, pick the element x (a fold index) whose fold_sizes[x] is the smallest.
            f = min(eligible, key=lambda x: fold_sizes[x])
            

        fold_blocks[f].append((s, e))
        fold_sizes[f] += blen
        # enforce cooldown for this fold: next block must start after buffer window
        available_from[f] = end_dt + pd.Timedelta(days=TBO_buffer_days + 1)

    # ---- Build output aligned to the ORIGINAL dates_range ----
    index_for_TBO = np.full((kfolds, n_total), -1, dtype=np.int64)  # default train

    for f in range(kfolds):
        # test mask on unique axis
        test_u = np.zeros(n_unique, dtype=bool)
        for s, e in fold_blocks[f]:
            test_u[s:e+1] = True

        # buffer mask on unique axis (± buffer around each test block)
        buffer_u = np.zeros(n_unique, dtype=bool)
        for s, e in fold_blocks[f]:
            buf_start = unique_days[s] - pd.Timedelta(days=TBO_buffer_days)
            buf_end   = unique_days[e] + pd.Timedelta(days=TBO_buffer_days)
            left  = int(unique_days.searchsorted(buf_start, side="left"))
            right = int(unique_days.searchsorted(buf_end,   side="right")) - 1
            left = max(0, left); right = min(n_unique - 1, right)
            buffer_u[left:right+1] = True

        # map back to original observations
        for u in range(n_unique):
            obs = orig_by_unique[u]
            if test_u[u]:
                index_for_TBO[f, obs] = 1
            elif buffer_u[u]:
                index_for_TBO[f, obs] = 0
            else:
                index_for_TBO[f, obs] = -1

    # sanity: each unique day must be test exactly once across folds
    if not np.all((index_for_TBO == 1).sum(axis=0) == 1):
        raise RuntimeError("Each date must be assigned to TEST exactly once across folds.")

    return index_for_TBO
