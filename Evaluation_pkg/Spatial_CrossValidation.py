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
from Training_pkg.TrainingModule import CNN_train, cnn_predict, CNN3D_train, cnn_predict_3D
from Training_pkg.data_func import CNNInputDatasets, CNN3DInputDatasets
from Training_pkg.iostream import load_daily_datesbased_model
from wandb_config import wandb_run_config, wandb_initialize, init_get_sweep_config
from multiprocessing import Manager

def spatial_cross_validation(total_channel_names, main_stream_channel_names,
                             side_stream_channel_names,sweep_id=None,):
    world_size = torch.cuda.device_count()
    typeName = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=False, species=species)

    Evaluation_type = 'Spatial_CrossValidation'

    #####################################################################
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
    #####################################################################
    # Start the hyperparameters search validation
    Statistics_list = ['test_R2','train_R2','geo_R2','RMSE','NRMSE','slope','PWA']
    seed       = 19980130
    rkf = RepeatedKFold(n_splits=Spatial_CV_folds, n_repeats=1, random_state=seed)
    manager = Manager()
    run_id_container = manager.dict() 
    if not Use_recorded_data_to_show_validation_results_Spatial_CV:
            Training_losses_recording, Training_acc_recording, valid_losses_recording, valid_acc_recording = initialize_Loss_Accuracy_Recordings(kfolds=Spatial_CV_folds,n_models=len(Spatial_CV_training_begindates),epoch=epoch,batchsize=batchsize)
            

            final_data_recording = np.array([],dtype=float)
            obs_data_recording = np.array([],dtype=float)
            geo_data_recording = np.array([],dtype=float)
            sites_recording = np.array([],dtype=int)
            dates_recording = np.array([],dtype=int)

            training_final_data_recording = np.array([],dtype=float)
            training_obs_data_recording = np.array([],dtype=float)
            training_sites_recording = np.array([],dtype=int)
            training_dates_recording = np.array([],dtype=int)

            for imodel in range(len(Spatial_CV_training_begindates)):
                    if Apply_CNN_architecture or Apply_3D_CNN_architecture:
                        # Get the initial true_input and training datasets for the current model (within the desired time range)
                        print('1...')
                        desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_CNN_Datasets.get_desired_range_inputdatasets(start_date=Spatial_CV_training_begindates[imodel],
                                                                                                        end_date=Spatial_CV_training_enddates[imodel]) # initial datasets
                        # Normalize the training datasets
                        print('2...')
                        normalized_TrainingDatasets  = Init_CNN_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)
                        
                        # Concatenate the training datasets and true input for the current model for training and tetsing purposes
                        print('3...')
                        cctnd_trainingdatasets, cctnd_true_input,cctnd_ground_observation_data,cctnd_geophysical_species_data, cctnd_sites_index, cctnd_dates = Init_CNN_Datasets.concatenate_trainingdatasets(desired_true_input=desired_true_input, 
                                                                                                                                                desired_normalized_trainingdatasets=normalized_TrainingDatasets,
                                                                                                                                                desired_ground_observation_data=desired_ground_observation_data,
                                                                                                                                                desired_geophysical_species_data=desired_geophysical_species_data)
                    
                    print('4...')
                    sites_index=np.arange(total_sites_number)
                    for ifold, (training_selected_sites,testing_selected_sites) in enumerate(rkf.split(sites_index)):
                        print('training_selected_sites: ',training_selected_sites.shape)
                        print('testing_selected_sites: ',testing_selected_sites.shape) 
                    
                        X_train, y_train, X_test, y_test, dates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datasets_based_site_index(train_site_index=training_selected_sites,
                                                                                                                                            test_site_index=testing_selected_sites,
                                                                                                                                            total_trainingdatasets=cctnd_trainingdatasets,
                                                                                                                                            total_true_input=cctnd_true_input,
                                                                                                                                            total_sites_index=cctnd_sites_index,
                                                                                                                                            total_dates=cctnd_dates)
                        print('cctnd_ground_observation_data[test_datasets_index]: ', cctnd_ground_observation_data[test_datasets_index])
                        print('cctnd_geophysical_species_data[test_datasets_index]: ', cctnd_geophysical_species_data[test_datasets_index])
                        print('test_datasets_index: ', test_datasets_index)

                        if Apply_CNN_architecture:
                            if Spatial_CV_Apply_wandb_sweep_Switch:
                                sweep_mode = True
                                temp_sweep_config = init_get_sweep_config()
                                entity = temp_sweep_config.get("entity", "ACAG-NorthAmericaDailyPM25")
                                project = temp_sweep_config.get("project", version)
                                name = temp_sweep_config.get("name", None)
                            else:
                                sweep_mode = False
                                temp_sweep_config = None
                                entity = None
                                project = None
                                name = None
                            if world_size > 1:
                                mp.spawn(CNN_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                  X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                                Evaluation_type,typeName,Spatial_CV_training_begindates[imodel],\
                                                Spatial_CV_training_enddates[imodel],ifold),nprocs=world_size)
                            else:
                                CNN_train(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                  X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height, \
                                                Evaluation_type,typeName,Spatial_CV_training_begindates[imodel],\
                                                Spatial_CV_training_enddates[imodel],ifold)
                        
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
                            Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=Spatial_CV_training_begindates[imodel],
                                                                        enddates=Spatial_CV_training_enddates[imodel], version=version,species=species,
                                                                        nchannel=len(main_stream_channel_names),special_name=description,ifold=ifold,width=width,height=height)
                            validation_output = cnn_predict(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                            mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                            training_output = cnn_predict(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                            mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                        
                        

                        if Apply_3D_CNN_architecture:
                            if Spatial_CV_Apply_wandb_sweep_Switch:
                                sweep_mode = True
                                temp_sweep_config = init_get_sweep_config()
                                entity = temp_sweep_config.get("entity", "ACAG-NorthAmericaDailyPM25")
                                project = temp_sweep_config.get("project", version)
                                name = temp_sweep_config.get("name", None)

                            else:
                                sweep_mode = False
                                temp_sweep_config = None
                                entity = None
                                project = None
                                name = None
                            if world_size > 1:
                                mp.spawn(CNN3D_train,args=(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                    X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                                    Evaluation_type,typeName,Spatial_CV_training_begindates[imodel],\
                                                    Spatial_CV_training_enddates[imodel],ifold),nprocs=world_size)
                            else:
                                CNN3D_train(world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,total_channel_names,X_train, y_train,\
                                                    X_test, y_test, TrainingDatasets_mean, TrainingDatasets_std,width,height,depth, \
                                                    Evaluation_type,typeName,Spatial_CV_training_begindates[imodel],\
                                                    Spatial_CV_training_enddates[imodel],ifold)
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
                            Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=Spatial_CV_training_begindates[imodel],
                                                                        enddates=Spatial_CV_training_enddates[imodel], version=version,species=species,
                                                                        nchannel=len(main_stream_channel_names),special_name=description,ifold=ifold,width=width,height=height,depth=depth)
                            
                            validation_output = cnn_predict_3D(inputarray=X_test, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                            mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                            training_output = cnn_predict_3D(inputarray=X_train, model=Daily_Model, batchsize=3000, initial_channel_names=total_channel_names,
                                                            mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                    
                        del Daily_Model
                        gc.collect()
                        
                        # Get the final output for the validation datasets
                        final_output = Get_final_output(Validation_Prediction=validation_output, validation_geophysical_species=cctnd_geophysical_species_data[test_datasets_index],
                                                        bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,
                                                        log_species=False, mean=true_input_mean, std=true_input_std)
                        training_final_output = Get_final_output(Validation_Prediction=training_output, validation_geophysical_species=cctnd_geophysical_species_data[train_datasets_index],
                                                        bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,
                                                        log_species=False, mean=true_input_mean, std=true_input_std)
                        # Calculate the statistics for the validation datasets


                        final_data_recording = np.concatenate((final_data_recording, final_output), axis=0)
                        obs_data_recording = np.concatenate((obs_data_recording, cctnd_ground_observation_data[test_datasets_index]), axis=0)
                        geo_data_recording = np.concatenate((geo_data_recording, cctnd_geophysical_species_data[test_datasets_index]), axis=0)
                        sites_recording = np.concatenate((sites_recording, sites_test), axis=0)
                        dates_recording = np.concatenate((dates_recording, dates_test), axis=0)

                        training_final_data_recording = np.concatenate((training_final_data_recording, training_final_output), axis=0)
                        training_obs_data_recording = np.concatenate((training_obs_data_recording, cctnd_ground_observation_data[train_datasets_index]), axis=0)
                        training_sites_recording = np.concatenate((training_sites_recording, sites_train), axis=0)
                        training_dates_recording = np.concatenate((training_dates_recording, dates_train), axis=0)
                    
            if Apply_CNN_architecture:
                save_data_recording(final_data_recording=final_data_recording, obs_data_recording=obs_data_recording, geo_data_recording=geo_data_recording,
                                sites_recording=sites_recording, dates_recording=dates_recording,
                                training_final_data_recording=training_final_data_recording, training_obs_data_recording=training_obs_data_recording,
                                training_sites_recording=training_sites_recording, training_dates_recording=training_dates_recording,
                                species=species,version=version,begindates=Spatial_CV_training_begindates[0],
                                enddates=Spatial_CV_training_enddates[-1],typeName=typeName,nchannel=len(main_stream_channel_names),
                                evaluation_type=Evaluation_type,height=height,width=width,project=project,entity=entity,sweep_id=sweep_id,name=name)
            elif Apply_3D_CNN_architecture:
                save_data_recording(final_data_recording=final_data_recording, obs_data_recording=obs_data_recording, geo_data_recording=geo_data_recording,
                                sites_recording=sites_recording, dates_recording=dates_recording,
                                training_final_data_recording=training_final_data_recording, training_obs_data_recording=training_obs_data_recording,
                                training_sites_recording=training_sites_recording, training_dates_recording=training_dates_recording,
                                species=species,version=version,begindates=Spatial_CV_training_begindates[0],
                                enddates=Spatial_CV_training_enddates[-1],typeName=typeName,nchannel=len(main_stream_channel_names),
                                evaluation_type=Evaluation_type,height=height,width=width,depth=depth,project=project,entity=entity,sweep_id=sweep_id,name=name)
    if Apply_CNN_architecture:
         final_data_recording, obs_data_recording, geo_data_recording, sites_recording, dates_recording, training_final_data_recording, training_obs_data_recording, training_sites_recording, training_dates_recording = load_data_recording(species=species,version=version,begindates=Spatial_CV_training_begindates[0],
                                                                                                                                                                                                                                         enddates=Spatial_CV_training_enddates[-1],typeName=typeName,nchannel=len(main_stream_channel_names),
                                                                                                                                                                                                                                         evaluation_type=Evaluation_type,width=width,height=height,special_name=description,project=project,entity=entity,sweep_id=sweep_id)
    if Apply_3D_CNN_architecture:
        final_data_recording, obs_data_recording, geo_data_recording, sites_recording, dates_recording, training_final_data_recording, training_obs_data_recording, training_sites_recording, training_dates_recording = load_data_recording(species=species,version=version,begindates=Spatial_CV_training_begindates[0],
                                                                                                                                                                                                                                         enddates=Spatial_CV_training_enddates[-1],typeName=typeName,nchannel=len(main_stream_channel_names),
                                                                                                                                                                                                                                         evaluation_type=Evaluation_type,width=width,height=height,special_name=description,depth=depth,project=project,entity=entity,sweep_id=sweep_id)
    Daily_statistics_recording, Monthly_statistics_recording, Annual_statistics_recording = calculate_statistics(test_begindates=Spatial_CV_validation_begindates[0],
                                                                                                                test_enddates=Spatial_CV_validation_enddates[-1],final_data_recording=final_data_recording,
                                                                                                                obs_data_recording=obs_data_recording,geo_data_recording=geo_data_recording,
                                                                                                                sites_recording=sites_recording,dates_recording=dates_recording,
                                                                                                                training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,
                                                                                                                training_sites_recording=training_sites_recording,
                                                                                                                training_dates_recording=training_dates_recording,
                                                                                                                Statistics_list=Statistics_list,)
    if Apply_CNN_architecture:
        csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                          main_stream_channel_names=main_stream_channel_names,test_begindate=Spatial_CV_validation_begindates[0],
                                          test_enddate=Spatial_CV_validation_enddates[-1],
                                          width=width,height=height,project=project,entity=entity,sweep_id=sweep_id)
    elif Apply_3D_CNN_architecture:
        csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                          main_stream_channel_names=main_stream_channel_names,test_begindate=Spatial_CV_validation_begindates[0],
                                          test_enddate=Spatial_CV_validation_enddates[-1],
                                          width=width,height=height,depth=depth,project=project,entity=entity,sweep_id=sweep_id)
    
    output_csv(outfile=csvfile_outfile,status='w',Area='North America',
                test_begindate=Spatial_CV_validation_begindates[0],test_enddate=Spatial_CV_validation_enddates[-1],
                Daily_statistics_recording=Daily_statistics_recording,
                Monthly_statistics_recording=Monthly_statistics_recording,
                Annual_statistics_recording=Annual_statistics_recording,)
    
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


    
    for idate in range(len(Spatial_CV_validation_begindates)):
        test_begindate =  Spatial_CV_validation_begindates[idate]
        test_enddate = Spatial_CV_validation_enddates[idate]          
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

        if Apply_CNN_architecture:
            csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                            main_stream_channel_names=main_stream_channel_names,test_begindate=test_begindate,test_enddate=test_enddate,
                                            width=width,height=height,entity=entity,project=project,sweep_id=sweep_id,name=name)
        elif Apply_3D_CNN_architecture:
            csvfile_outfile = get_csvfile_outfile(Evaluation_type=Evaluation_type,typeName=typeName,Model_structure_type=Model_structure_type,
                                            main_stream_channel_names=main_stream_channel_names,test_begindate=test_begindate,test_enddate=test_enddate,
                                            width=width,height=height,depth=depth,entity=entity,project=project,sweep_id=sweep_id,name=name)
        
        output_csv(outfile=csvfile_outfile,status='w',Area='North America',
                    test_begindate=test_begindate,test_enddate=test_enddate,
                    Daily_statistics_recording=Daily_statistics_recording,
                    Monthly_statistics_recording=Monthly_statistics_recording,
                    Annual_statistics_recording=Annual_statistics_recording,)       
    #calculate the correlation coefficient
    correlation_coefficient = np.corrcoef(obs_data_recording, geo_data_recording)[0, 1]
    print(f'Correlation Coefficient between Ground-based PM2.5 and Geophysical PM2.5: {correlation_coefficient:.4f}')
    

    del Init_CNN_Datasets, final_data_recording, obs_data_recording, geo_data_recording, sites_recording, dates_recording
    del training_final_data_recording, training_obs_data_recording, training_sites_recording, training_dates_recording
    gc.collect()
                
    return