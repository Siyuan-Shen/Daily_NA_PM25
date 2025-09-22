import torch
import numpy as np
import torch
import torch.nn as nn
import os
import gc
from sklearn.model_selection import RepeatedKFold
import random
import csv
import shap
from Evaluation_pkg.utils import *
import time

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

from Visualization_pkg.Evaluation_plots import shap_value_plot
from Visualization_pkg.iostream import get_figure_outfile_path
def Spatial_CV_SHAP_Analysis(total_channel_names, main_stream_channel_names,
                             side_stream_channel_names,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typeName = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=False, species=species)
    Evaluation_type = 'SHAPAnalysis_SpatialCV'
    Statistics_list = ['test_R2','train_R2','geo_R2','RMSE','NRMSE','slope','PWA']
    seed       = 19980130
    rkf = RepeatedKFold(n_splits=Spatial_CV_folds, n_repeats=1, random_state=seed)
    nchannel = len(total_channel_names)
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
        shap_values_values, shap_values_base,shap_values_data = np.zeros([0,nchannel,width,height],dtype=np.float32),np.array([],dtype=np.float32),np.zeros([0,nchannel,width,height],dtype=np.float32) #initialize_AVD_SHAPValues_DataRecording(beginyear=test_beginyear,endyear=test_endyear)
        
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
        shap_values_values, shap_values_base,shap_values_data = np.zeros([0,nchannel,depth,width,height],dtype=np.float32),np.array([],dtype=np.float32),np.zeros([0,nchannel,depth,width,height],dtype=np.float32) #initialize_AVD_SHAPValues_DataRecording(beginyear=test_beginyear,endyear=test_endyear)


    if Spatial_CV_SHAP_Analysis_Calculation_Switch:
        print('Spatial CV SHAP Analysis Calculation Switch is ON')
        print('Calculating SHAP values for each fold...')
         ### Load the datasets
        
        
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
                        
                    if Apply_CNN_architecture:
                        Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=Spatial_CV_training_begindates[imodel],
                                                                        enddates=Spatial_CV_training_enddates[imodel], version=version,species=species,
                                                                        nchannel=len(main_stream_channel_names),special_name=description,ifold=ifold,width=width,height=height)
                    elif Apply_3D_CNN_architecture:
                        Daily_Model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, begindates=Spatial_CV_training_begindates[imodel],
                                                                        enddates=Spatial_CV_training_enddates[imodel], version=version,species=species,
                                                                        nchannel=len(main_stream_channel_names),special_name=description,ifold=ifold,width=width,height=height,depth=depth)
                    background_data_number = min(len(y_train),Spatial_CV_SHAP_Analysis_background_number)
                    data_to_explain_number = min(len(y_test), Spatial_CV_SHAP_Analysis_test_number)
                    Back_Ground_Data = torch.Tensor(X_train[np.sort(np.random.choice(X_train.shape[0],background_data_number, replace=False))])
                    Data_to_Explain  = torch.Tensor(X_test[np.sort(np.random.choice(X_test.shape[0], data_to_explain_number, replace=False))])
                    print('Data_to_Explain.shape: {}, type: {}'.format(Data_to_Explain.shape, type(Data_to_Explain)))
                    Back_Ground_Data = Back_Ground_Data.to(device)
                    Data_to_Explain  = Data_to_Explain.to(device)
                    CNNModel_Explainer = shap.DeepExplainer(model=Daily_Model,data=Back_Ground_Data) 
                    #CNNModel_Explainer =  shap.Explainer(model=cnn_model,data=Back_Ground_Data)
                    shap_values = CNNModel_Explainer.shap_values(Data_to_Explain,check_additivity=False)
                    shap_values = np.squeeze(shap_values)
                    print(shap_values.shape)
                    Data_to_Explain = Data_to_Explain.cpu().detach().numpy()
                    shap_values_values = np.append(shap_values_values, shap_values, axis=0)
                    shap_values_data   = np.append(shap_values_data, Data_to_Explain, axis=0)
            if Apply_CNN_architecture:
                save_SHAPValues_data_recording(shap_values_values=shap_values_values,shap_values_data=shap_values_data,
                                           species=species,version=version,begindates=Spatial_CV_training_begindates[imodel],
                                           enddates=Spatial_CV_training_enddates[imodel],typeName=typeName,evaluation_type=Evaluation_type,
                                           nchannel=nchannel,width=width,height=height,)
            elif Apply_3D_CNN_architecture:
                save_SHAPValues_data_recording(shap_values_values=shap_values_values,shap_values_data=shap_values_data,
                                           species=species,version=version,begindates=Spatial_CV_training_begindates[imodel],
                                           enddates=Spatial_CV_training_enddates[imodel],typeName=typeName,evaluation_type=Evaluation_type,
                                           nchannel=nchannel,width=width,height=height,depth=depth)
    if Spatial_CV_SHAP_Analysis_visualization_Switch:
        if Apply_CNN_architecture:
            shap_values_values,shap_values_data = load_SHAPValues_data_recording(species=species,version=version,evaluation_type=Evaluation_type,
                                                                                  typeName=typeName,begindates=Spatial_CV_training_begindates[0],
                                                                                    enddates=Spatial_CV_training_enddates[0],nchannel=nchannel,width=width,height=height)
        elif Apply_3D_CNN_architecture:
            shap_values_values,shap_values_data = load_SHAPValues_data_recording(species=species,version=version,evaluation_type=Evaluation_type,
                                                                                  typeName=typeName,begindates=Spatial_CV_training_begindates[0],
                                                                                    enddates=Spatial_CV_training_enddates[0],nchannel=nchannel,width=width,height=height,depth=depth)
        print('shap_values_values.shape: ', shap_values_values.shape)
        print('shap_values_data.shape: ', shap_values_data.shape)
        if Spatial_CV_SHAP_Analysis_plot_type == 'beeswarm':
            if Apply_CNN_architecture:
                shap_values_values = np.sum(shap_values_values, axis=(2,3))
                shap_values_data   = np.sum(shap_values_data, axis=(2,3))
                shap_values_data_min = np.min(shap_values_data,axis=0)
                shap_values_data_max = np.max(shap_values_data,axis=0)
                shap_value_plot_outfile = get_figure_outfile_path(outdir=figure_outdir,evaluation_type=Evaluation_type, figure_type='Spatial_CV_SHAPValues',typeName=typeName,
                                                          begindate=Spatial_CV_training_begindates[0],enddate=Spatial_CV_training_enddates[0],
                                                          nchannel=nchannel,width=width,height=height)
            elif Apply_3D_CNN_architecture:
                shap_values_values = np.sum(shap_values_values, axis=(2,3,4))
                shap_values_data   = np.sum(shap_values_data, axis=(2,3,4))
                shap_values_data_min = np.min(shap_values_data,axis=0)
                shap_values_data_max = np.max(shap_values_data,axis=0)
                shap_value_plot_outfile = get_figure_outfile_path(outdir=figure_outdir,evaluation_type=Evaluation_type, figure_type='Spatial_CV_SHAPValues',typeName=typeName,
                                                          begindate=Spatial_CV_training_begindates[0],enddate=Spatial_CV_training_enddates[0],
                                                          nchannel=nchannel,width=width,height=height,depth=depth)
            print('shap_values_data.shape: ', shap_values_data.shape)
            shap_values_data = (shap_values_data - shap_values_data_min) / (shap_values_data_max-shap_values_data_min)
            print(np.min(shap_values_data,axis=0),np.max(shap_values_data,axis=0))
            shap_values_with_feature_names = shap.Explanation(values=shap_values_values,data=shap_values_data,feature_names=total_channel_names)
            shap_value_plot(shap_values_with_feature_names=shap_values_with_feature_names,plot_type=Spatial_CV_SHAP_Analysis_plot_type,outfile=shap_value_plot_outfile,)
        
        