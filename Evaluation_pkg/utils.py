import toml
import numpy as np
from datetime import datetime, timedelta
import os
import wandb
from Training_pkg.utils import *
from config import cfg
####################################################################################
###                            Evaluation Settings                               ###
####################################################################################



# Hyperparameter Search Validation Settings
Hyperparameter_Search_Validation_Settings_cfg = cfg['Hyperparameters_Search_Validation-Settings']

Hyperparameters_Search_Validation_Switch = Hyperparameter_Search_Validation_Settings_cfg['Hyperparameters_Search_Validation_Switch']
Use_recorded_data_to_show_validation_results = Hyperparameter_Search_Validation_Settings_cfg['Use_recorded_data_to_show_validation_results']
HSV_Apply_wandb_sweep_Switch = Hyperparameter_Search_Validation_Settings_cfg['HSV_Apply_wandb_sweep_Switch']
wandb_sweep_count = Hyperparameter_Search_Validation_Settings_cfg['wandb_sweep_count']
Hyperparameters_Search_Validation_Training_Settings_cfg = Hyperparameter_Search_Validation_Settings_cfg['Training-Settings']

HSV_Spatial_splitting_Switch                             = Hyperparameters_Search_Validation_Training_Settings_cfg['Spatial_splitting_Switch']
HSV_Spatial_splitting_begindates                         = Hyperparameters_Search_Validation_Training_Settings_cfg['Spatial_splitting_begindates']
HSV_Spatial_splitting_enddates                           = Hyperparameters_Search_Validation_Training_Settings_cfg['Spatial_splitting_enddates']
HSV_Spatial_splitting_training_portion                   = Hyperparameters_Search_Validation_Training_Settings_cfg['Spatial_splitting_training_portion']
HSV_Spatial_splitting_validation_portion                 = Hyperparameters_Search_Validation_Training_Settings_cfg['Spatial_splitting_validation_portion']

HSV_Temporal_splitting_Switch                            = Hyperparameters_Search_Validation_Training_Settings_cfg['Temporal_splitting_Switch']
HSV_Temporal_splitting_training_begindates               = Hyperparameters_Search_Validation_Training_Settings_cfg['Temporal_splitting_training_begindates']
HSV_Temporal_splitting_training_enddates                 = Hyperparameters_Search_Validation_Training_Settings_cfg['Temporal_splitting_training_enddates']
HSV_Temporal_splitting_validation_begindates             = Hyperparameters_Search_Validation_Training_Settings_cfg['Temporal_splitting_validation_begindates']
HSV_Temporal_splitting_validation_enddates               = Hyperparameters_Search_Validation_Training_Settings_cfg['Temporal_splitting_validation_enddates']


####################################################################################

# Random Cross-Validation Settings

Random_CV_Settings_cfg = cfg['Random-CrossValidation']
Random_CV_Switch = Random_CV_Settings_cfg['Random_CrossValidation_Switch']
Random_CV_Apply_wandb_sweep_Switch = Random_CV_Settings_cfg['Random_CV_Apply_wandb_sweep_Switch']
wandb_sweep_count_Random_CV = Random_CV_Settings_cfg['wandb_sweep_count_Random_CV']
Use_recorded_data_to_show_validation_results_Random_CV = Random_CV_Settings_cfg['Use_recorded_data_to_show_validation_results']

Random_CV_Training_Settings_cfg = Random_CV_Settings_cfg['Training-Settings']
Random_CV_folds = Random_CV_Training_Settings_cfg['Random_CV_folds']
Random_CV_training_begindates = Random_CV_Training_Settings_cfg['Random_CV_training_begindates']
Random_CV_training_enddates = Random_CV_Training_Settings_cfg['Random_CV_training_enddates']
Random_CV_validation_begindates = Random_CV_Training_Settings_cfg['Random_CV_validation_begindates']
Random_CV_validation_enddates = Random_CV_Training_Settings_cfg['Random_CV_validation_enddates']
Random_CV_validation_addtional_regions = Random_CV_Training_Settings_cfg['additional_validation_regions']

Random_CV_Visualization_Settings_cfg = Random_CV_Settings_cfg['Visualization_Settings']
Random_CV_regression_plot_switch = Random_CV_Visualization_Settings_cfg['regression_plot_switch']
Random_CV_plot_begindates = Random_CV_Visualization_Settings_cfg['plot_begindates']
Random_CV_plot_enddates = Random_CV_Visualization_Settings_cfg['plot_enddates']

Random_CV_SHAP_Analysis_Settings_cfg = Random_CV_Settings_cfg['SHAP_Analysis_Settings']
Random_CV_SHAP_Analysis_Switch = Random_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_switch']
Random_CV_SHAP_Analysis_Calculation_Switch = Random_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_Calculation_Switch']
Random_CV_SHAP_Analysis_visualization_Switch = Random_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_visualization_Switch']
Random_CV_SHAP_Analysis_background_number = Random_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_background_number']
Random_CV_SHAP_Analysis_test_number = Random_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_test_number']
Random_CV_SHAP_Analysis_plot_type = Random_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_plot_type']

####################################################################################

# Spatial Cross-Validation Settings

Spatial_CV_Settings_cfg = cfg['Spatial-CrossValidation']
Spatial_CrossValidation_Switch = Spatial_CV_Settings_cfg['Spatial_CrossValidation_Switch']
Spatial_CV_Apply_wandb_sweep_Switch = Spatial_CV_Settings_cfg['Spatial_CV_Apply_wandb_sweep_Switch']
wandb_sweep_count_Spatial_CV = Spatial_CV_Settings_cfg['wandb_sweep_count_Spatial_CV']
Use_recorded_data_to_show_validation_results_Spatial_CV = Spatial_CV_Settings_cfg['Use_recorded_data_to_show_validation_results']

Spatial_CV_Training_Settings_cfg = Spatial_CV_Settings_cfg['Training-Settings']
Spatial_CV_folds = Spatial_CV_Training_Settings_cfg['Spatial_CV_folds']
Spatial_CV_training_begindates = Spatial_CV_Training_Settings_cfg['Spatial_CV_training_begindates']
Spatial_CV_training_enddates = Spatial_CV_Training_Settings_cfg['Spatial_CV_training_enddates']
Spatial_CV_validation_begindates = Spatial_CV_Training_Settings_cfg['Spatial_CV_validation_begindates']
Spatial_CV_validation_enddates = Spatial_CV_Training_Settings_cfg['Spatial_CV_validation_enddates']
Spatial_CV_validation_addtional_regions = Spatial_CV_Training_Settings_cfg['additional_validation_regions']

Spatial_CV_Visualization_Settings_cfg = Spatial_CV_Settings_cfg['Visualization_Settings']
Spatial_CV_regression_plot_switch = Spatial_CV_Visualization_Settings_cfg['regression_plot_switch']
Spatial_CV_plot_begindates = Spatial_CV_Visualization_Settings_cfg['plot_begindates']
Spatial_CV_plot_enddates = Spatial_CV_Visualization_Settings_cfg['plot_enddates']

Spatial_CV_SHAP_Analysis_Settings_cfg = Spatial_CV_Settings_cfg['SHAP_Analysis_Settings']
Spatial_CV_SHAP_Analysis_Switch = Spatial_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_switch']
Spatial_CV_SHAP_Analysis_Calculation_Switch = Spatial_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_Calculation_Switch']
Spatial_CV_SHAP_Analysis_visualization_Switch = Spatial_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_visualization_Switch']
Spatial_CV_SHAP_Analysis_background_number = Spatial_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_background_number']
Spatial_CV_SHAP_Analysis_test_number = Spatial_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_test_number']
Spatial_CV_SHAP_Analysis_plot_type = Spatial_CV_SHAP_Analysis_Settings_cfg['SHAP_Analysis_plot_type']

####################################################################################

# BLISCO Cross-Validation Settings

BLISCO_CV_Settings_cfg = cfg['BLISCO-CrossValidation']
BLISCO_CrossValidation_Switch = BLISCO_CV_Settings_cfg['BLISCO_CV_Switch']
Use_recorded_data_to_show_validation_results_BLISCO_CV = BLISCO_CV_Settings_cfg['Use_recorded_data_to_show_validation_results']

BLISCO_CV_Training_Settings_cfg = BLISCO_CV_Settings_cfg['Training-Settings']
BLISCO_CV_folds = BLISCO_CV_Training_Settings_cfg['BLISCO_CV_folds']
BLISCO_CV_buffer_radius_km = BLISCO_CV_Training_Settings_cfg['BLISCO_CV_buffer_radius_km']
BLISCO_CV_seeds_number = BLISCO_CV_Training_Settings_cfg['BLISCO_CV_seeds_number']
BLISCO_CV_training_begindates = BLISCO_CV_Training_Settings_cfg['BLISCO_CV_training_begindates']
BLISCO_CV_training_enddates = BLISCO_CV_Training_Settings_cfg['BLISCO_CV_training_enddates']
BLISCO_CV_validation_begindates = BLISCO_CV_Training_Settings_cfg['BLISCO_CV_validation_begindates']
BLISCO_CV_validation_enddates = BLISCO_CV_Training_Settings_cfg['BLISCO_CV_validation_enddates']
BLISCO_CV_validation_addtional_regions = BLISCO_CV_Training_Settings_cfg['additional_validation_regions']

BLISCO_CV_Visualization_Settings_cfg = BLISCO_CV_Settings_cfg['Visualization_Settings']
Test_Train_Buffers_Distributions_plot_switch = BLISCO_CV_Visualization_Settings_cfg['Test_Train_Buffers_Distributions_plot_switch']

####################################################################################
def Get_typeName(bias, normalize_bias, normalize_species, absolute_species, log_species, species):
    if bias == True:
        typeName = '{}-bias'.format(species)
    elif normalize_bias:
        if normalize_type == 'Gaussian':
            typeName = 'GaussianNormalized-{}-bias'.format(species)
        elif normalize_type == 'MinMax':
            typeName = 'MinMaxNormalized-{}-bias'.format(species)
    elif normalize_species == True:
        if normalize_type == 'Gaussian':
            typeName = 'GaussianNormalized-{}'.format(species)
        elif normalize_type == 'MinMax':
            typeName = 'MinMaxNormalized-{}'.format(species)
    elif absolute_species == True:
        typeName = 'Absolute-{}'.format(species)
    elif log_species == True:
        typeName = 'Log-{}'.format(species)
    return  typeName

def initialize_Loss_Accuracy_Recordings(kfolds,n_models,epoch,batchsize):
    Training_losses_recording = np.zeros((kfolds,n_models,epoch*10),dtype=np.float32)
    Training_acc_recording    = np.zeros((kfolds,n_models,epoch*10),dtype=np.float32)
    valid_losses_recording    = np.zeros((kfolds,n_models,epoch*10),dtype=np.float32)
    valid_acc_recording       = np.zeros((kfolds,n_models,epoch*10),dtype=np.float32)
    print('Training_losses_recording.shape: '.format(Training_losses_recording.shape) + '----------------------')
    return Training_losses_recording, Training_acc_recording, valid_losses_recording, valid_acc_recording

def initialize_statistics_recordings(test_start_date,test_end_date,Statistics_list):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual','MAM','JJA','SON','DJF']
    
    Daily_statistics_recording = {}
    Monthly_statistics_recording = {}
    Annual_statistics_recording = {}

    Daily_statistics_recording['Purely_Spatial'] = {}
    Daily_statistics_recording['All_points'] = {}
    Daily_statistics_recording['Monthly_Scale'] = {}

    Monthly_statistics_recording['Purely_Spatial'] = {}
    Monthly_statistics_recording['All_points'] = {}

    Annual_statistics_recording['Purely_Spatial'] = {}
    Annual_statistics_recording['All_points'] = {}


    for imonth in MONTH:
        Daily_statistics_recording['Monthly_Scale'][imonth] = {}
        Monthly_statistics_recording['Purely_Spatial'][imonth] = {}
        Monthly_statistics_recording['All_points'][imonth] = {}
        Annual_statistics_recording['Purely_Spatial'][imonth] = {}
        Annual_statistics_recording['All_points'][imonth] = {}
      
    Monthly_statistics_recording['Purely_Spatial']['AllMonths'] = {}
    Monthly_statistics_recording['All_points']['AllMonths'] = {}

    for stat in Statistics_list:
        Daily_statistics_recording['Purely_Spatial'][stat] = np.array([],dtype=np.float64)
        Daily_statistics_recording['All_points'][stat] = np.array([],dtype=np.float64)
        Monthly_statistics_recording['Purely_Spatial']['AllMonths'][stat] = np.array([],dtype=np.float64)
        Monthly_statistics_recording['All_points']['AllMonths'][stat] = np.array([],dtype=np.float64)
        for imonth in MONTH:
            Daily_statistics_recording['Monthly_Scale'][imonth][stat] = np.array([],dtype=np.float64)
            Monthly_statistics_recording['Purely_Spatial'][imonth][stat] = np.array([],dtype=np.float64)
            Monthly_statistics_recording['All_points'][imonth][stat] = np.array([],dtype=np.float64)
            Annual_statistics_recording['Purely_Spatial'][imonth][stat] = np.array([],dtype=np.float64)
            Annual_statistics_recording['All_points'][imonth][stat] = np.array([],dtype=np.float64)
        
    return Daily_statistics_recording, Monthly_statistics_recording, Annual_statistics_recording

def get_csvfile_outfile(Evaluation_type, typeName,Model_structure_type,main_stream_channel_names,test_begindate,test_enddate,**args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)
    entity = args.get('entity', 'ACAG-NorthAmericaDailyPM25')
    project = args.get('project', 'Daily_PM25_DL_2024')
    sweep_id = args.get('sweep_id', None)
    name = args.get('name', None)
    d_model = args.get('d_model', 64)
    n_head = args.get('n_head', 8)
    ffn_hidden = args.get('ffn_hidden', 256)
    num_layers = args.get('num_layers', 6)
    max_len = args.get('max_len', 1000)
    CNN_nchannel = args.get('CNN_nchannel', 4)
    Transformer_nchannel = args.get('Transformer_nchannel', 3)

    if Apply_CNN_architecture:
        
        csvfile_outdir = csv_outdir + '{}/{}/Results/results-{}/statistical_indicators/{}_{}_{}_{}_{}_{}Channel_{}x{}{}/'.format(species,version,Evaluation_type,
                                                                                                                                                      Evaluation_type,Model_structure_type,typeName,
                                                                                                                                                      species,version,
                                                                                                                                                      len(main_stream_channel_names),width,height,description)    
        if not os.path.isdir(csvfile_outdir):
            os.makedirs(csvfile_outdir)
        
        if (Spatial_CrossValidation_Switch and Spatial_CV_Apply_wandb_sweep_Switch) or (Hyperparameters_Search_Validation_Switch and HSV_Apply_wandb_sweep_Switch):

            csvfile_outdir = csvfile_outdir + 'sweep-{}/'.format(name)
            if not os.path.isdir(csvfile_outdir):
                os.makedirs(csvfile_outdir)
            
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}_{}Channel_{}-{}_{}x{}_sweep-{}.csv'.format(species,version,
                                                                                                                Evaluation_type,Model_structure_type,typeName,
                                                                                                                len(main_stream_channel_names),test_begindate,
                                                                                                                test_enddate,width,height,sweep_id)
        else:
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}_{}Channel_{}-{}_{}x{}{}.csv'.format(species,version,
                                                                                                                Evaluation_type,Model_structure_type,typeName,
                                                                                                                len(main_stream_channel_names),test_begindate,
                                                                                                                test_enddate,width,height,description)
    elif Apply_3D_CNN_architecture:
        csvfile_outdir = csv_outdir + '{}/{}/Results/results-{}/statistical_indicators/{}_{}_{}_{}_{}_{}Channel_{}x{}x{}{}/'.format(species,version,Evaluation_type,
                                                                                      Evaluation_type,Model_structure_type,typeName, species,version,
                                                                                        len(main_stream_channel_names),depth,width,height,description)
        if not os.path.isdir(csvfile_outdir):
            os.makedirs(csvfile_outdir)
        if (Spatial_CrossValidation_Switch and Spatial_CV_Apply_wandb_sweep_Switch) or (Hyperparameters_Search_Validation_Switch and HSV_Apply_wandb_sweep_Switch):

            csvfile_outdir = csvfile_outdir + 'sweep-{}/'.format(name)
            if not os.path.isdir(csvfile_outdir):
                os.makedirs(csvfile_outdir)
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}_{}Channel_{}-{}_{}x{}x{}_sweep-{}.csv'.format(species,version,
                                                                                                                Evaluation_type,Model_structure_type,typeName,
                                                                                                                len(main_stream_channel_names),test_begindate,
                                                                                                                test_enddate,depth,width,height,sweep_id)
        else:
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}_{}Channel_{}-{}_{}x{}x{}.csv'.format(species,version,
                                                                                                                Evaluation_type,Model_structure_type,typeName,
                                                                                                                len(main_stream_channel_names),test_begindate,
                                                                                                                test_enddate,depth,width,height,description)
    elif Apply_Transformer_architecture:
        csvfile_outdir = csv_outdir + '{}/{}/Results/results-{}/statistical_indicators/{}_{}_{}_{}_{}_{}Channel_{}dmodel_{}heads_{}ffnHidden_{}numlayers_{}lens{}/'.format(species,version,Evaluation_type,
                                                                                      Evaluation_type,Model_structure_type,typeName, species,version,
                                                                                        len(main_stream_channel_names),d_model,n_head,ffn_hidden,num_layers,max_len,description)
        if not os.path.isdir(csvfile_outdir):
            os.makedirs(csvfile_outdir)
        if (Spatial_CrossValidation_Switch and Spatial_CV_Apply_wandb_sweep_Switch) or (Hyperparameters_Search_Validation_Switch and HSV_Apply_wandb_sweep_Switch):

            csvfile_outdir = csvfile_outdir + 'sweep-{}/'.format(name)
            if not os.path.isdir(csvfile_outdir):
                os.makedirs(csvfile_outdir)
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}_{}Channel_{}-{}_{}dmodel_{}heads_{}ffnHidden_{}numlayers_{}lens_sweep-{}.csv'.format(species,version,
                                                                                                                Evaluation_type,Model_structure_type,typeName,
                                                                                                                len(main_stream_channel_names),test_begindate,
                                                                                                                test_enddate,d_model,n_head,ffn_hidden,num_layers,max_len,sweep_id)
        else:
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}_{}Channel_{}-{}_{}dmodel_{}heads_{}ffnHidden_{}numlayers_{}lens{}.csv'.format(species,version,
                                                                                                                Evaluation_type,Model_structure_type,typeName,
                                                                                                                len(main_stream_channel_names),test_begindate,
                                                                                                                test_enddate,d_model,n_head,ffn_hidden,num_layers,max_len,description)
    elif Apply_CNN_Transformer_architecture:
        csvfile_outdir = csv_outdir + '{}/{}/Results/results-{}/statistical_indicators/{}_{}_{}_{}_{}_{}CNNChannel_{}TransformerChannel_{}dmodel_{}heads_{}ffnHidden_{}numlayers_{}lens{}/'.format(species,version,Evaluation_type,
                                                                                      Evaluation_type,Model_structure_type,typeName, species,version,CNN_nchannel,Transformer_nchannel,d_model,n_head,ffn_hidden,num_layers,max_len,description)
        if not os.path.isdir(csvfile_outdir):
            os.makedirs(csvfile_outdir)
        if (Spatial_CrossValidation_Switch and Spatial_CV_Apply_wandb_sweep_Switch) or (Hyperparameters_Search_Validation_Switch and HSV_Apply_wandb_sweep_Switch):

            csvfile_outdir = csvfile_outdir + 'sweep-{}/'.format(name)
            if not os.path.isdir(csvfile_outdir):
                os.makedirs(csvfile_outdir)
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}_{}CNNChannel_{}TransformerChannel_{}-{}_{}dmodel_{}heads_{}ffnHidden_{}numlayers_{}lens_sweep-{}.csv'.format(species,version,
                                                                                                                Evaluation_type,Model_structure_type,typeName,test_begindate,
                                                                                                                test_enddate,CNN_nchannel,Transformer_nchannel,d_model,n_head,ffn_hidden,num_layers,max_len,sweep_id)
        else:
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}_{}CNNChannel_{}TransformerChannel_{}-{}_{}dmodel_{}heads_{}ffnHidden_{}numlayers_{}lens{}.csv'.format(species,version,
                                                                                                                Evaluation_type,Model_structure_type,typeName,test_begindate,
                                                                                                                test_enddate,CNN_nchannel,Transformer_nchannel,d_model,n_head,ffn_hidden,num_layers,max_len,description)
    return csvfile_outfile

    
def get_nearest_point_index(sitelon, sitelat, lon_grid, lat_grid):
    '''
    func: get the index of stations on the grids map
    inputs:
        sitelon, sitelat: stations location, eg:[42.353,110.137] 0th dim:lat 1st dim:lat
        lon_grid: grids longitude
        lat_grid: grids latitude
    return:
        index: [index_lat,index_lon]
    '''
    # step1: get the spatial resolution; Default: the latitude and longitude have the same resolution
    det = 0.01
    # step2:
    lon_min = np.min(lon_grid)
    lat_min = np.min(lat_grid)
    index_lon = np.round((sitelon - lon_min) / det)
    index_lat = np.round((sitelat - lat_min) / det)
    index_lon = index_lon.astype(int)
    index_lat = index_lat.astype(int)
    print('site_lat: {}, \n lat_min: {}'.format(sitelat, lat_min))
    return index_lon,index_lat

def getGrg_YYYY_MM_DD(date):
    MONTHs = ['01','02','03','04','05','06','07','08','09','10','11','12']
    DAYs   = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
    YYYY = str(np.floor(date/10000).astype(int))
    MM = MONTHs[int(np.floor((date%10000)/100))-1]
    DD = DAYs[int(date%100)-1]
    return YYYY, MM, DD


def create_date_range(start_date, end_date):
    """
    Create a time series between two dates.
    
    Parameters:
    -----------
    start_date : str or int
        Start date in YYYYMMDD format (e.g., '20180101' or 20180101)
    end_date : str or int
        End date in YYYYMMDD format (e.g., '20181231' or 20181231)
        
    Returns:
    --------
    dates : list
        List of datetime objects representing the date range
    int_dates : list
        List of date integers in YYYYMMDD format
    """
    # Convert start and end dates to datetime objects
    str_start_date = str(start_date)
    str_end_date = str(end_date)
    start = datetime.strptime(str_start_date, '%Y%m%d')
    end = datetime.strptime(str_end_date, '%Y%m%d')
    
    # Generate the date range
    dates = []
    int_dates = []
    current = start
    
    while current <= end:
        dates.append(current)
        int_dates.append(int(current.strftime('%Y%m%d')))
        current += timedelta(days=1)
    int_dates = np.array(int_dates)
    return dates, int_dates

def get_YYYY_MM(start_date, end_date):
    dates, dates_series = create_date_range(start_date, end_date)
    MONTHs = ['01','02','03','04','05','06','07','08','09','10','11','12']
    DAYs   = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
    MM_series_dict = {}
    YYYY_series_dict = {}
    
    YYYY_series = np.floor(dates_series/10000).astype(int)
    MM_series = np.floor((dates_series%10000)/100).astype(int)
    DD_series = dates_series%100
    total_unique_YYYY = np.unique(YYYY_series)
    for iyear, YYYY in enumerate(total_unique_YYYY):
        YYYY_series_dict[YYYY] = {}
        YYYY_series_dict[YYYY]['start_date'] = np.min(dates_series[np.where(YYYY_series == YYYY)[0]])
        YYYY_series_dict[YYYY]['end_date'] = np.max(dates_series[np.where(YYYY_series == YYYY)[0]])
        
    for imonth,MM in enumerate(MONTHs):
        MM_series_dict[MM] = {}
        MM_series_dict[MM]['start_day'] = {}
        MM_series_dict[MM]['end_day'] = {}
        MM_series_dict[MM]['YYYY'] = {}
        temp_YYYY = YYYY_series[np.where(MM_series == (imonth+1))[0]]
        if len(temp_YYYY) > 0:
            unique_YYYY = np.unique(temp_YYYY)
            MM_series_dict[MM]['YYYY']= unique_YYYY
            for iyear in unique_YYYY:
                temp_YYYYMM_index = np.where((YYYY_series == iyear) & (MM_series == (imonth+1)))[0]
                temp_DD_series = DD_series[temp_YYYYMM_index]
                MM_series_dict[MM]['start_day'][iyear] = np.min(temp_DD_series)
                MM_series_dict[MM]['end_day'][iyear] = np.max(temp_DD_series)
    return YYYY_series_dict,MM_series_dict, total_unique_YYYY



################################################################
## Calculate teh distances between stations
################################################################
import math
import time 


def GetBufferTrainingIndex(test_index:np.array,train_index:np.array,buffer:float,sitelat:np.array, sitelon:np.array):
    """_summary_

    Args:
        test_index (np.array): _description_
        train_index (np.array): _description_
        buffer (float): _description_
    """
    time_start = time.time()
    for isite in range(len(test_index)):
        train_index = find_sites_nearby(test_lat=sitelat[test_index[isite]],test_lon=sitelon[test_index[isite]],train_index=train_index,
                                        train_lat=sitelat,train_lon=sitelon,buffer_radius=buffer)
    time_end = time.time()
    #print('Number of train index: ',len(train_index),'\nNumber of test index: ', len(test_index),'\nTime consume: ',str(np.round(time_end-time_start,4)),'s')
    return train_index


def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Radius of the Earth in kilometers (use 3956 for miles)
    r = 6371
    
    # Calculate the distance
    distance = r * c
    
    return distance


def calculate_distance_forArray(site_lat:np.float32,site_lon:np.float32,
                                SATLAT_MAP:np.array,SATLON_MAP:np.array,r=6371.01):
    if np.ndim(SATLAT_MAP) == 0:
        dist_map = calculate_distance(site_lat,site_lon,SATLAT_MAP,SATLON_MAP)
    elif np.ndim(SATLAT_MAP) == 1:
        dist_map = np.zeros(SATLAT_MAP.shape,dtype = np.float64)
        for ix in range(SATLAT_MAP.shape[0]):
            dist_map[ix] = calculate_distance(site_lat,site_lon,SATLAT_MAP[ix],SATLON_MAP[ix])
    elif np.ndim(SATLAT_MAP) == 2:
        dist_map = np.zeros(SATLAT_MAP.shape,dtype = np.float64)
        for ix in range(SATLAT_MAP.shape[0]):
            for iy in range(SATLAT_MAP.shape[1]):
                dist_map[ix,iy] = calculate_distance(site_lat,site_lon,SATLAT_MAP[ix,iy],SATLON_MAP[ix,iy])
   
   #other_sites_pos1_array = np.zeros(len(SATLAT_MAP),dtype=np.float64)
    #other_sites_pos2_array = np.zeros(len(SATLAT_MAP),dtype=np.float64)
    #for i in range(len(SATLAT_MAP)):
       # other_sites_pos1_array[i] = math.radians(SATLAT_MAP[i])
       # other_sites_pos2_array[i] = math.radians(SATLON_MAP[i])
    
    #site_pos1 = site_lat * np.pi / 180.0
    #site_pos2 = site_lon * np.pi / 180.0
    #other_sites_pos1_array = SATLAT_MAP * np.pi / 180.0
    #other_sites_pos2_array = SATLON_MAP * np.pi / 180.0
    #dist_map = r * np.arccos(np.sin(site_pos1)*np.sin(other_sites_pos1_array)+np.cos(site_pos1)*np.cos(other_sites_pos1_array)*np.cos(site_pos2-other_sites_pos2_array))
    
    return dist_map

def find_sites_nearby(test_lat: np.float32, test_lon: np.float32,train_index:np.array,
                      train_lat: np.array, train_lon: np.array, buffer_radius: np.float32):
    """This function is used to get the sites index within the buffe area and exclue them from the training index. 

    Args:
        test_lat (np.float32): Test site latitude.
        test_lon (np.float32): Test site longitude.
        train_index (np.array): Training index(remain). This function should be in a loop,
        and all input training index already exclude other sites within the buffer zone near other testing site.
        train_lat (np.array): The initial sites lat array.
        train_lon (np.array): The initial sites lon array.
        buffer_radius (np.float32): The buffer radius.

    Returns:
        np.array : The train index exclude the sites within the input test sites surronding buffer zone.
    """
    lat_min = max(-69.95, (test_lat - 0.1 * buffer_radius))
    lat_max = min(69.95, (test_lat + 0.1 * buffer_radius))
    lon_min = max(-179.95, (test_lon - 0.1 * buffer_radius))
    lon_max = min(179.95, (test_lon + 0.1 * buffer_radius))
    # Find the sites within the square first
    lat_index = np.intersect1d(np.where(train_lat>lat_min),np.where(train_lat<lat_max))
    lon_index = np.intersect1d(np.where(train_lon>lon_min),np.where(train_lon<lon_max))
    sites_nearby_index = np.intersect1d(lat_index,lon_index)
           
    sites_lat_nearby = train_lat[sites_nearby_index]
    sites_lon_nearby = train_lon[sites_nearby_index]

    # Find the sites within the buffer zones
    sites_within_radius_index = np.array([],dtype=int)
    for isite in range(len(sites_nearby_index)):
        distance = calculate_distance(test_lat,test_lon,train_lat[sites_nearby_index[isite]],train_lon[sites_nearby_index[isite]])
        if distance < buffer_radius:
            sites_within_radius_index = np.append(sites_within_radius_index,sites_nearby_index[isite])
    sites_within_index,X_index,Y_index = np.intersect1d(train_index,sites_within_radius_index,return_indices=True)
    train_index = np.delete(train_index,X_index)
    return train_index

