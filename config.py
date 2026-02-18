import time 
#########################################################################################################################################################
Pathbegin_YEAR = 2018
Pathend_YEAR = 2023
AVD_OBS_version = 'AVD_d20250804' # 'AVD_d20240814' 'AVD_d20250804'
AVD_GEO_version = 'vAOD20240322vGEO20241212'

Use_AOD_nan_values_filtered_Obs = False ## True: use the AOD that is not filled to filter the obs, high quality geophysical a priori; False: use the filled AOD
Include_NAPS = True ### True: include NAPS data; False: not include NAPS data
NAPS_version = 'NAPS_20250807' # 'NAPS_20250807', 'NAPS-20240813'

if Include_NAPS:
    observation_data_NAPS_insertion = NAPS_version
    geophysical_data_NAPS_insertion = NAPS_version
    training_data_NAPS_insertion = NAPS_version
else:
    observation_data_NAPS_insertion = 'No_NAPS'
    geophysical_data_NAPS_insertion = 'No_NAPS'
    training_data_NAPS_insertion = 'No_NAPS'

if Use_AOD_nan_values_filtered_Obs:
    observation_data_insertion = '_AODFiltered_{}'.format(AVD_GEO_version)
    geophysical_data_insertion = 'Filter_AOD_nan_obs'
    training_data_insertion = 'Filter_AOD_nan_obs'
else:
    observation_data_insertion = ''
    geophysical_data_insertion = 'Filled_AOD'
    training_data_insertion = 'Filled_AOD'

cfg = {
    'Pathway' : {

        #### Ground Observation Data, Geophysical Species Data, and Geophysical Biases Data ####
        "learning_objective": {
            "ground_observation_data_dir"      : f"/s.siyuan/s3/my-projects2/Projects/Daily_PM25_DL_2024/data/GroundBased_Observations/{AVD_OBS_version}/{observation_data_NAPS_insertion}/",
            "geophysical_species_data_dir"     : f"/s.siyuan/s3/my-projects2/Projects/Daily_PM25_DL_2024/data/Geophysical_PM25_Bias_Datasets/{AVD_OBS_version}/{geophysical_data_NAPS_insertion}/{AVD_GEO_version}/{geophysical_data_insertion}/{Pathbegin_YEAR}-{Pathend_YEAR}/",
            "geophysical_biases_data_dir"      : f"/s.siyuan/s3/my-projects2/Projects/Daily_PM25_DL_2024/data/Geophysical_PM25_Bias_Datasets/{AVD_OBS_version}/{geophysical_data_NAPS_insertion}/{AVD_GEO_version}/{geophysical_data_insertion}/{Pathbegin_YEAR}-{Pathend_YEAR}/",
            "ground_observation_data_infile"   : f"daily_PM25_data{observation_data_insertion}_{Pathbegin_YEAR}0101-{Pathend_YEAR}1231.npy",
            "geophysical_species_data_infile"  : "geophysical_PM25_datasets.npy",
            "geophysical_biases_data_infile"   : "geophysical_bias_datasets.npy",
            "observation_datapoints_threshold" : 0 # Minimum number of observation data points for each site, typically 100 * number of years
        },
        
        
        #### Training Datasets infiles
        "TrainingDataset": {
            "CNN_Training_infiles"          : f"/s.siyuan/s3/my-projects2/Projects/Daily_PM25_DL_2024/data/Training_Datasets/{AVD_OBS_version}/{training_data_NAPS_insertion}/{AVD_GEO_version}/{training_data_insertion}/2DCNN/{Pathbegin_YEAR}-{Pathend_YEAR}/CNN_training_datasets_{{}}_5x5_{Pathbegin_YEAR}0101-{Pathend_YEAR}1231.npy",
            "CNN3D_Training_infiles"        : f"/s.siyuan/s3/my-projects2/Projects/Daily_PM25_DL_2024/data/Training_Datasets/{AVD_OBS_version}/{training_data_NAPS_insertion}/{AVD_GEO_version}/{training_data_insertion}/3DCNN/{Pathbegin_YEAR}-{Pathend_YEAR}/CNN_training_datasets_{{}}_3x5x5_{Pathbegin_YEAR}0101-{Pathend_YEAR}1231.npy",
            "Transformer_Training_infiles"  : f"/s.siyuan/s3/my-projects2/Projects/Daily_PM25_DL_2024/data/Training_Datasets/{AVD_OBS_version}/{training_data_NAPS_insertion}/{AVD_GEO_version}/{training_data_insertion}/Transformer/{Pathbegin_YEAR}-{Pathend_YEAR}/Transformer_training_datasets_{{}}_{Pathbegin_YEAR}0101-{Pathend_YEAR}1231.npy",
            "CNN_Transformer_Training_infiles": f"/s.siyuan/s3/my-projects2/Projects/Daily_PM25_DL_2024/data/Training_Datasets/{AVD_OBS_version}/{training_data_NAPS_insertion}/{AVD_GEO_version}/{training_data_insertion}/2DTransformer/{Pathbegin_YEAR}-{Pathend_YEAR}/Transformer_training_datasets_{{}}_{Pathbegin_YEAR}0101-{Pathend_YEAR}1231.npy",
        },

        #### Validation Datasets outdirs
        "Results": {
            "csv_outdir"               : "/s.siyuan/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
            "model_outdir"             : "/s.siyuan/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
            "data_recording_outdir"    : "/s.siyuan/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
            "figure_outdir"            : "/s.siyuan/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
        },
        
        #### Map Data indir
        "MapData_indir" : {
            "MapData_Indir" : '/s.siyuan/s3/my-projects2/Projects/Daily_PM25_DL_2024/data/Input_Variables_MapData/',
            "MapData_fromRegionalComponentProject_Indir" : '/s.siyuan/s3/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/'
        },
        #### Other Data indir
        "Data_indir" : {
            "NA_Mask_indir"            : '/s.siyuan/s3/my-projects/mask/NA_Masks/Cropped_NA_Masks/', ## Use in Evaluation/iostream.py, North America region masks
            "mask_indir"               : "/s.siyuan/s3/my-projects/mask/Land_Ocean_Mask/", ## Use in Estimation_pkg/data_func.py, land ocean mask
            "LATLON_indir"             : "/s.siyuan/s3/my-projects2/Projects/Daily_PM25_DL_2024/data/",
        },
        'Config_outdir'   : "/s.siyuan/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
        ####
    },


    #########################################################################################################################################################
    #### Training and Validation Settings ####

    'Hyperparameters_Search_Validation-Settings' : {

        ### Hyperparameters Search and Validation Settings
        "Hyperparameters_Search_Validation_Switch": False,
        "HSV_Apply_wandb_sweep_Switch"            : True,
        "wandb_sweep_count"                       : 100,
        "Use_recorded_data_to_show_validation_results": False, # Default: False. Not applicable.


        "Training-Settings": {
            "Spatial_splitting_Switch"       : True,
            "Spatial_splitting_begindates"   : [20220101],
            "Spatial_splitting_enddates"     : [20231231],
            "Spatial_splitting_training_portion"  : 0.8,
            "Spatial_splitting_validation_portion": 0.2,


            "Temporal_splitting_Switch": False,
            "Temporal_splitting_training_begindates": [20220101],
            "Temporal_splitting_training_enddates": [20221231],
            "Temporal_splitting_validation_begindates": [20230101],
            "Temporal_splitting_validation_enddates": [20231231]
        }
    },

    #########################################################################################################################################################

    'Random-CrossValidation' : {

        "Random_CrossValidation_Switch": False,
        "Random_CV_Apply_wandb_sweep_Switch": False,
        "wandb_sweep_count_Random_CV": 100,
        "Use_recorded_data_to_show_validation_results": False,

        ##### Random Cross Validation Settings #####
        "Training-Settings": {
            "Random_CV_folds": 10,
            "Random_CV_training_begindates": [20190101,20200101,20210101,20220101,20230101],
            "Random_CV_training_enddates":  [20191231,20201231,20211231,20221231,20231231],
            "Random_CV_validation_begindates": [20190101,20220101],
            "Random_CV_validation_enddates": [20231231,20231231],
            "additional_validation_regions": [
                "Canada", "Contiguous United States", "Midwestern United States", "Northeastern United States",
                "Northern North America", "Northwestern United States", "Southern United States", "Southwestern United States"
            ],
        },

        ##### Random Cross Visualization Settings ####
        "Visualization_Settings": {
            "regression_plot_switch": True,
            "plot_begindates": [20190101],
            "plot_enddates": [20231231]
        },
        "Forced_Slope_Unity": {
            "ForcedSlopeUnity": True
        },
    },
    #########################################################################################################################################################

    'Spatial-CrossValidation' : {

        "Spatial_CrossValidation_Switch": False,
        "Spatial_CV_Apply_wandb_sweep_Switch": False,
        "wandb_sweep_count_Spatial_CV": 100,
        "Use_recorded_data_to_show_validation_results": False,

        ##### Spatial Cross Validation Settings #####
        "Training-Settings": {
            "Spatial_CV_folds": 10,
            "Spatial_CV_training_begindates": [20190101,20200101,20210101,20220101,20230101],
            "Spatial_CV_training_enddates": [20191231,20201231,20211231,20221231,20231231],
            "Spatial_CV_validation_begindates": [20190101,20220101],
            "Spatial_CV_validation_enddates": [20231231,20231231],
            "additional_validation_regions": [
                "Canada", "Contiguous United States", "Midwestern United States", "Northeastern United States",
                "Northern North America", "Northwestern United States", "Southern United States", "Southwestern United States"
            ],
        },

        ##### Spatial Cross Visualization Settings ####
        "Visualization_Settings": {
            "regression_plot_switch": True,
            "plot_begindates": [20190101],
            "plot_enddates": [20231231]
        },
        "Forced_Slope_Unity": {
            "ForcedSlopeUnity": True
        },

        ##### SHAP Analysis Settings ####
        "SHAP_Analysis_Settings": {
            "SHAP_Analysis_switch": False, ### Not controlled by Spatial_CrossValidation_Switch
            "SHAP_Analysis_Calculation_Switch": True,
            "SHAP_Analysis_visualization_Switch": True,
            "SHAP_Analysis_background_number": 2000,
            "SHAP_Analysis_test_number": 400,
            "SHAP_Analysis_plot_type": "beeswarm"
        }
    },
    #########################################################################################################################################################
    'BLISCO-CrossValidation' : {
        'BLISCO_CV_Switch': False,
        'Use_recorded_data_to_show_validation_results': False,
        
        ##### BLISCO Cross Validation Settings #####
        'Training-Settings': {
            'BLISCO_CV_folds': 10, # larger or at least equal to seeds number
            'BLISCO_CV_buffer_radius_km': [80],
            'BLISCO_CV_seeds_number': 10,
            'BLISCO_CV_training_begindates': [20190101,20200101,20210101,20220101,20230101],
            'BLISCO_CV_training_enddates': [20191231,20201231,20211231,20221231,20231231],
            'BLISCO_CV_validation_begindates': [20190101,20220101],
            'BLISCO_CV_validation_enddates': [20231231,20231231],
            'additional_validation_regions': [
                "Canada", "Contiguous United States", "Midwestern United States", "Northeastern United States",
                "Northern North America", "Northwestern United States", "Southern United States", "Southwestern United States"
            ],
        },
        
        ##### BLISCO Cross Visualization Settings ####
        'Visualization_Settings': {
            'Test_Train_Buffers_Distributions_plot_switch': False
        },
    },
    
    #########################################################################################################################################################
    'Temporal-CrossValidation' : {

        "Temporal_CrossValidation_Switch": False,
        "Use_recorded_data_to_show_validation_results": False,

        ##### Temporal Cross Validation Settings #####
        "Training-Settings": {
            "Temporal_CV_folds": 10,
            "Temporal_CV_training_begindates": [20190101,20200101,20210101,20220101,20230101],
            "Temporal_CV_training_enddates": [20191231,20201231,20211231,20221231,20231231],
            "Temporal_CV_validation_begindates": [20190101,20220101],
            "Temporal_CV_validation_enddates": [20231231,20231231],
            "additional_validation_regions": [
                "Canada", "Contiguous United States", "Midwestern United States", "Northeastern United States",
                "Northern North America", "Northwestern United States", "Southern United States", "Southwestern United States"
            ],
        },
        ##### Temporal Cross Visualization Settings ####
        "Visualization_Settings": {
            "regression_plot_switch": True,
            "plot_begindates": [20190101],
            "plot_enddates": [20231231]
        },
        "Forced_Slope_Unity": {
            "ForcedSlopeUnity": True
        },
    },
    #########################################################################################################################################################
    'Temporal-Buffer-Out-CrossValidation' : {

        "Temporal_Buffer_Out_CrossValidation_Switch": True,
        "Use_recorded_data_to_show_validation_results": False,

        ##### Temporal Cross Validation Settings #####
        "Training-Settings": {
            "Temporal_Buffer_Out_CV_folds": 10,
            "Temporal_Buffer_Out_CV_max_test_days": 1,
            "Temporal_Buffer_days": [0,1,3,5,7,15],
            "Temporal_Buffer_Out_CV_training_begindates": [20190101,20200101,20210101,20220101,20230101],
            "Temporal_Buffer_Out_CV_training_enddates": [20191231,20201231,20211231,20221231,20231231],
            "Temporal_Buffer_Out_CV_validation_begindates": [20190101,20220101],
            "Temporal_Buffer_Out_CV_validation_enddates": [20231231,20231231],
            "additional_validation_regions": [
                "Canada", "Contiguous United States", "Midwestern United States", "Northeastern United States",
                "Northern North America", "Northwestern United States", "Southern United States", "Southwestern United States"
            ],
        },
        ##### Temporal Cross Visualization Settings ####
        "Visualization_Settings": {
            "regression_plot_switch": True,
            "plot_begindates": [20190101],
            "plot_enddates": [20231231]
        },
        "Forced_Slope_Unity": {
            "ForcedSlopeUnity": True
        },
    },
    #########################################################################################################################################################
    'Estimation-Settings' : {
        'Estimation_Switch': False,
        'Train_model_Switch': True,
        'Map_estimation_Switch': False,
        'Estimation_visualization_Switch': False,

        ###### Training Settings ######
        'Training_Settings': {
            'Training_begin_dates': [20190101,20200101,20210101,20220101,20230101],
            'Training_end_dates': [20191231,20201231,20211231,20221231,20231231],
        },
        ###### Estimation Settings ######
        'Map_Estimation_Settings': {
            'Eatimation_Daily_Switch': True,
            'Estimation_trained_begin_dates': [201570101,20230101],
            'Estimation_trained_end_dates': [202151231,20231231],
            'Estimation_begindates': [[20150101],[20230101]],
            'Estimation_enddates': [[20151231],[20231231]],
            'Extent': [10.055,69.945,-169.945,-40.055],
            'Estimation_Area': 'NorthAmerica',

            'Save_Monthly_Average_Switch': True,
            'Save_Monthly_Average_begindates': [20220101],
            'Save_Monthly_Average_enddates': [20231231],

            'Save_Annual_Average_Switch': True,
            'Save_Annual_Average_beginyear': [2022],
            'Save_Annual_Average_endyear': [2023],
        },

        ###### Estimation Visualization Settings ######
        'Visualization_Settings': {
            'Map_Plot_Switch':True,
            'Daily_Plot_Switch': True,
            'Daily_Plot_begindates': [20190101,20200101,20210101,20220101,20230101],
            'Daily_Plot_enddates': [20191231,20201231,20211231,20221231,20231231],
            'Monthly_Plot_Switch': True,
            'Monthly_Plot_begindates': [20220101],
            'Monthly_Plot_enddates': [20231231],
            'Annual_Plot_Switch': True,
            'Annual_Plot_beginyears': [2022],
            'Annual_Plot_endyears': [2023],
            
        },

    },
    #########################################################################################################################################################


    'Training-Settings' : {
        "identity": {
            "version": "v1.0.0",
            "description": f"_{AVD_OBS_version}_{geophysical_data_insertion}_{geophysical_data_NAPS_insertion}_OneModelEachYear_RatioCalibration_BenchMark",
            "author": "Siyuan Shen",
            "email": "s.siyuan@wustl.edu",
            "runningdate": "{}-{}-{}".format(time.strftime("%Y"), time.strftime("%m"), time.strftime("%d"))
        },
        "learning-objective": {
            "species": "PM25",
            "normalize_type": "Gaussian", # Options: "Gaussian", "MinMax", "Robust",only applicable to learning objects normalize_species or normalize_bias
            "bias": False,
            "normalize_bias": False,
            "normalize_species": True,
            "absolute_species": False,
            "log_species": False,
        },
        "hyper-parameters": {
            "epoch": 71, # 2DCNN: 131; 3DCNN:71; Transformer:111
            "batchsize": 256,# 2DCNN: 256; 3DCNN:256; 3DCNN MoCE: 256; Transformer:32

            ##################################################################################################################
            ## This is for 2DCNN, 3DCNN, and transformer architectures. tSATPM25 must be included.
            ## If use MoCE architecture, the selected channels for each expert and gate are defined in Net_Architecture_config.py
            "channel_names": 
               
                [
                 "tSATAOD_Ratio_Calibration", "tSATPM25_Ratio_Calibration", #"eta",
                "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
                "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",#"NMVOC_anthro_emi",
                "Urban_Builtup_Lands", 
               "elevation", "Population", "lat", "lon", "sin_days", "cos_days",
               # "ocfire", "pm2p5fire", "mami", "tcfire",
               # 'Crop_Nat_Vege_Mos', 'Permanent_Wetlands', 'Croplands', 
               # 'major_roads', 'minor_roads', 'motorway', 'primary', 'secondary', 'trunk', 'unclassified', 'residential'
            ], 

            ##################################################################################################################
            ## This is for the CNN part of the CNN-Transformer architecture. tSATPM25 must be included.
            "CNN_Embedding_channel_names": [
                    "tSATAOD", "tSATPM25", #"eta",

                     "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",
                     "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                     "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                     "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",
                     "Urban_Builtup_Lands", "elevation", "Population", "lat", "lon",
            ], 
            ## This is for the transformer part of the CNN-Transformer architecture. tSATPM25 must be included.
            "Transformer_Embedding_channel_names": [
                    "tSATAOD", "tSATPM25", #"eta",
                    "GC_PM25",  "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",
                    "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                    "sin_days", "cos_days"
            ], 

            ##################################################################################################################
            "training_data_normalization_type": "Gaussian", # Options: "Gaussian", "MinMax","Robust", applicable to training datasets
        },

        "Loss-Functions": {
            "Regression_loss_type": "MSE",
            "Classification_loss_type": "CrossEntropyLoss",
            "GeoMSE": {
                "GeoMSE_Lamba1_Penalty1": 10.0,
                "GeoMSE_Lamba1_Penalty2": 8.0,
                "GeoMSE_Gamma": 2.5
            },
            "MultiHead_Loss": {
                "ResNet_MultiHeadNet_regression_loss_coefficient": 1,
                "ResNet_MultiHeadNet_classfication_loss_coefficient": 1
            }
        },

        "optimizer": {
            "Adam": {
                "Settings": True,
                "beta0": 0.9,
                "beta1": 0.999,
                "eps": 1e-8
            }
        },

        "learning_rate": {
            "learning_rate0": 0.0003, # 2D CNN: 0.001; 3DCNN:0.0001, transformer: 0.001
            "ExponentialLR": {
                "Settings": False,
                "gamma": 0.9
            },
            "CosineAnnealingLR": {
                "Settings": True,
                "T_max": 10,
                "eta_min": 1e-8
            },
            "CosineAnnealingRestartsLR": {
                "Settings": False,
                "T_0": 10,
                "T_mult": 2,
                "eta_min": 0
            }
        },

        "activation_func": {
            "activation_func_name": "relu", ## This is for initialization only
            "ReLU": {"Settings": False},
            "Tanh": {"Settings": True},
            "GeLU": {"Settings": False},
            "Sigmoid": {"Settings": False},
            "Mish": {"Settings": False},
            "ELU": {"Settings": False}
        }
    }
}