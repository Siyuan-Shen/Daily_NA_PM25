#########################################################################################################################################################
Pathbegin_YEAR = 2018
Pathend_YEAR = 2023
AVD_OBS_version = 'AVD_d20240814'
AVD_GEO_version = 'vAOD20240322vGEO20241212'

cfg = {
    'Pathway' : {

        #### Ground Observation Data, Geophysical Species Data, and Geophysical Biases Data ####
        "learning_objective": {
            "ground_observation_data_dir"      : f"/my-projects2/Projects/Daily_PM25_DL_2024/data/GroundBased_Observations/{AVD_OBS_version}/",
            "geophysical_species_data_dir"     : f"/my-projects2/Projects/Daily_PM25_DL_2024/data/Geophysical_PM25_Bias_Datasets/{AVD_OBS_version}/{AVD_GEO_version}/{Pathbegin_YEAR}-{Pathend_YEAR}/",
            "geophysical_biases_data_dir"      : f"/my-projects2/Projects/Daily_PM25_DL_2024/data/Geophysical_PM25_Bias_Datasets/{AVD_OBS_version}/{AVD_GEO_version}/{Pathbegin_YEAR}-{Pathend_YEAR}/",
            "ground_observation_data_infile"   : f"daily_PM25_data_{Pathbegin_YEAR}0101-{Pathend_YEAR}1231.npy",
            "geophysical_species_data_infile"  : "geophysical_PM25_datasets.npy",
            "geophysical_biases_data_infile"   : "geophysical_bias_datasets.npy",
            "observation_datapoints_threshold" : 0 # Minimum number of observation data points for each site, typically 100 * number of years
        },

        #### Training Datasets infiles
        "TrainingDataset": {
            "CNN_Training_infiles"          : f"/my-projects2/Projects/Daily_PM25_DL_2024/data/Training_Datasets/{AVD_OBS_version}/{AVD_GEO_version}/2DCNN/{Pathbegin_YEAR}-{Pathend_YEAR}/CNN_training_datasets_{{}}_11x11_{Pathbegin_YEAR}0101-{Pathend_YEAR}1231.npy",
            "CNN3D_Training_infiles"        : f"/my-projects2/Projects/Daily_PM25_DL_2024/data/Training_Datasets/{AVD_OBS_version}/{AVD_GEO_version}/3DCNN/{Pathbegin_YEAR}-{Pathend_YEAR}/CNN_training_datasets_{{}}_3x11x11_{Pathbegin_YEAR}0101-{Pathend_YEAR}1231.npy",
            "Transformer_Training_infiles"  : f"/my-projects2/Projects/Daily_PM25_DL_2024/data/Training_Datasets/{AVD_OBS_version}/{AVD_GEO_version}/Transformer/{Pathbegin_YEAR}-{Pathend_YEAR}/Transformer_training_datasets_{{}}_{Pathbegin_YEAR}0101-{Pathend_YEAR}1231.npy",
        },

        #### Validation Datasets outdirs
        "Results": {
            "csv_outdir"               : "/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
            "model_outdir"             : "/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
            "data_recording_outdir"    : "/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
            "figure_outdir"            : "/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
        },

        #### Other Data indir
        "Data_indir" : {
            "mask_indir"               : "/my-projects/mask/Land_Ocean_Mask/", ## Use in Estimation_pkg/data_func.py
            "LATLON_indir"             : "/my-projects2/Projects/Daily_PM25_DL_2024/data/",
        },
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

    'Spatial-CrossValidation' : {

        "Spatial_CrossValidation_Switch": True,
        "Spatial_CV_Apply_wandb_sweep_Switch": False,
        "wandb_sweep_count_Spatial_CV": 100,
        "Use_recorded_data_to_show_validation_results": False,

        ##### Spatial Cross Validation Settings #####
        "Training-Settings": {
            "Spatial_CV_folds": 10,
            "Spatial_CV_training_begindates": [20220101],
            "Spatial_CV_training_enddates": [20231231],
            "Spatial_CV_validation_begindates": [20220101],
            "Spatial_CV_validation_enddates": [20231231],
            "additional_validation_regions": [
                "Canada", "Contiguous United States", "Midwestern United States", "Northeastern United States",
                "Northern North America", "Northwestern United States", "Southern United States", "Southwestern United States"
            ],
        },

        ##### Spatial Cross Visualization Settings ####
        "Visualization_Settings": {
            "regression_plot_switch": True,
            "plot_begindates": [20220101,20220101],
            "plot_enddates": [20231231,20221231]
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
    'Estimation-Settings' : {
        'Estimation_Switch': False,
        'Train_model_Switch': True,
        'Map_estimation_Switch': True,
        'Estimation_visualization_Switch': True,

        ###### Training Settings ######
        'Training_Settings': {
            'Training_begin_dates': [20220101],
            'Training_end_dates': [20231231],
        },
        ###### Estimation Settings ######
        'Map_Estimation_Settings': {
            'Eatimation_Daily_Switch': True,
            'Estimation_trained_begin_dates': [20220101],
            'Estimation_trained_end_dates': [20231231],
            'Estimation_begindates': [20220101],
            'Estimation_enddates': [20231231],
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
            'Daily_Plot_begindates': [20220101],
            'Daily_Plot_enddates': [20231231],
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
            "version": "v0.1.0",
            "description": "_Transformer_baseline1",
            "author": "Siyuan Shen",
            "email": "s.siyuan@wustl.edu",
            "runningdate": "2025-08-08"
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
            "epoch": 131, # 3DCNN:131; Transformer:131
            "batchsize": 32,# 3DCNN:128; Transformer:32
            "channel_names": [
                "eta", "tSATAOD", "tSATPM25", 
                "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
                "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                "BC_anthro_emi", "NMVOC_anthro_emi", "DST_offline_emi", "SSLT_offline_emi",
                "Urban_Builtup_Lands", "elevation", "Population", "lat", "lon", "sin_days", "cos_days"
            ],
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
            "learning_rate0": 0.001,
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
            "activation_func_name": "relu",
            "ReLU": {"Settings": False},
            "Tanh": {"Settings": True},
            "GeLU": {"Settings": False},
            "Sigmoid": {"Settings": False},
            "Mish": {"Settings": False},
            "ELU": {"Settings": False}
        }
    }
}