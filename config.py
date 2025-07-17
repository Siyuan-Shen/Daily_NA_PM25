#########################################################################################################################################################

cfg = {
    'Pathway' : {
        #### Ground Observation Data, Geophysical Species Data, and Geophysical Biases Data ####
        "learning_objective": {
            "ground_observation_data_dir"      : "/my-projects2/Projects/Daily_PM25_DL_2024/data/GroundBased_Observations/AVD_d20240814/",
            "geophysical_species_data_dir"     : "/my-projects2/Projects/Daily_PM25_DL_2024/data/Geophysical_PM25_Bias_Datasets/AVD_d20240814/vAOD20240322vGEO20241212/2022-2023/",
            "geophysical_biases_data_dir"      : "/my-projects2/Projects/Daily_PM25_DL_2024/data/Geophysical_PM25_Bias_Datasets/AVD_d20240814/vAOD20240322vGEO20241212/2022-2023/",
            "ground_observation_data_infile"   : "daily_PM25_data_20220101-20231231.npy",
            "geophysical_species_data_infile"  : "geophysical_PM25_datasets.npy",
            "geophysical_biases_data_infile"   : "geophysical_bias_datasets.npy",
            "observation_datapoints_threshold" : 200 # Minimum number of observation data points for each site, typically 100 * number of years
        },

        #### Training Datasets infiles
        "TrainingDataset": {
            "CNN_Training_infiles"    : "/my-projects2/Projects/Daily_PM25_DL_2024/data/Training_Datasets/AVD_d20240814/vAOD20240322vGEO20241212/2DCNN/2022-2023/CNN_training_datasets_{}_11x11_20220101-20231231.npy",
            "CNN3D_Training_infiles"  : "/my-projects2/Projects/Daily_PM25_DL_2024/data/Training_Datasets/AVD_d20240814/vAOD20240322vGEO20241212/3DCNN/2022-2023/CNN_training_datasets_{}_3x11x11_20220101-20231231.npy"
        },

        #### Validation Datasets outdirs
        "Results": {
            "csv_outdir"               : "/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
            "model_outdir"             : "/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/",
            "data_recording_outdir"    : "/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/"
        },
    },


    #########################################################################################################################################################
    #### Training and Validation Settings ####

    'Hyperparameters_Search_Validation-Settings' : {

        ### Hyperparameters Search and Validation Settings
        "Hyperparameters_Search_Validation_Switch": True,
        "HSV_Apply_wandb_sweep_Switch"            : True,
        "wandb_sweep_count"                       : 100,
        "Use_recorded_data_to_show_validation_results": False,


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

        "Spatial_CrossValidation_Switch": False,
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
            "plot_begindates": [20220101],
            "plot_enddates": [20231231]
        },
        "Forced_Slope_Unity": {
            "ForcedSlopeUnity": True
        },

        ##### SHAP Analysis Settings ####
        "SHAP_Analysis_Settings": {
            "SHAP_Analysis_switch": False,
            "SHAP_Analysis_Calculation_Switch": True,
            "SHAP_Analysis_visualization_Switch": True,
            "SHAP_Analysis_background_number": 2000,
            "SHAP_Analysis_test_number": 400,
            "SHAP_Analysis_plot_type": "beeswarm"
        }
    },
    #########################################################################################################################################################


    'Training-Settings' : {
        "identity": {
            "version": "v0.1.0",
            "description": "_3DCNN_Model_Structure_Searching",
            "author": "Siyuan Shen",
            "email": "s.siyuan@wustl.edu",
            "runningdate": "2025-07-15"
        },
        "learning-objective": {
            "species": "PM25",
            "bias": False,
            "normalize_bias": False,
            "normalize_species": True,
            "absolute_species": False
        },
        "hyper-parameters": {
            "epoch": 131,
            "batchsize": 512,
            "channel_names": [
                "eta", "tSATAOD", "tSATPM25", "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_BC", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",
                "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                "BC_anthro_emi", "NMVOC_anthro_emi", "DST_offline_emi", "SSLT_offline_emi",
                "Urban_Builtup_Lands", "elevation", "Population", "lat", "lon", "sin_days", "cos_days"
            ]
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
            "learning_rate0": 0.0001,
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