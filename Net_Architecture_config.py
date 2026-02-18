
# config_network_architecture.py
##
# If want to change the network architecture, please change the settings here.
##
# If want to add new architecture, please add the settings here and modify the model structure code accordingly.
# Step 1: Add the settings in this config file.
# Step 2: import the settings in the model structure code (e.g., Model_Structure_pkg/utils.py).
# Step 3: Modify the model structure code to implement the new architecture according to the settings. 
# Step 4: Add the new architecture option in the wandb_config.py file for logging and sweeping.
# Step 5: Change the code in Training_pkg/TrainingModule.py to use the new architecture.
# Step 6: Test the new architecture with a small dataset to ensure it works correctly.
cfg  = {
    'network-architecture':{
        "Apply_CNN_architecture": False,
        "Apply_3D_CNN_architecture": True,
        "Apply_Transformer_architecture": False,
        "Apply_CNN_Transformer_architecture": False,

        "CNN-architecture": {
            "CovLayer_padding_mode": "reflect",
            "Pooling_padding_mode": "reflect",

            "ResNet": {
                "Settings": True,
                "Blocks": "BasicBlock",
                "blocks_num": [1, 1, 1, 1],
                "Pooling_layer_type": "MaxPooling2d"
            },

            "TwoCombineModels": {
                "Settings": False,
                "Variable": "GeoPM25",
                "threshold": 4.0
            },

            "ResNet_MLP": {
                "Settings": False,
                "Blocks": "Bottleneck",
                "blocks_num": [1, 1, 1, 1]
            },

            "ResNet_Classification": {
                "Settings": False,
                "Blocks": "BasicBlock",
                "blocks_num": [1, 1, 0, 1],
                "left_bin": -5.0,
                "right_bin": 5.0,
                "bins_number": 101
            },

            "ResNet_MultiHeadNet": {
                "Settings": False,
                "Blocks": "BasicBlock",
                "blocks_num": [1, 1, 0, 1],
                "left_bin": -5.0,
                "right_bin": 5.0,
                "bins_number": 101,
                "regression_portion": 0.5,
                "classifcation_portion": 0.5
            },

            "LateFusion": {
                "Settings": False,
                "Blocks": "Bottleneck",
                "blocks_num": [1, 1, 1, 1],
                "initial_channels": [
                    "AOD", "EtaAOD_Bias", "EtaCoastal", "EtaMixing", "EtaSGAOD_Bias", "EtaSGTOPO_Bias", "GeoPM25", "ETA",
                    "GeoNH4", "GeoNIT", "GeoSO4", "GeoBC", "GeoOM", "GeoDUST", "GeoSS",
                    "NA_CNN_PM25", "GC_PM25", "GC_NH4", "GC_SO4", "GC_NIT", "GC_SOA", "GC_OC", "GC_OM", "GC_BC", "GC_DST", "GC_SSLT",
                    "NH3_anthro_emi", "NO_anthro_emi", "SO2_anthro_emi", "DST_offline_emi", "SSLT_offline_emi",
                    "RH", "T2M", "U10M", "V10M", "PRECTOT", "PBLH", "Urban_Builtup_Lands"
                ],
                "LateFusion_channels": ["Lat", "Lon", "elevation", "Month_of_Year"]
            },

            "MultiHeadLateFusion": {
                "Settings": False,
                "Blocks": "BasicBlock",
                "blocks_num": [1, 1, 1, 1],
                "initial_channels": [
                    "EtaAOD_Bias", "EtaCoastal", "EtaMixing", "EtaSGAOD_Bias", "EtaSGTOPO_Bias", "AOD", "ETA", "GeoPM25",
                    "GeoNH4", "GeoNIT", "GeoSO4", "GeoBC", "GeoOM", "GeoDUST", "GeoSS",
                    "NA_CNN_PM25", "GC_PM25", "GC_NH4", "GC_SO4", "GC_NIT", "GC_SOA", "GC_OC", "GC_OM", "GC_BC", "GC_DST", "GC_SSLT",
                    "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "DST_offline_emi", "SSLT_offline_emi",
                    "PBLH", "RH", "T2M", "U10M", "V10M", "PRECTOT", "Urban_Builtup_Lands"
                ],
                "LateFusion_channels": ["Lat", "Lon", "elevation", "Month_of_Year"],
                "left_bin": -10.0,
                "right_bin": 10.0,
                "bins_number": 201,
                "regression_portion": 0.5,
                "classifcation_portion": 0.5
            }
        },

        "CNN3D-architecture": {
            "CovLayer_padding_mode_3D": "replicate", # 'replicate' or 'reflect' or 'zeros' or 'circular'
            "Pooling_padding_mode_3D": "replicate", # 'replicate' or 'reflect' or 'constant' or 'circular'
            "ResNet3D": {
                "Settings": True, ## Default is True, do not turn it off even if you use MoE
                "Blocks": "BasicBlock", ## for 3D CNN, MoE, and MoCE architectures (both base and side models)
                ## blocks_num, output_channels, pooling_kernel_size are used in ResNet3D, MoE, and MoCE base model
                "blocks_num": [1, 1, 1, 1],
                "output_channels": [128, 256, 512, 1024],
                "pooling_layer_switch": True,
                "pooling_kernel_size": (1,3,3),
                "pooling_layer_type_3D": "MaxPooling3d",
                "Depth": 3
            },
            
        "MoE-architecture": {
            "Settings": False, ## Turn on MoE architecture, and also turn on ResNet3D Settings. Only one of MoE or MoCE can be True.
            "num_experts": 4,
            "gating_hidden_size": 128,
            "selected_channels": ["tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",
                                 #"ocfire", "pm2p5fire", "mami", "tcfire",
                                 "PBLH", "RH", "V10M", "U10M",], # "PS",  "PRECTOT", "T2M",],
            },
        
        'MoCE-architecture': { # Mixture of Channels Experts, the channels are decided here instead of the config.py
            'Settings': True, ## Turn on MoCE architecture, and also turn on ResNet. Only one of MoE or MoCE can be True.
            'num_experts': 4,
            'gating_hidden_size': 128,
            'gate_selected_channels': ["tSATAOD_Ratio_Calibration", "tSATPM25_Ratio_Calibration","lat", "lon", "sin_days", "cos_days",
                                "ocfire", "pm2p5fire", "mami", "tcfire",
                                 "PBLH", "RH",  "V10M", "U10M", "PS","PRECTOT", "T2M", ],
            
            'base_model_channels' : ["tSATAOD_Ratio_Calibration", "tSATPM25_Ratio_Calibration", 
                "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",
                "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",
                "Urban_Builtup_Lands", 
               "elevation", "Population", "lat", "lon", "sin_days", "cos_days",],
            
            "side_blocks_num": [1, 1, 1, 1],
            "side_output_channels": [128, 256, 512, 1024],
            "side_pooling_kernel_switch": True,
            "side_pooling_layer_type_3D": "MaxPooling3d",
            "side_pooling_kernel_size": (1,3,3),
            ## Define the selected channels for each expert (length should be equal to num_experts)
            'side_experts_channels_list': [ # len = num_experts - 1
                [  "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
                    "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                    "lat", "lon", "sin_days", "cos_days", ],
                ["tSATAOD_Ratio_Calibration", "tSATPM25_Ratio_Calibration","lat", "lon", "sin_days", "cos_days", "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                         "elevation", "Population", ],
                [ "tSATAOD_Ratio_Calibration", "tSATPM25_Ratio_Calibration", "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days",
                        "ocfire", "pm2p5fire", "mami", "tcfire",],
            ],
            },   
        },


        "Transformer-architecture": {
            "Settings": True,
            "trg_dim": 1,  # Dimension of the target features; default is 1
            "d_model": 64,  # Dimension of the model (hidden size); default is 64; Make d_model divisible by n_head.
            "n_head": 4,  # Number of attention heads; default is 8
            "ffn_hidden": 64,  # Dimension of the feed-forward network hidden layer
            "num_layers": 6,  # Number of encoder/decoder layers; default is 6
            "max_len": 30,  # Maximum length of the input sequence; default is 30. This is for the range that the model calculate the loss.
            "spin_up_len": 1,  # Spin-up length for the model; default is 7. This is for the range that the model DO NOT calculate the loss.
            "drop_prob": 0.01,  # Dropout probability; default is 0.1
        },

        "CNN-Transformer-architecture": {
            "Settings": True,
            "CovLayer_padding_mode": "reflect",
            "Pooling_padding_mode": "reflect",
            "ResNet": {
                "Blocks": "BasicBlock",
                "blocks_num": [1, 1, 1, 1],
                "output_channels": [64, 128, 256, 512],
                "Pooling_layer_type": "MaxPooling2d"
            },
            "trg_dim": 1,  # Dimension of the target features; default is 1
            "d_model": 512,  # Dimension of the model (hidden size); default is 512; Make d_model divisible by n_head.
            "n_head": 4,  # Number of attention heads; default is 8
            "ffn_hidden": 64,  # Dimension of the feed-forward network hidden layer
            "num_layers": 6,  # Number of encoder/decoder layers; default is 6
            "max_len": 30,  # Maximum length of the input sequence; default is 30. This is for the range that the model calculate the loss.
            "spin_up_len": 1,  # Spin-up length for the model; default is 7. This is for the range that the model DO NOT calculate the loss.
            "drop_prob": 0.01,  # Dropout probability; default is 0.1
        }
    }
}

"""
Four experts side channels:
[  "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
                    "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                    "lat", "lon", "sin_days", "cos_days", ],
                ["tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days", "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                         "elevation", "Population", ],
                [ "tSATAOD", "tSATPM25", "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days",
                        "ocfire", "pm2p5fire", "mami", "tcfire",],
                        
Six experts side channels:

[  "tSATAOD", "tSATPM25",
                     "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",
                    "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS",
                     "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                    "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",
                    "Urban_Builtup_Lands", 'Grasslands','Evergreen-Broadleaf-Forests',
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days", ],
                ["tSATPM25", "tSATAOD", "lat", "lon", "sin_days", "cos_days", "ocfire", "pm2p5fire", "mami", "tcfire",],
                ["tSATAOD", "tSATPM25",
                        "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                         "elevation", "Population", "lat", "lon", "sin_days", "cos_days",],
                ["tSATAOD","PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS",
                     "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                    "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",
                    "Urban_Builtup_Lands",
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days", "ocfire", "pm2p5fire", "mami", "tcfire", ],
                ["tSATAOD","PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS",
                     "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                    "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",
                    "Urban_Builtup_Lands", 'Grasslands','Evergreen-Broadleaf-Forests',
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days", "ocfire", "pm2p5fire", "mami", "tcfire", ]
"""