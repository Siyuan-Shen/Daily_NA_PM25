
# config_network_architecture.py

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
            "CovLayer_padding_mode_3D": "replicate",
            "Pooling_padding_mode_3D": "replicate",
            "ResNet3D": {
                "Settings": True,
                "Blocks": "BasicBlock",
                "blocks_num": [1, 1, 1, 1],
                "output_channels": [128, 256, 512, 1024],
                "Pooling_layer_type_3D": "MaxPooling3d",
                "Depth": 3
            }
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
