
# config_network_architecture.py

cfg  = {
    'network-architecture':{
        "Apply_CNN_architecture": False,
        "Apply_3D_CNN_architecture": True,
    
        "CNN-architecture": {
            "CovLayer_padding_mode": "reflect",
            "Pooling_padding_mode": "reflect",

            "ResNet": {
                "Settings": True,
                "Blocks": "BasicBlock",
                "blocks_num": [2, 2, 2, 2],
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
        }
    }
}
