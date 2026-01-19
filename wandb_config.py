import wandb
import os
import itertools

from Training_pkg.utils import description,Apply_Transformer_architecture,Apply_CNN_Transformer_architecture,Apply_CNN_architecture,Apply_3D_CNN_architecture, version, learning_rate0, epoch, batchsize,ResCNN3D_blocks_num,ResCNN3D_output_channels,ResNet_blocks_num
from Model_Structure_pkg.utils import *
def wandb_run_config():
    if Apply_CNN_architecture:
        run_config = {
            "learning_rate0": learning_rate0,  # Replace with your learning rate variable
            "architecture": "CNN",
            "epoch": epoch,
            "batch_size": batchsize,  # Replace with your batch size variable
            "ResNet_blocks_num": ResNet_blocks_num,
        }
    if Apply_3D_CNN_architecture:
        
        run_config = {
            "learning_rate0": learning_rate0,  # Replace with your learning rate variable
            "architecture": "3DCNN",
            "epoch": epoch,
            "batch_size": batchsize,  # Replace with your batch size variable
            "CovLayer_padding_mode_3D": CovLayer_padding_mode_3D,
            "Pooling_padding_mode_3D": Pooling_padding_mode_3D,
            "ResCNN3D_blocks_num": ResCNN3D_blocks_num,
            "ResCNN3D_output_channels": ResCNN3D_output_channels,
            'pooling_layer_switch': pooling_layer_switch,
            'pooling_layer_type_3D': pooling_layer_type_3D,
            'ResCNN3D_pooling_kernel_size':ResCNN3D_pooling_kernel_size, 
        }
        if MoE_Settings:
            run_config["MoE_num_experts"] = MoE_num_experts
            run_config["MoE_gating_hidden_size"] = MoE_gating_hidden_size
            run_config["MoE_selected_channels"] = MoE_selected_channels
        
        if MoCE_Settings:
            run_config["MoCE_num_experts"] = MoCE_num_experts
            run_config["MoCE_gating_hidden_size"] = MoCE_gating_hidden_size
            run_config["MoCE_selected_channels_index_for_gate"] = MoCE_selected_channels_index_for_gate
            run_config["MoCE_base_model_channels"] = MoCE_base_model_channels
            run_config["MoCE_side_blocks_num"] = MoCE_side_blocks_num
            run_config["MoCE_side_output_channels"] = MoCE_side_output_channels
            run_config["MoCE_side_pooling_kernel_switch"] = MoCE_side_pooling_kernel_switch
            run_config["MoCE_side_pooling_layer_type_3D"] = MoCE_side_pooling_layer_type_3D
            run_config["MoCE_side_pooling_kernel_size"] = MoCE_side_pooling_kernel_size
            run_config["MoCE_side_experts_channels_list"] = MoCE_side_experts_channels_list
            
    if Apply_Transformer_architecture:
        run_config = {
            "learning_rate0": learning_rate0,
            "architecture": "Transformer",
            "epoch": epoch,
            "batch_size": batchsize,
            'd_model': Transformer_d_model,
            'n_head': Transformer_n_head,
            'ffn_hidden': Transformer_ffn_hidden,
            'num_layers': Transformer_num_layers,
            'max_len': Transformer_max_len,
            'spin_up_len': Transformer_spin_up_len,
            'drop_prob': Transformer_drop_prob
            
        }

    if Apply_CNN_Transformer_architecture:
        run_config = {
            "learning_rate0": learning_rate0,
            "architecture": "CNN_Transformer",
            "epoch": epoch,
            "batch_size": batchsize,
            "CNN_blocks_num": CNN_Transformer_ResNet_blocks_num,
            "CNN_output_channels": CNN_Transformer_ResNet_output_channels,
            'd_model': CNN_Transformer_d_model,
            'n_head': CNN_Transformer_n_head,
            'ffn_hidden': CNN_Transformer_ffn_hidden,
            'num_layers': CNN_Transformer_num_layers,
            'max_len': CNN_Transformer_max_len,
            'spin_up_len': CNN_Transformer_spin_up_len,
            'drop_prob': CNN_Transformer_drop_prob
        }
    return run_config

def wandb_initialize(run_config,rank,sweep_mode=None,sweep_id=None):

    # Initialize a new wandb run
    wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="ACAG-NorthAmericaDailyPM25",
            # Set the wandb project where this run will be logged.
            project=version,
            # Track hyperparameters and run metadata.
            ## get random string for the name
            name= '{}{}'.format(wandb.util.generate_id(),description),
            config=run_config,
            mode="offline" if rank != 0 else "online",
            group=sweep_id if sweep_mode else None  # Group runs under a sweep if sweep_mode is provided
        )
    return 

def init_get_sweep_config():
     # Initialize a new wandb run
    wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="ACAG-NorthAmericaDailyPM25",
            # Set the wandb project where this run will be logged.
            project=version,
            # Track hyperparameters and run metadata.
            ## get random string for the name
            
        )
    temp_sweep_config = dict(wandb.config)
    wandb.finish()  # Finish the run to avoid memory leaks
    return temp_sweep_config


### This is for the sweep for MoCE channel pool selection.
MOCE_CHANNEL_POOL =[
                    [  "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
                    "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                    "lat", "lon", "sin_days", "cos_days", ],
                    ["tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days", "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                            "elevation", "Population", ],
                    [ "tSATAOD", "tSATPM25", "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                        "elevation", "Population", "lat", "lon", "sin_days", "cos_days",
                            "ocfire", "pm2p5fire", "mami", "tcfire",],] ### Put the comments outside, otherwise the length will be real length + 1
                            
""" 
                    [  "tSATAOD", "tSATPM25", #"eta",
                     "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
                    "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                     "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                    "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",#"NMVOC_anthro_emi",
                    "Urban_Builtup_Lands", 'Grasslands','Evergreen-Broadleaf-Forests',
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days", ],  
                     [  "tSATAOD", "tSATPM25", #"eta",
                     "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
                    "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                     "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                    "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",#"NMVOC_anthro_emi",
                    "Urban_Builtup_Lands", 
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days", ],  
                    [  "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
                    "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                    "lat", "lon", "sin_days", "cos_days", ],
                    ["tSATPM25", "tSATAOD", "lat", "lon", "sin_days", "cos_days", "ocfire", "pm2p5fire", "mami", "tcfire",],
                    [ "tSATPM25", "tSATAOD", "lat", "lon", "sin_days", "cos_days", "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", ], 
                    
                    [ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days","NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                         "BC_anthro_emi", "Urban_Builtup_Lands", "Population", ], 
                    ["tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days", "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                         "elevation", "Population", ],
                    
                    [ "tSATAOD", 
                        "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
                        "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                        "Urban_Builtup_Lands", 'Grasslands','Evergreen-Broadleaf-Forests',
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days",],
                    
                     ["tSATAOD", "tSATPM25",
                        "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                         "elevation", "Population", "lat", "lon", "sin_days", "cos_days",],
                     
                    [ "tSATAOD", "tSATPM25", "Urban_Builtup_Lands",'Grasslands','Evergreen-Broadleaf-Forests',
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days",
                        "ocfire", "pm2p5fire", "mami", "tcfire",],
                    
                    ["tSATAOD","PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                     "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                    "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",#"NMVOC_anthro_emi",
                    "Urban_Builtup_Lands", 
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days", "ocfire", "pm2p5fire", "mami", "tcfire", ],
                    
                    ["tSATAOD","PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                     "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                    "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",#"NMVOC_anthro_emi",
                    "Urban_Builtup_Lands", 'Grasslands','Evergreen-Broadleaf-Forests',
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days", "ocfire", "pm2p5fire", "mami", "tcfire", ]
"""

def wandb_sweep_config():
    # Define the sweep configuration
    if Apply_CNN_architecture:
        sweep_configuration = {
            'name':'HSV_2DCNN_Sweep_Normalized_Speices',  # Name of the sweep
            'entity': 'ACAG-NorthAmericaDailyPM25',  # Your wandb entity (team name)
            'project': version,  # Your wandb project name
            'method': "random",  # 'grid', 'random', 'bayes'
            'metric': {
                'name': 'test_R2',
                'goal': 'maximize'  # 'minimize' or 'maximize'
            },
            'parameters': {
                'learning_rate0': {
                    'values': [ 0.01,0.001,0.0001,0.1]
                },
                'batch_size': {
                    'values': [32,64,128,256,512]
                },
                'epoch':{
                    'values': [31,51,71,91,111,131]
                },
                'channel_to_exclude': {
                    'values': [['']]#['GC_PM25'],['GC_SO4'],['GC_NH4'],['GC_NIT'],['GC_BC'],['GC_OM'],['GC_SOA'],['GC_DST'],['GC_SSLT'],
                                #                     ['PBLH'],['RH'],['PRECTOT'],['T2M'],['V10M'],['U10M'],['PS'],#'USTAR',
                                #                      ['NH3_anthro_emi'],['SO2_anthro_emi'],['NO_anthro_emi'],['OC_anthro_emi'],['BC_anthro_emi'],['NMVOC_anthro_emi'],
                                #                      ['DST_offline_emi'],['SSLT_offline_emi'],
                                #                    ['Urban_Builtup_Lands'], #  'Crop_Nat_Vege_Mos','Permanent_Wetlands','Croplands',
                                                    #  'major_roads','minor_roads','motorway',
                                #                      ['elevation'],['Population'],
                                 #                     ['lat'],['lon'],['sin_days'],['cos_days']]                
                },
                'channel_to_add' : {
                    'values': [[]]
                    },

                'ResNet_blocks_num': {
                    'values': [[2,2,2,2],[1,1,1,1],[3,3,3,3],[4,4,4,4]]
                },
            }
        }
    if Apply_3D_CNN_architecture:
        sweep_configuration = {
            'name':'HSV_3DCNN_MoCE_Sweep_Normalized_Speices_fixed_channels_Net_Structure',  # Name of the sweep
            'entity': 'ACAG-NorthAmericaDailyPM25',  # Your wandb entity (team name)
            'project': version,  # Your wandb project name
            'method': "random",  # 'grid', 'random', 'bayes'
            'metric': {
                'name': 'test_R2',
                'goal': 'maximize'  # 'minimize' or 'maximize'
            },
            'parameters': {
                'learning_rate0': {
                    'values': [ 0.0003]
                },
                'batch_size': {
                    'values': [256]
                },
                'epoch':{
                    'values': [71,91,111]
                },
                'pooling_layer_switch': {
                    'values': [True,False]
                },
                'pooling_layer_type_3D': {
                    'values': ['MaxPooling3d','AvgPooling3d'] #'AvgPooling3d', 'MaxPooling3d'
                },
                
                'ResCNN3D_pooling_kernel_size': {
                    'values': [(1,3,3)]
                },
                
                'CovLayer_padding_mode_3D': {
                    'values': ['replicate','reflect', 'zeros', 'circular']#, 'reflect', 'zeros', 'circular'] # 'replicate' or 'reflect' or 'zeros' or 'circular'
                },
                'Pooling_padding_mode_3D': {
                    'values': ['replicate','reflect', 'constant', 'circular']#, 'reflect', 'constant', 'circular'] # 'replicate' or 'reflect' or 'constant' or 'circular'
                },
                 
                'ResCNN3D_blocks_num': {
                    'values': [[1,1,1,1],]
                },

                'ResCNN3D_output_channels': {
                    'values': [[128,256,512,1024],]  # Example values for output channels
                },
                
                'channel_to_exclude': {
                    'values': [[]] #["ocfire", "pm2p5fire", "mami", "tcfire",
               # 'Crop_Nat_Vege_Mos', 'Permanent_Wetlands', 'Croplands', 
                #'major_roads', 'minor_roads', 'motorway', 'primary', 'secondary', 'trunk', 'unclassified', 'residential'],
                     #          ["ocfire", "pm2p5fire", "mami", "tcfire",],
                       #        ['Crop_Nat_Vege_Mos', 'Permanent_Wetlands', 'Croplands',],
                      #        ['major_roads', 'minor_roads', 'motorway', 'primary', 'secondary', 'trunk', 'unclassified', 'residential'],
                      #         ['Crop_Nat_Vege_Mos', 'Permanent_Wetlands','minor_roads', 'trunk', 'unclassified', 'residential'],
                      #         ["ocfire"], ['pm2p5fire'], ['mami'], ['tcfire'],
                      #         ['Crop_Nat_Vege_Mos'], ['Permanent_Wetlands'], ['Croplands'],
                      #           ['major_roads'], ['minor_roads'], ['motorway'], ['primary'], ['secondary'], ['trunk'], ['unclassified'], ['residential'],
                    #]
                                #['GC_PM25'],['GC_SO4'],['GC_NH4'],['GC_NIT'],['GC_BC'],['GC_OM'],['GC_SOA'],['GC_DST'],['GC_SSLT'],
                               #                      ['PBLH'],['RH'],['PRECTOT'],['T2M'],['V10M'],['U10M'],['PS'],['USTAR'],
                               #                      ['NH3_anthro_emi'],['SO2_anthro_emi'],['NO_anthro_emi'],['OC_anthro_emi'],['BC_anthro_emi'],['NMVOC_anthro_emi'],
                               #                       ['DST_offline_emi'],['SSLT_offline_emi'],
                               #                         ['Urban_Builtup_Lands'],  ['Crop_Nat_Vege_Mos'],['Permanent_Wetlands'],['Croplands'],
                               #                         ["ocfire"], ["pm2p5fire"], ["mami"], ["tcfire"],
                               #                         ["primary"], ["residential"],['secondary'],["trunk"],["unclassified"],
                               #                       ['elevation'],['Population'],
                               #                      ['lat'],['lon'],['sin_days'],['cos_days']]                
                },
                'channel_to_add' : {
                    'values': [#["ocfire"], ["pm2p5fire"], ["mami"], ["tcfire"],
                               [],#['Barren'], ['Closed-Shrublands'], ['Crop_Nat_Vege_Mos'], ['Croplands'], ['Deciduous-Broadleaf-Forests'], ['Deciduous-Needleleaf-Forests'], ['Evergreen-Broadleaf-Forests'], ['Evergreen-Needleleaf-Forests'], ['Grasslands'], ['Mixed-Forests'], ['Open-Shrublands'], ['Permanent-Snow-Ice'], ['Permanent_Wetlands'], ['Savannas'],  ['Urban_Builtup_Lands'], ['Woody-Savannas'],
                                #['major_roads'], ['minor_roads'], ['motorway'], ['primary'], ['secondary'], ['trunk'], ['unclassified'], ['residential'],
                               ]},
            }
        }
        if MoE_Settings:
            sweep_configuration['parameters']['MoE_num_experts'] = {
                'values': [4]
            }
            sweep_configuration['parameters']['MoE_gating_hidden_size'] = {
                'values': [128]
            }
            sweep_configuration['parameters']['MoE_selected_channels'] = {
                'values': [ #[ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",],
                            [ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",   
                                "PBLH", "RH","V10M", "U10M", ],
                                #[ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",
                                #"Urban_Builtup_Lands", "Population", ],
                               #[ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",
                               # "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",],
                               # [ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",
                               # "PBLH", "RH","V10M", "U10M",
                               # "Urban_Builtup_Lands", "Population",
                               # "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",],
                            ]   
            }
        if MoCE_Settings:
            sweep_configuration['parameters']['MoCE_num_experts'] = {
                'values': [4]
            }
            sweep_configuration['parameters']['MoCE_gating_hidden_size'] = {
                'values': [64,128,256,512]
            }
            sweep_configuration['parameters']['MoCE_selected_channels_index_for_gate'] = {
                'values': [[ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",],
                           [ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",   
                               "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", ],
                                [ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",
                                "Urban_Builtup_Lands",  'Grasslands','Evergreen-Broadleaf-Forests', "Population", ],
                               #[ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",
                               # "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",],
                                [ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",
                                "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",],
                                ["tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",
                                 "ocfire", "pm2p5fire", "mami", "tcfire",
                                 "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", ],
                               # "Urban_Builtup_Lands", "Population",
                               # "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",],
                            ]   
            }
            sweep_configuration['parameters']['MoCE_base_model_channels'] = {
                'values': [[ "tSATAOD", "tSATPM25", #"eta",
                     "GC_PM25", "GC_SO4", "GC_NH4", "GC_NIT", "GC_OM", "GC_SOA", "GC_DST", "GC_SSLT",#"GC_BC",
               "PBLH", "RH", "PRECTOT", "T2M", "V10M", "U10M", "PS", 
                "NH3_anthro_emi", "SO2_anthro_emi", "NO_anthro_emi", "OC_anthro_emi",
                    "BC_anthro_emi",  "DST_offline_emi", "SSLT_offline_emi",#"NMVOC_anthro_emi",
                    "Urban_Builtup_Lands", 
                    "elevation", "Population", "lat", "lon", "sin_days", "cos_days", ]]
            }
            
            sweep_configuration['parameters']['MoCE_side_blocks_num'] = {
                'values': [[1,1,1,1],[1,0,1,0],[1,1,1,0],[1,0,0,1]]
            }
            sweep_configuration['parameters']['MoCE_side_output_channels'] = {
                'values': [[128,256,512,1024],[128,128,256,256],[256,512,512,1024], [64,128,256,512],]
            }
            sweep_configuration['parameters']['MoCE_side_pooling_kernel_switch'] = {
                'values': [True,False]
            }
            sweep_configuration['parameters']['MoCE_side_pooling_layer_type_3D'] = {
                'values': ['MaxPooling3d','AvgPooling3d'] #'AvgPooling3d', 'MaxPooling3d'
            }
            sweep_configuration['parameters']['MoCE_side_pooling_kernel_size'] = {
                'values': [(1,3,3),]
            }
            combo_indices = list(itertools.combinations(range(len(MOCE_CHANNEL_POOL)), MoCE_num_experts - 1)) ## have to adjust MoCE_num_experts at Net_Architecture.py accordingly.
            sweep_configuration['parameters']['channel_combo'] = {
                'values': [list(c) for c in combo_indices]  # e.g. [0, 4, 7]
            }

            
    if Apply_Transformer_architecture:
        sweep_configuration = {
            'name': 'HSV_Transformer_Sweep_Normalized_Speices',
            'entity': 'ACAG-NorthAmericaDailyPM25',
            'project': version,
            'method': "random",
            'metric': {
                'name': 'test_R2',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate0': {
                    'values': [0.0001, 0.001]
                },
                'batch_size': {
                    'values': [32,]
                },
                'epoch': {
                    'values': [111, 131,151]
                },
                'd_model': {
                    'values': [32, 64, 128]  # Example values for model dimension
                },
                'n_head': {
                    'values': [4,  8,]  # Example values for number of attention heads
                },
                'ffn_hidden': {
                    'values': [32, 64,128]  # Example values for feed-forward network hidden layer dimension
                },
                'num_layers': {
                    'values': [3, 4, 6, 8]  # Example values for number of encoder/decoder layers
                },
                'max_len': {
                    'values': [30, 60, 90, 365]  # Example values for maximum length of the input sequence
                },
                'spin_up_len': {
                    'values': [1, 3, 7]  # Example values for spin-up length
                },
                'drop_prob': {
                    'values': [0.01,  0.001]  # Example values for dropout probability
                },
                'channel_to_exclude': {
                    'values': [[]]  # No channels to exclude for Transformer
                },
                'channel_to_add' : {
                    'values': [[]]
                }

            }
        }
    
    if Apply_CNN_Transformer_architecture:
        sweep_configuration = {
            'name': 'HSV_CNN_Transformer_Sweep_Normalized_Speices',
            'entity': 'ACAG-NorthAmericaDailyPM25',
            'project': version,
            'method': "random",
            'metric': {
                'name': 'test_R2',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate0': {
                    'values': [0.0001, 0.001]
                },
                'batch_size': {
                    'values': [32,64,128,256]
                },
                'epoch': {
                    'values': [111,131,151]
                },
                'CNN_blocks_num': {
                    'values': [[2,2,2,2],[1,1,1,1],[3,3,3,3],[4,4,4,4]]
                },
                "CNN_output_channels": {
                    'values': [[64,128,256,512],[128,256,512,512],[128,256,512,1024]]  # Example values for output channels
                },
                'd_model': {
                    'values': [256, 512]  # Example values for model dimension
                },
                'n_head': {
                    'values': [4,  8]  # Example values for number of attention heads
                },
                'ffn_hidden': {
                    'values': [64, 128]  # Example values for feed-forward network hidden layer dimension
                },
                'num_layers': {
                    'values': [3, 4, 6]  # Example values for number of encoder/decoder layers
                },
                'max_len': {
                    'values': [30, 60]  # Example values for maximum length of the input sequence
                },
                'spin_up_len': {
                    'values': [1, 3]  # Example values for spin-up length
                },
                'drop_prob': {
                    'values': [0.01,  0.001]  # Example values for dropout probability
                },
                'CNN_channel_to_exclude': {
                    'values': [[]] 
                },
                'Transformer_channel_to_exclude': {
                    'values': [[]] 
                }
            }
        }
    return sweep_configuration

def wandb_parameters_return(wandb_config):
    ### This is used in TrainingModeule.py to get the hyperparameters from wandb_config.
    if Apply_CNN_architecture:
        print('wandb_parameters_return: ', wandb_config)
        batchsize_value = wandb_config['batch_size']
        learning_rate0_value = wandb_config['learning_rate0']
        epoch_value = wandb_config['epoch']
        return batchsize_value, learning_rate0_value, epoch_value
    if Apply_3D_CNN_architecture:
        print('wandb_parameters_return: ', wandb_config)
        batchsize_value = wandb_config['batch_size']
        learning_rate0_value = wandb_config['learning_rate0']
        epoch_value = wandb_config['epoch']
        return batchsize_value, learning_rate0_value, epoch_value
    if Apply_Transformer_architecture:
        print('wandb_parameters_return: ', wandb_config)
        batchsize_value = wandb_config['batch_size']
        learning_rate0_value = wandb_config['learning_rate0']
        epoch_value = wandb_config['epoch']
        d_model_value = wandb_config['d_model']
        n_head_value = wandb_config['n_head']
        ffn_hidden_value = wandb_config['ffn_hidden']
        num_layers_value = wandb_config['num_layers']
        max_len_value = wandb_config['max_len']
        spin_up_len_value = wandb_config['spin_up_len']
        drop_prob_value = wandb_config['drop_prob']
        return batchsize_value, learning_rate0_value, epoch_value, d_model_value, n_head_value, ffn_hidden_value, num_layers_value, max_len_value, spin_up_len_value, drop_prob_value
    if Apply_CNN_Transformer_architecture:
        print('wandb_parameters_return: ', wandb_config)
        batchsize_value = wandb_config['batch_size']
        learning_rate0_value = wandb_config['learning_rate0']
        epoch_value = wandb_config['epoch']
        CNN_blocks_num_value = wandb_config['CNN_blocks_num']
        CNN_output_channels_value = wandb_config['CNN_output_channels']
        d_model_value = wandb_config['d_model']
        n_head_value = wandb_config['n_head']
        ffn_hidden_value = wandb_config['ffn_hidden']
        num_layers_value = wandb_config['num_layers']
        max_len_value = wandb_config['max_len']
        spin_up_len_value = wandb_config['spin_up_len']
        drop_prob_value = wandb_config['drop_prob']
        return batchsize_value, learning_rate0_value, epoch_value, CNN_blocks_num_value, CNN_output_channels_value, d_model_value, n_head_value, ffn_hidden_value, num_layers_value, max_len_value, spin_up_len_value, drop_prob_value