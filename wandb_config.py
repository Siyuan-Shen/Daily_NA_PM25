import wandb
import os
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
            "ResCNN3D_blocks_num": ResCNN3D_blocks_num,
            "ResCNN3D_output_channels": ResCNN3D_output_channels,
            
        }
        if MoE_Settings:
            run_config["MoE_num_experts"] = MoE_num_experts
            run_config["MoE_gating_hidden_size"] = MoE_gating_hidden_size
            run_config["MoE_selected_channels"] = MoE_selected_channels
            
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

                'ResNet_blocks_num': {
                    'values': [[2,2,2,2],[1,1,1,1],[3,3,3,3],[4,4,4,4]]
                },
            }
        }
    if Apply_3D_CNN_architecture:
        sweep_configuration = {
            'name':'HSV_3DCNN_Sweep_Normalized_Speices_Variables_test',  # Name of the sweep
            'entity': 'ACAG-NorthAmericaDailyPM25',  # Your wandb entity (team name)
            'project': version,  # Your wandb project name
            'method': "random",  # 'grid', 'random', 'bayes'
            'metric': {
                'name': 'test_R2',
                'goal': 'maximize'  # 'minimize' or 'maximize'
            },
            'parameters': {
                'learning_rate0': {
                    'values': [0.01,0.001,0.0003,0.0001]
                },
                'batch_size': {
                    'values': [32,64,128,256,512]
                },
                'epoch':{
                    'values': [31,51,71,91,111,131]
                },
               'channel_to_exclude': {
                    'values': [[]] 
                },
                'ResCNN3D_blocks_num': {
                    'values': [[1,1,1,1],[1,0,0,1],[2,1,1,1],[2,2,2,2]]
                },

                'ResCNN3D_output_channels': {
                    'values': [[128,256,512,1024],]  # Example values for output channels
                },
                
                'channel_to_exclude': {
                    'values': [[]]#['GC_PM25'],['GC_SO4'],['GC_NH4'],['GC_NIT'],['GC_BC'],['GC_OM'],['GC_SOA'],['GC_DST'],['GC_SSLT'],
                               #                      ['PBLH'],['RH'],['PRECTOT'],['T2M'],['V10M'],['U10M'],['PS'],['USTAR'],
                               #                      ['NH3_anthro_emi'],['SO2_anthro_emi'],['NO_anthro_emi'],['OC_anthro_emi'],['BC_anthro_emi'],['NMVOC_anthro_emi'],
                               #                       ['DST_offline_emi'],['SSLT_offline_emi'],
                               #                         ['Urban_Builtup_Lands'],  ['Crop_Nat_Vege_Mos'],['Permanent_Wetlands'],['Croplands'],
                               #                         ["ocfire"], ["pm2p5fire"], ["mami"], ["tcfire"],
                               #                         ["primary"], ["residential"],['secondary'],["trunk"],["unclassified"],
                               #                       ['elevation'],['Population'],
                               #                      ['lat'],['lon'],['sin_days'],['cos_days']]                
                },
            }
        }
        if MoE_Settings:
            sweep_configuration["MoE_num_experts"] = [4,6,8]
            sweep_configuration["MoE_gating_hidden_size"] = [32,64,128]
            sweep_configuration["MoE_selected_channels"] = [[ "tSATAOD", "tSATPM25","lat", "lon", "sin_days", "cos_days",],
                                                            ]
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