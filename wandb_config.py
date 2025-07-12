import wandb
import os
from Training_pkg.utils import description,Apply_CNN_architecture,Apply_3D_CNN_architecture, version, learning_rate0, epoch, batchsize

def wandb_run_config():
    if Apply_CNN_architecture:
        run_config = {
            "learning_rate0": learning_rate0,  # Replace with your learning rate variable
            "architecture": "CNN",
            "epoch": epoch,
            "batch_size": batchsize,  # Replace with your batch size variable
        }
    if Apply_3D_CNN_architecture:
        run_config = {
            "learning_rate0": learning_rate0,  # Replace with your learning rate variable
            "architecture": "3DCNN",
            "epoch": epoch,
            "batch_size": batchsize,  # Replace with your batch size variable
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
                    'values': [[]]#['GC_PM25'],['GC_SO4'],['GC_NH4'],['GC_NIT'],['GC_BC'],['GC_OM'],['GC_SOA'],['GC_DST'],['GC_SSLT'],
                                #                     ['PBLH'],['RH'],['PRECTOT'],['T2M'],['V10M'],['U10M'],['PS'],#'USTAR',
                                #                      ['NH3_anthro_emi'],['SO2_anthro_emi'],['NO_anthro_emi'],['OC_anthro_emi'],['BC_anthro_emi'],['NMVOC_anthro_emi'],
                                #                      ['DST_offline_emi'],['SSLT_offline_emi'],
                                #                    ['Urban_Builtup_Lands'], #  'Crop_Nat_Vege_Mos','Permanent_Wetlands','Croplands',
                                                    #  'major_roads','minor_roads','motorway',
                                #                      ['elevation'],['Population'],
                                 #                     ['lat'],['lon'],['sin_days'],['cos_days']]
                }
            }
        }
    if Apply_3D_CNN_architecture:
        sweep_configuration = {
            'name':'HSV_3DCNN_Sweep_Normalized_Speices',  # Name of the sweep
            'entity': 'ACAG-NorthAmericaDailyPM25',  # Your wandb entity (team name)
            'project': version,  # Your wandb project name
            'method': "random",  # 'grid', 'random', 'bayes'
            'metric': {
                'name': 'test_R2',
                'goal': 'maximize'  # 'minimize' or 'maximize'
            },
            'parameters': {
                'learning_rate0': {
                    'values': [ 0.01,0.001, 0.0001]
                },
                'batch_size': {
                    'values': [32,64,128,256,512]
                },
                'epoch':{
                    'values': [31,51,71,91,111,131]
                },
               'channel_to_exclude': {
                    'values': [['']
                               ]
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