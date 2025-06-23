import wandb
from Training_pkg.utils import description,Apply_CNN_architecture,Apply_3D_CNN_architecture, version, learning_rate0, epoch, batchsize


def wandb_initialize():
    if Apply_CNN_architecture:
        # Initialize a new wandb run
        wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="ACAG-NorthAmericaDailyPM25",
            # Set the wandb project where this run will be logged.
            project=version,
            # Track hyperparameters and run metadata.
            name=description,
            config={
                "learning_rate": learning_rate0,  # Replace with your learning rate variable
                "architecture": "CNN",
                "epochs": epoch,
                "batch_size": batchsize,  # Replace with your batch size variable
            },## This config will be ignored if wandb_sweep_config is used
        )
        # Return the wandb run object
    if Apply_3D_CNN_architecture:
        # Initialize a new wandb run
        wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="ACAG-NorthAmericaDailyPM25",
            # Set the wandb project where this run will be logged.
            project=version,
            # Track hyperparameters and run metadata.
            name=description,
            config={
                "learning_rate": learning_rate0,  # Replace with your learning rate variable
                "architecture": "3DCNN",
                "epochs": epoch,
                "batch_size": batchsize,  # Replace with your batch size variable
            },## This config will be ignored if wandb_sweep_config is used
        )
    return

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
                    'values': [0.001, 0.0001]
                },
                'batch_size': {
                    'values': [64,128,256]
                },
                'epoch':{
                    'values': [51, 71,91,111,131]
                },
                'channel_to_exclude': {
                    'values': [[]]
                }
            }
        }
    if Apply_3D_CNN_architecture:
        sweep_configuration = {
            'name':'HSV_3DCNN_Sweep_Channels_Exclusion',  # Name of the sweep
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
                    'values': [32,64,128,256]
                },
                'epoch':{
                    'values': [71,91,111,131]
                },
               'channel_to_exclude': {
                    'values': [['']
                               ]
                }
            }
        }
    return sweep_configuration

def wandb_sweep_parameters_return(sweep_config):
    if Apply_CNN_architecture:
        print('wandb_sweep_parameters_return: ', sweep_config)
        batchsize_value = sweep_config.batch_size
        learning_rate0_value = sweep_config.learning_rate0
        epoch_value = sweep_config.epoch
        return batchsize_value, learning_rate0_value, epoch_value
    if Apply_3D_CNN_architecture:
        print('wandb_sweep_parameters_return: ', sweep_config)
        batchsize_value = sweep_config.batch_size
        learning_rate0_value = sweep_config.learning_rate0
        epoch_value = sweep_config.epoch
        return batchsize_value, learning_rate0_value, epoch_value