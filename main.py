import toml
import pprint
import wandb
from wandb_config import  wandb_initialize, wandb_sweep_config
from Evaluation_pkg.Hyperparameter_Search_Validation import Hyperparameters_Search_Training_Testing_Validation
from Evaluation_pkg.Spatial_CrossValidation import spatial_cross_validation
from Evaluation_pkg.utils import *
from Training_pkg.utils import *
from Model_Structure_pkg.utils import *

wandb.login(key='f256dceb0a92527f2588e098c831713ce3428bda')
cfg = toml.load('./config.toml')
pprint.pprint(cfg)

def Hyperparameters_Search_Training_Testing_Validation_main(total_channel_names, main_stream_channel_names, side_channel_names):
    
    wandb_initialize()
    wandb_config = wandb.config
    print('wandb_config: ', wandb_config)
    if wandb_config.channel_to_exclude is not None:
        print('wandb_config.channel_to_exclude: ', wandb_config.channel_to_exclude)
        total_channel_names,main_stream_channel_names, side_channel_names = Get_channel_names(channels_to_exclude=wandb_config.channel_to_exclude)
    Hyperparameters_Search_Training_Testing_Validation(wandb_config=wandb_config,total_channel_names=total_channel_names,main_stream_channel_names=main_stream_channel_names,
                                                              side_stream_channel_names=side_channel_names,
                                                              ) 


def Spatial_Cross_Validation_main(total_channel_names, main_stream_channel_names, side_channel_names):
    wandb_initialize()
    wandb_config = wandb.config
    print('wandb_config: ', wandb_config)
    if wandb_config.channel_to_exclude is not None:
        print('wandb_config.channel_to_exclude: ', wandb_config.channel_to_exclude)
        total_channel_names,main_stream_channel_names, side_channel_names = Get_channel_names(channels_to_exclude=wandb_config.channel_to_exclude)
    spatial_cross_validation(wandb_config=wandb_config,total_channel_names=total_channel_names, main_stream_channel_names=main_stream_channel_names,
                             side_stream_channel_names=side_channel_names)
    

    
if __name__ == "__main__":
    total_channel_names, main_stream_channel_names, side_channel_names = Get_channel_names(channels_to_exclude=[])
    if Hyperparameters_Search_Validation_Switch:
        if HSV_Apply_wandb_sweep_Switch:
            sweep_config = wandb_sweep_config()
            sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['project'],entity=sweep_config['entity'])
            wandb.agent(sweep_id, function=lambda: Hyperparameters_Search_Training_Testing_Validation_main(total_channel_names=total_channel_names,
                                                                                                   main_stream_channel_names=main_stream_channel_names,
                                                                                                   side_channel_names=side_channel_names), count=wandb_sweep_count)
        else:
            wandb_initialize()
            wandb_config = None
            Hyperparameters_Search_Training_Testing_Validation(wandb_config=wandb_config,total_channel_names=total_channel_names,main_stream_channel_names=main_stream_channel_names,
                                                           side_stream_channel_names=side_channel_names,
                                                           )
    
    if Spatial_CrossValidation_Switch:
        if Spatial_CV_Apply_wandb_sweep_Switch:
            sweep_config = wandb_sweep_config()
            sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['project'],entity=sweep_config['entity'])
            wandb.agent(sweep_id, function=lambda: Spatial_Cross_Validation_main(total_channel_names=total_channel_names,
                                                                                 main_stream_channel_names=main_stream_channel_names,
                                                                                 side_channel_names=side_channel_names), count=wandb_sweep_count)
        else:
            wandb_initialize()
            wandb_config = None
            spatial_cross_validation(wandb_config=wandb_config,total_channel_names=total_channel_names, main_stream_channel_names=main_stream_channel_names,
                                     side_stream_channel_names=side_channel_names)
            
