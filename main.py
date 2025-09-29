import toml
import pprint
import wandb
import time
from wandb_config import  wandb_initialize, wandb_sweep_config,wandb_run_config
from Evaluation_pkg.Hyperparameter_Search_Validation import Hyperparameters_Search_Training_Testing_Validation
from Evaluation_pkg.Spatial_CrossValidation import spatial_cross_validation
from Evaluation_pkg.Random_CrossValidation import random_cross_validation    
from Evaluation_pkg.SHAPvalue_analysis import Spatial_CV_SHAP_Analysis
from Evaluation_pkg.utils import *

from Estimation_pkg.Estimation import Estimation_Func
from Estimation_pkg.utils import *
from Training_pkg.utils import *
from Model_Structure_pkg.utils import *
from config import cfg


    

    
if __name__ == "__main__":
    wandb.login(key='f256dceb0a92527f2588e098c831713ce3428bda')

    pprint.pprint(cfg)
    import os

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Pick an unused port

    start_time = time.time()


    def Hyperparameters_Search_Training_Testing_Validation_main(total_channel_names, main_stream_channel_names, side_channel_names,sweep_id=None, entity=None, project=None, width=None, height=None, depth=None):
        
        Hyperparameters_Search_Training_Testing_Validation(total_channel_names=total_channel_names,main_stream_channel_names=main_stream_channel_names,
                                                                side_stream_channel_names=side_channel_names,sweep_id=sweep_id
                                                                ) 


    def Spatial_Cross_Validation_main(total_channel_names, main_stream_channel_names, side_channel_names,sweep_id=None):

        
        spatial_cross_validation(total_channel_names=total_channel_names, main_stream_channel_names=main_stream_channel_names,
                                side_stream_channel_names=side_channel_names,sweep_id=sweep_id)
    



    
    #### Start Running the main functions ####
    total_channel_names, main_stream_channel_names, side_channel_names = Get_channel_names(channels_to_exclude=[])
    if Hyperparameters_Search_Validation_Switch:
        if HSV_Apply_wandb_sweep_Switch:
            sweep_config = wandb_sweep_config()
            sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['project'],entity=sweep_config['entity'])
            wandb.agent(sweep_id, function=lambda: Hyperparameters_Search_Training_Testing_Validation_main(total_channel_names=total_channel_names,
                                                                                                main_stream_channel_names=main_stream_channel_names,
                                                                                                side_channel_names=side_channel_names,sweep_id=sweep_id), count=wandb_sweep_count)
        else:
            Hyperparameters_Search_Training_Testing_Validation(total_channel_names=total_channel_names,main_stream_channel_names=main_stream_channel_names,
                                                        side_stream_channel_names=side_channel_names,
                                                        )
    
    if Spatial_CrossValidation_Switch:
        if Spatial_CV_Apply_wandb_sweep_Switch:
            sweep_config = wandb_sweep_config()
            sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['project'],entity=sweep_config['entity'])
            wandb.agent(sweep_id, function=lambda: Spatial_Cross_Validation_main(total_channel_names=total_channel_names,
                                                                                 main_stream_channel_names=main_stream_channel_names,
                                                                                 side_channel_names=side_channel_names,sweep_id=sweep_id), count=wandb_sweep_count)
        else:
            spatial_cross_validation(total_channel_names=total_channel_names, main_stream_channel_names=main_stream_channel_names,
                                     side_stream_channel_names=side_channel_names)
    
    if Random_CrossValidation_Switch:
        if Random_CV_Apply_wandb_sweep_Switch:
            sweep_config = wandb_sweep_config()
            sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['project'],entity=sweep_config['entity'])
            wandb.agent(sweep_id, function=lambda: random_cross_validation(total_channel_names=total_channel_names,
                                                                                 main_stream_channel_names=main_stream_channel_names,
                                                                                 side_channel_names=side_channel_names,sweep_id=sweep_id), count=wandb_sweep_count)
        else:
            random_cross_validation(total_channel_names=total_channel_names, main_stream_channel_names=main_stream_channel_names,
                                     side_stream_channel_names=side_channel_names)
    if Spatial_CV_SHAP_Analysis_Switch:
        Spatial_CV_SHAP_Analysis(total_channel_names=total_channel_names, main_stream_channel_names=main_stream_channel_names,
                                 side_stream_channel_names=side_channel_names)

    if Estimation_Switch:
        Estimation_Func(total_channel_names, main_stream_channel_names, side_channel_names,)
    end_time = time.time()
    world_size = torch.cuda.device_count()
    print('Total time taken: {:.2f} secons, with {} GPUs'.format(end_time - start_time, world_size))