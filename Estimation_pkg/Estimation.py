from Estimation_pkg.training_func import Train_Model_forEstimation
from Estimation_pkg.utils import *

from Training_pkg.iostream import load_daily_datesbased_model


def Estimation_Func(total_channel_names, main_stream_channel_names,
                             side_stream_channel_names,sweep_id=None,):


    ### Training
    if Estimation_Train_model_Switch:
        Train_Model_forEstimation(total_channel_names, main_stream_channel_names,
                                   side_stream_channel_names, sweep_id=None,)

    ### Estimation
    if Map_estimation_Switch:
        
        None

    return