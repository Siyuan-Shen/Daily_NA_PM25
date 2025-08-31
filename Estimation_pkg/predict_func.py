import numpy as np
import gc
import os
import torch
import torch.nn as nn
import time
from Estimation_pkg.utils import *
from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import Split_Datasets_based_site_index,randomly_select_training_testing_indices,Get_final_output
from Evaluation_pkg.iostream import *
from Evaluation_pkg.Statistics_Calculation_func import calculate_statistics
from Model_Structure_pkg.CNN_Module import initial_cnn_network
from Model_Structure_pkg.ResCNN3D_Module import initial_3dcnn_net
from Model_Structure_pkg.utils import *

from Training_pkg.utils import *
from Training_pkg.utils import epoch as config_epoch, batchsize as config_batchsize, learning_rate0 as config_learning_rate0
from Training_pkg.TensorData_func import Dataset_Val, Dataset
from Training_pkg.TrainingModule import CNN_train, cnn_predict, CNN3D_train, cnn_predict_3D
from Training_pkg.data_func import CNNInputDatasets, CNN3DInputDatasets
from Training_pkg.iostream import load_daily_datesbased_model
from multiprocessing import Manager
import torch.multiprocessing as mp


def map_predict():
    
    return

def cnn_mapdata_predict_func():
    return