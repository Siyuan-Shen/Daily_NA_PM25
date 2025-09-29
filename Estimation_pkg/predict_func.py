import numpy as np
import gc
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import time
from Estimation_pkg.utils import *
from Estimation_pkg.data_func import get_extent_index, get_landtype
from Estimation_pkg.iostream import *
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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



def ddp_setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cnn_mapdata_predict_func(rank, world_size,model,predict_begindate,predict_endate,total_channel_names,
                                evaluation_type, typeName,Area,extent,
                                train_mean, train_std, true_mean, true_std, width,height):
    nchannel = len(total_channel_names)
    print('world_size: {}'.format(world_size))
    try:
        print(f"[Rank {rank}] Starting 3D CNN predict")
        # Your original CNN3D_train logic goes here...
    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        print(f"[Rank {rank}] Exception occurred:\n{traceback.format_exc()}")
        raise e
    print(f"Rank {rank} process started.")

    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        device = rank
        model.to(device)
        model = DDP(model, device_ids=[device])
    
    AllDates, int_AllDates = create_date_range(predict_begindate, predict_endate)
    lat_index, lon_index = get_extent_index(extent)
    landtype = get_landtype('2020',extent)
    lat_infile = LATLON_indir + 'NA_SATLAT_0p01.npy'
    lon_infile = LATLON_indir + 'NA_SATLON_0p01.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)

    with torch.no_grad():
        model.eval()

        for date in int_AllDates:
            # Your original CNN3D_train logic goes here...
            YYYY,MM,DD = getGrg_YYYY_MM_DD(date)
            print(f"Processing date: {YYYY}-{MM}-{DD}")
            output = np.full((len(lat_index),len(lon_index)),-9999.0,dtype=np.float32)
            
            ## Load Map Data for the date    
            YYYY, MM, DD = getGrg_YYYY_MM_DD(date)
            temp_map_data = load_map_data(total_channel_names, YYYY, MM, DD)
            
            ## Convert to 3D CNN reading and predict
            for ix in range(len(lat_index)//world_size):
                ix = ix*world_size + rank
                land_index = np.where(landtype[ix,:] != 0)

                print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' )
                if len(land_index[0]) == 0:
                    None
                else:
                    temp_input = np.zeros((len(land_index[0]), nchannel, width, width), dtype=np.float32)
                    for iy in range(len(land_index[0])):
                        temp_input[iy,:,:,:] = temp_map_data[:,int(lat_index[ix] - (width - 1) / 2):int(lat_index[ix] + (width + 1) / 2), int(lon_index[land_index[0][iy]] - (width - 1) / 2):int(lon_index[land_index[0][iy]] + (width + 1) / 2)]

                    temp_input -= train_mean
                    temp_input /= train_std
                
                final_output = []
                final_output = np.array(final_output)
                if world_size <= 1:
                    predict_loader = DataLoader(Dataset_Val(temp_input), 2000, shuffle=False)
                elif world_size > 1:
                    predict_dataset = Dataset_Val(temp_input)
                    predict_loader = DataLoader(predict_dataset, 2000, shuffle=False)
                
                for i, image in enumerate(predict_loader):
                    # Your original prediction logic goes here...
                    image = image.to(device)
                    temp_output = model(image).cpu().detach().numpy()
                    final_output = np.append(final_output, temp_output)

                output[ix,land_index[0]] = final_output
            
            if world_size > 1:
                out_t = torch.from_numpy(output).to(device)
                dist.all_reduce(out_t, op=dist.ReduceOp.MAX)
                output = out_t.cpu().numpy()
            ## Save the output
            if rank == 0:
                output = map_data_final_output(output,
                         bias,normalize_bias,normalize_species,absolute_species,log_species,
                         true_mean, true_std, YYYY, MM, DD)
                save_Estimation_Map(mapdata = output, outdir = data_recording_outdir, 
            file_target='Map_Estimation', typeName=typeName, Area=Area, YYYY=YYYY, MM=MM, DD=DD, nchannel=nchannel,
            width=width, height=height 
            )
    if world_size > 1:
        dist.barrier()
        destroy_process_group()

    return

def cnn3D_mapdata_predict_func(rank, world_size,model,predict_begindate,predict_endate,total_channel_names,
                                evaluation_type, typeName,Area,extent,
                                train_mean, train_std, true_mean, true_std, width,height,depth):
    nchannel = len(total_channel_names)
    print('world_size: {}'.format(world_size))
    try:
        print(f"[Rank {rank}] Starting 3D CNN predict")
        # Your original CNN3D_train logic goes here...
    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        print(f"[Rank {rank}] Exception occurred:\n{traceback.format_exc()}")
        raise e
    print(f"Rank {rank} process started.")

    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        device = rank
        model.to(device)
        model = DDP(model, device_ids=[device])
    
    AllDates, int_AllDates = create_date_range(predict_begindate, predict_endate)
    lat_index, lon_index = get_extent_index(extent)
    landtype = get_landtype('2020',extent)
    lat_infile = LATLON_indir + 'NA_SATLAT_0p01.npy'
    lon_infile = LATLON_indir + 'NA_SATLON_0p01.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)

    with torch.no_grad():
        model.eval()

        for date in int_AllDates:
            # Your original CNN3D_train logic goes here...
            YYYY,MM,DD = getGrg_YYYY_MM_DD(date)
            print(f"Processing date: {YYYY}-{MM}-{DD}")
            output = np.full((len(lat_index),len(lon_index)),-9999.0,dtype=np.float32)
            temp_map_data = np.zeros((len(total_channel_names),  depth,  len(SATLAT), len(SATLON)))

            ## Load Map Data for the date
            for iday in range(depth):
                temp_date = get_previous_date_YYYY_MM_DD(date,iday)
                YYYY, MM, DD = getGrg_YYYY_MM_DD(temp_date)
                temp_map_data[:, (depth - iday -1),:,:] = load_map_data(total_channel_names, YYYY, MM, DD)
            
            ## Convert to 3D CNN reading and predict
            for ix in range(len(lat_index)//world_size):
                ix = ix*world_size + rank
                land_index = np.where(landtype[ix,:] != 0)

                print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' )
                if len(land_index[0]) == 0:
                    None
                else:
                    temp_input = np.zeros((len(land_index[0]), nchannel, depth, width, width), dtype=np.float32)
                    for iy in range(len(land_index[0])):
                        temp_input[iy,:,:,:,:] = temp_map_data[:,:,int(lat_index[ix] - (width - 1) / 2):int(lat_index[ix] + (width + 1) / 2), int(lon_index[land_index[0][iy]] - (width - 1) / 2):int(lon_index[land_index[0][iy]] + (width + 1) / 2)]
                
                    temp_input -= train_mean
                    temp_input /= train_std
                
                final_output = []
                final_output = np.array(final_output)
                if world_size <= 1:
                    predict_loader = DataLoader(Dataset_Val(temp_input), 2000, shuffle=False)
                elif world_size > 1:
                    predict_dataset = Dataset_Val(temp_input)
                    predict_loader = DataLoader(predict_dataset, 2000, shuffle=False)
                
                for i, image in enumerate(predict_loader):
                    # Your original prediction logic goes here...
                    image = image.to(device)
                    temp_output = model(image).cpu().detach().numpy()
                    final_output = np.append(final_output, temp_output)

                output[ix,land_index[0]] = final_output
            
            if world_size > 1:
                out_t = torch.from_numpy(output).to(device)
                dist.all_reduce(out_t, op=dist.ReduceOp.MAX)
                output = out_t.cpu().numpy()
            ## Save the output
            if rank == 0:
                output = map_data_final_output(output,
                         bias,normalize_bias,normalize_species,absolute_species,log_species,
                         true_mean, true_std, YYYY, MM, DD)
                save_Estimation_Map(mapdata = output, outdir = data_recording_outdir, 
            file_target='Map_Estimation', typeName=typeName, Area=Area, YYYY=YYYY, MM=MM, DD=DD, nchannel=nchannel,
            width=width, height=height, depth=depth 
            )
    if world_size > 1:
        dist.barrier()
        destroy_process_group()
    return

def map_data_final_output(Validation_Prediction,
                         bias,normalize_bias,normalize_species,absolute_species,log_species,
                         true_mean, true_std, YYYY, MM, DD):
    lat_index, lon_index = get_extent_index(Extent)
    if bias == True:
        GeoSpecies = load_map_data('tSATPM25', YYYY, MM, DD)
        validation_geophysical_species = GeoSpecies[lat_index[0]:lat_index[-1]+1,lon_index[0]:lon_index[-1]+1]
        final_data = Validation_Prediction + validation_geophysical_species
    elif normalize_bias == True:
        GeoSpecies = load_map_data('tSATPM25', YYYY, MM, DD)
        validation_geophysical_species = GeoSpecies[lat_index[0]:lat_index[-1]+1,lon_index[0]:lon_index[-1]+1]
        final_data = Validation_Prediction * true_std + true_mean + validation_geophysical_species
    elif normalize_species == True:
        final_data = Validation_Prediction * true_std + true_mean
    elif absolute_species == True:
        final_data = Validation_Prediction
    elif log_species == True:
        final_data = np.exp(Validation_Prediction) - 1
    return final_data