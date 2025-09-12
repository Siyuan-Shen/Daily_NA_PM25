import os 
from Estimation_pkg.utils import inputfiles_table
from Estimation_pkg.data_func import get_extent_index

from Training_pkg.utils import *
import numpy as np
import time

## The save and load of trained models are defined in Training_pkg/iostream.py


def get_Estimation_recording_filename(outdir, file_target,typeName,Area,YYYY,MM,DD,nchannel, **args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)

    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'
        outdir = outdir + f'{file_target}/{YYYY}/{MM}/'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        recording_filename = outdir + f'{file_target}_{Model_structure_type}_{typeName}_{Area}_{YYYY}{MM}{DD}_{width}x{height}_{nchannel}Channel{description}.npy'
    elif Apply_3D_CNN_architecture:
        Model_structure_type = '3DCNNModel'
        outdir = outdir + f'{file_target}/{YYYY}/{MM}/'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        recording_filename = outdir + f'{file_target}_{Model_structure_type}_{typeName}_{Area}_{YYYY}{MM}{DD}_{depth}x{width}x{height}_{nchannel}Channel{description}.npy'
    return recording_filename
    
def save_Estimation_Map(mapdata,outdir, file_target,typeName,Area,YYYY,MM,DD,nchannel, **args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)
    
    if Apply_CNN_architecture:
        outfile = get_Estimation_recording_filename(outdir, file_target,typeName,Area,YYYY,MM,DD,nchannel,width=width,height=height)
    elif Apply_3D_CNN_architecture:
        outfile = get_Estimation_recording_filename(outdir, file_target,typeName,Area,YYYY,MM,DD,nchannel,width=width,height=height,depth=depth)

    np.save(outfile, mapdata)
    print(f'Saved the {file_target} map data to {outfile}')
    return

def load_Estimation_Map(outdir, file_target,typeName,Area,YYYY,MM,DD,nchannel, **args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)

    if Apply_CNN_architecture:
        infile = get_Estimation_recording_filename(outdir, file_target,typeName,Area,YYYY,MM,DD,nchannel,width,height)
    elif Apply_3D_CNN_architecture:
        infile = get_Estimation_recording_filename(outdir, file_target,typeName,Area,YYYY,MM,DD,nchannel,width,height,depth)

    if not os.path.isfile(infile):
        raise ValueError(f'The {infile} does not exist!')
    
    mapdata = np.load(infile)
    print(f'Loaded the {file_target} map data from {infile}')
    return mapdata


def load_map_data(channel_names, YYYY, MM,DD):
    inputfiles = inputfiles_table(YYYY=YYYY,MM=MM,DD=DD)
    indir = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/'
    lat_infile = indir + 'tSATLAT_NA.npy'
    lon_infile = indir + 'tSATLON_NA.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)
    output = np.zeros((len(channel_names), len(SATLAT), len(SATLON)))
    loading_time_start = time.time()
    for i in range(len(channel_names)):
        infile = inputfiles[channel_names[i]]
        tempdata = np.load(infile)
        print('{} has been loaded!'.format(channel_names[i]))
        output[i,:,:] = tempdata
    loading_time_end = time.time()
    print('Loading time cost: ', loading_time_end - loading_time_start, 's')
    return output
