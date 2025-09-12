import numpy as np
import mat73 as mat
from Training_pkg.utils import *



def get_landtype(YYYY,extent)->np.array:
    #landtype_infile = '/my-projects/Projects/MLCNN_PM25_2021/data/inputdata/Other_Variables_MAP_INPUT/{}/MCD12C1_LandCoverMap_{}.npy'.format(YYYY,YYYY)
    #landtype = np.load(landtype_infile)
    Mask_indir = '/my-projects/mask/NA_Masks/Cropped_NA_Masks/'
    '''
    Contiguous_US_data = nc.Dataset(Mask_indir+'Cropped_REGIONMASK-Contiguous United States.nc')
    Canada_data        = nc.Dataset(Mask_indir+'Cropped_REGIONMASK-Canada.nc')
    Alaska_data        = nc.Dataset(Mask_indir+'Cropped_REGIONMASK-Alaska.nc')
    Contiguous_US_mask = np.array(Contiguous_US_data['regionmask'][:])
    Canada_mask        = np.array(Canada_data['regionmask'][:])
    Alaska_mask        = np.array(Alaska_data['regionmask'][:])
    landtype = Contiguous_US_mask + Canada_mask + Alaska_mask
    lat_index,lon_index = get_extent_index(extent=extent)
    '''
    landtype_infile = '/my-projects/mask/Land_Ocean_Mask/NewLandMask-0.01.mat'
    LandMask = mat.loadmat(landtype_infile)
    MASKp1 = LandMask['MASKp1']
    MASKp2 = LandMask['MASKp2']
    MASKp3 = LandMask['MASKp3']
    MASKp4 = LandMask['MASKp4']
    MASKp5 = LandMask['MASKp5']
    MASKp6 = LandMask['MASKp6']
    MASKp7 = LandMask['MASKp7']
    MASKp_land = MASKp1 +MASKp2 + MASKp3 + MASKp4 + MASKp5 + MASKp6 + MASKp7 
    landtype = np.zeros((13000,36000),dtype=np.float32)
    landtype = MASKp_land
    lat_index,lon_index = get_GL_extent_index(extent=extent)
    
    output = np.zeros((len(lat_index),len(lon_index)), dtype=int)

    for ix in range(len(lat_index)):
        output[ix,:] = landtype[lat_index[ix],lon_index]
    return output


def get_GL_extent_index(extent)->np.array:
    '''
    :param extent:
        The range of the input. [Bottom_Lat, Up_Lat, Left_Lon, Right_Lon]
    :return:
        lat_index, lon_index
    '''
    SATLAT = np.load(LATLON_indir + 'GL_SATLAT_0p01.npy')
    SATLON = np.load(LATLON_indir + 'GL_SATLON_0p01.npy')
    lat_index = np.where((SATLAT >= extent[0])&(SATLAT<=extent[1]))
    lon_index = np.where((SATLON >= extent[2])&(SATLON<=extent[3]))
    lat_index = np.squeeze(np.array(lat_index))
    lon_index = np.squeeze(np.array(lon_index))
    return lat_index,lon_index

def get_extent_index(extent)->np.array:
    '''
    :param extent:
        The range of the input. [Bottom_Lat, Up_Lat, Left_Lon, Right_Lon]
    :return:
        lat_index, lon_index
    '''
    
    lat_infile = LATLON_indir + 'NA_SATLAT_0p01.npy'
    lon_infile = LATLON_indir + 'NA_SATLON_0p01.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)
    lat_index = np.where((SATLAT >= extent[0])&(SATLAT<=extent[1]))
    lon_index = np.where((SATLON >= extent[2])&(SATLON<=extent[3]))
    lat_index = np.squeeze(np.array(lat_index))
    lon_index = np.squeeze(np.array(lon_index))
    return lat_index,lon_index

