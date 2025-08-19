from config import cfg
import datetime


Estimation_Settings = cfg['Estimation-Settings']

Estimation_Switch = Estimation_Settings['Estimation_Switch']
Estimation_Train_model_Switch = Estimation_Settings['Train_model_Switch']
Map_estimation_Switch = Estimation_Settings['Map_estimation_Switch']
Estimation_visualization_Switch = Estimation_Settings['Estimation_visualization_Switch']

Training_Settings = Estimation_Settings['Training_Settings']
Training_begin_dates = Training_Settings['Training_begin_dates']
Training_end_dates = Training_Settings['Training_end_dates']

Map_Estimation_Settings = Estimation_Settings['Map_Estimation_Settings']
Eatimation_Daily_Switch = Map_Estimation_Settings['Eatimation_Daily_Switch']
Estimation_trained_begin_dates = Map_Estimation_Settings['Estimation_trained_begin_dates']
Estimation_trained_end_dates = Map_Estimation_Settings['Estimation_trained_end_dates']
Estimation_begindates = Map_Estimation_Settings['Estimation_begindates']
Estimation_enddates = Map_Estimation_Settings['Estimation_enddates']
Extent = Map_Estimation_Settings['Extent']
Estimation_Area = Map_Estimation_Settings['Estimation_Area']


Save_Monthly_Average_Switch = Map_Estimation_Settings['Save_Monthly_Average_Switch']
Save_Monthly_Average_begindates = Map_Estimation_Settings['Save_Monthly_Average_begindates']
Save_Monthly_Average_enddates = Map_Estimation_Settings['Save_Monthly_Average_enddates']

Save_Annual_Average_Switch = Map_Estimation_Settings['Save_Annual_Average_Switch']
Save_Annual_Average_beginyear = Map_Estimation_Settings['Save_Annual_Average_beginyear']
Save_Annual_Average_endyear = Map_Estimation_Settings['Save_Annual_Average_endyear']

Visualization_Settings = Estimation_Settings['Visualization_Settings']
Map_Plot_Switch = Visualization_Settings['Map_Plot_Switch']
Daily_Plot_Switch = Visualization_Settings['Daily_Plot_Switch']
Daily_Plot_begindates = Visualization_Settings['Daily_Plot_begindates']
Daily_Plot_enddates = Visualization_Settings['Daily_Plot_enddates']
Monthly_Plot_Switch = Visualization_Settings['Monthly_Plot_Switch']
Monthly_Plot_begindates = Visualization_Settings['Monthly_Plot_begindates']
Monthly_Plot_enddates = Visualization_Settings['Monthly_Plot_enddates'] 
Annual_Plot_Switch = Visualization_Settings['Annual_Plot_Switch']
Annual_Plot_beginyears = Visualization_Settings['Annual_Plot_beginyears']
Annual_Plot_endyears = Visualization_Settings['Annual_Plot_endyears']

GeoPM25_AOD_ETA_version = 'vAOD20240322vGEO20241212'

MapData_Indir = '/my-projects2/Projects/Daily_PM25_DL_2024/data/Input_Variables_MapData/'
MapData_fromRegionalComponentProject_Indir = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/'
AVD_version = 'AVD_d20240814'
Geophysical_version = 'vAOD20240322vGEO20241212'

############################################################################################
# The indir of input mapdata.
Geophysical_indir        = '{MapData_Indir}Geophysical_Variables/'
GEOS_Chem_indir          = '{MapData_Indir}GEOS-Chem_input/'
Meteorology_indir        = '{MapData_Indir}Meteorology_input/'
Anthropogenic_Emissions_input_indir = '{MapData_fromRegionalComponentProject_Indir}Anthropogenic_Emissions_input/'
Offline_Emissions_input_indir       = '{MapData_fromRegionalComponentProject_Indir}Offline_Emissions_input/'

LandCover_input_indir               = '{MapData_fromRegionalComponentProject_Indir}LandCover_input/'
Population_input_indir              = '{MapData_fromRegionalComponentProject_Indir}Population_input/'
OpenStreetMap_log_road_indir        = '{MapData_fromRegionalComponentProject_Indir}OpenStreetMap_log_road_map_data/'
OpenStreetMap_road_density_indir    = '{MapData_fromRegionalComponentProject_Indir}OpenStreetMap_RoadDensity_input/'
OpenStreetMap_nearest_dist_indir    = '{MapData_fromRegionalComponentProject_Indir}OpenStreetMap_RoadDensity_NearestDistances_forEachPixels_input/'
Geographical_Variables_input_indir  = '{MapData_fromRegionalComponentProject_Indir}Geographical_Variables_input/'
Spatiotemporal_input_indir          = '{MapData_Indir}Spatiotemporal_input/'

def inputfiles_table(YYYY, MM, DD):

    inputfiles_dict = {
        ############################################################################################################
        # Geophysical Variables
        'gGCFRAC5km'            : Geophysical_indir + '{}/{}/{}/cropped_gGCFRAC5km_{}{}{}.npy'.format(Geophysical_version,YYYY,MM,YYYY,MM,DD),
        'gSATPM25orig5km'       : Geophysical_indir + '{}/{}/{}/cropped_gSATPM25orig5km_{}{}{}.npy'.format(Geophysical_version,YYYY,MM,YYYY,MM,DD),
        'gSATPM25orig5kmdelta'  : Geophysical_indir + '{}/{}/{}/cropped_gSATPM25orig5kmdelta_{}{}{}.npy'.format(Geophysical_version,YYYY,MM,YYYY,MM,DD),
        'eta'                   : Geophysical_indir + '{}/{}/{}/eta_{}{}{}.npy'.format(Geophysical_version,YYYY,MM,YYYY,MM,DD),
        'tSATAOD'               : Geophysical_indir + '{}/{}/{}/tSATAOD_fine0p01_{}{}{}.npy'.format(Geophysical_version,YYYY,MM,YYYY,MM,DD),
        'tSATPM25'              : Geophysical_indir + '{}/{}/{}/tSATPM25_{}{}{}.npy'.format(Geophysical_version,YYYY,MM,YYYY,MM,DD),
        ############################################################################################################
        # GEOS-Chem input variables
        'GC_PM25'               : GEOS_Chem_indir + '{}/{}/PM25_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'GC_SO4'                : GEOS_Chem_indir + '{}/{}/SO4_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'GC_NH4'                : GEOS_Chem_indir + '{}/{}/NH4_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'GC_NIT'                : GEOS_Chem_indir + '{}/{}/NIT_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'GC_BC'                 : GEOS_Chem_indir + '{}/{}/BC_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'GC_OM'                 : GEOS_Chem_indir + '{}/{}/OM_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'GC_SOA'                : GEOS_Chem_indir + '{}/{}/SOA_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'GC_DST'                : GEOS_Chem_indir + '{}/{}/DST_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'GC_SSLT'               : GEOS_Chem_indir + '{}/{}/SSLT_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),

        ############################################################################################################
        # Meteorology input variables
        'USTAR'                 : Meteorology_indir + '{}/{}/USTAR_GEOSFP_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'GWETTOP'               : Meteorology_indir + '{}/{}/GWETTOP_GEOSFP_001x00_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'PRECTOT'               : Meteorology_indir + '{}/{}/PRECTOT_GEOSFP_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'PBLH'                  : Meteorology_indir + '{}/{}/PBLH_GEOSFP_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'T2M'                   : Meteorology_indir + '{}/{}/T2M_GEOSFP_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'V10M'                  : Meteorology_indir + '{}/{}/V10M_GEOSFP_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'U10M'                  : Meteorology_indir + '{}/{}/U10M_GEOSFP_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'RH'                    : Meteorology_indir + '{}/{}/RH_GEOSFP_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),
        'PS'                    : Meteorology_indir + '{}/{}/PS_GEOSFP_001x001_NA_map_{}{}{}.npy'.format(YYYY,MM,YYYY,MM,DD),

        #########################################################################################################
        # Anthropogenic Emission input variables

        'NH3_anthro_emi'     : Anthropogenic_Emissions_input_indir + '{}/NH3-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'SO2_anthro_emi'     : Anthropogenic_Emissions_input_indir + '{}/SO2-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'NO_anthro_emi'      : Anthropogenic_Emissions_input_indir + '{}/NO-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'OC_anthro_emi'      : Anthropogenic_Emissions_input_indir + '{}/OC-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'BC_anthro_emi'      : Anthropogenic_Emissions_input_indir + '{}/BC-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'NMVOC_anthro_emi'   : Anthropogenic_Emissions_input_indir + '{}/NMVOC-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        
        #########################################################################################################
        # Variables from Offline Natural Emissions
        'DST_offline_emi'    : Offline_Emissions_input_indir + '{}/DST-em-EMI_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'SSLT_offline_emi'   : Offline_Emissions_input_indir + '{}/SSLT-em-EMI_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        
        ##################### [Variables from Land Cover] ###################
        'Crop_Nat_Vege_Mos'  : LandCover_input_indir + 'Cropland-Natural-Vegetation-Mosaics/Cropland-Natural-Vegetation-Mosaics-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        'Permanent_Wetlands' : LandCover_input_indir + 'Permanent-Wetlands/Permanent-Wetlands-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        'Croplands'          : LandCover_input_indir + 'Croplands/Croplands-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        'Urban_Builtup_Lands': LandCover_input_indir + 'Urban-Builtup-Lands/Urban-Builtup-Lands-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        
        ##################### [Open Street Map log data] ###################
        'log_major_roads'      : OpenStreetMap_log_road_indir + 'OpenStreetMap-major_roads-LogRoadMap_001x001.npy',
        'log_major_roads_dist' : OpenStreetMap_log_road_indir + 'OpenStreetMap-major_roads_NearestDistances-LogRoadMap_001x001.npy',
        'log_minor_roads'      : OpenStreetMap_log_road_indir + 'OpenStreetMap-minor_roads-LogRoadMap_001x001.npy',
        'log_minor_roads_dist' : OpenStreetMap_log_road_indir + 'OpenStreetMap-minor_roads_NearestDistances-LogRoadMap_001x001.npy',
        'log_motorway'         : OpenStreetMap_log_road_indir + 'OpenStreetMap-motorway-LogRoadMap_001x001.npy',
        'log_motorway_dist'    : OpenStreetMap_log_road_indir + 'OpenStreetMap-motorway_NearestDistances-LogRoadMap_001x001.npy',
        'log_primary'          : OpenStreetMap_log_road_indir + 'OpenStreetMap-primary-LogRoadMap_001x001.npy',
        'log_primary_dist'     : OpenStreetMap_log_road_indir + 'OpenStreetMap-primary_NearestDistances-LogRoadMap_001x001.npy',
        'log_secondary'        : OpenStreetMap_log_road_indir + 'OpenStreetMap-secondary-LogRoadMap_001x001.npy',
        'log_secondary_dist'   : OpenStreetMap_log_road_indir + 'OpenStreetMap-secondary_NearestDistances-LogRoadMap_001x001.npy',
        'log_trunk'            : OpenStreetMap_log_road_indir + 'OpenStreetMap-trunk-LogRoadMap_001x001.npy',
        'log_trunk_dist'       : OpenStreetMap_log_road_indir + 'OpenStreetMap-trunk_NearestDistances-LogRoadMap_001x001.npy',
        'log_unclassified'     : OpenStreetMap_log_road_indir + 'OpenStreetMap-unclassified-LogRoadMap_001x001.npy',
        'log_unclassified_dist': OpenStreetMap_log_road_indir + 'OpenStreetMap-unclassified_NearestDistances-LogRoadMap_001x001.npy',
        'log_residential'      : OpenStreetMap_log_road_indir + 'OpenStreetMap-residential-LogRoadMap_001x001.npy',
        'log_residential_dist' : OpenStreetMap_log_road_indir + 'OpenStreetMap-residential_NearestDistances-LogRoadMap_001x001.npy',

        ##################### [Open Street Map Road Density] ###################
        'major_roads'        : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-major_roads-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'minor_roads'        : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-minor_roads-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'motorway'           : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-motorway-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'primary'            : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-primary-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'secondary'          : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-secondary-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'trunk'              : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-trunk-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'unclassified'       : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-unclassified-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'residential'        : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-residential-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        
        ##################### [Open Street Map Road Density nearest distances] ###################
        'major_roads_dist'   : OpenStreetMap_nearest_dist_indir + 'major_roads/OpenStreetMap-major_roads-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'minor_roads_dist'   : OpenStreetMap_nearest_dist_indir + 'minor_roads/OpenStreetMap-minor_roads-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'motorway_dist'      : OpenStreetMap_nearest_dist_indir + 'motorway_NearestDistances/OpenStreetMap-motorway_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'primary_dist'       : OpenStreetMap_nearest_dist_indir + 'primary_NearestDistances/OpenStreetMap-primary_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'secondary_dist'     : OpenStreetMap_nearest_dist_indir + 'secondary_NearestDistances/OpenStreetMap-secondary_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'trunk_dist'         : OpenStreetMap_nearest_dist_indir + 'trunk_NearestDistances/OpenStreetMap-trunk_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'unclassified_dist'  : OpenStreetMap_nearest_dist_indir + 'unclassified_NearestDistances/OpenStreetMap-unclassified_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'residential_dist'   : OpenStreetMap_nearest_dist_indir + 'residential_NearestDistances/OpenStreetMap-residential_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        
        ###################### [Geographical Variables] ###################
        'elevation'          : Geographical_Variables_input_indir + 'elevation/elevartion_001x001_NA.npy',

        ###################### [Population Information] ####################
        'Population'         : Population_input_indir + 'WorldPopGrid-{}-0.01.npy'.format(YYYY),

        ###################### [Spatiotemporal Variables] ##################
        'lat'                : Spatiotemporal_input_indir + 'SAT_LAT_Map.npy',
        'lon'                : Spatiotemporal_input_indir + 'SAT_LON_Map.npy',
    }

    if judge_leap_year(YYYY):
        # Add leap year specific files
        inputfiles_dict.update({
            'sin_days' : Spatiotemporal_input_indir + 'Day_of_the_year/sin_days_of_the_Leap_year_{}{}.npy'.format(MM, DD),
            'cos_days' : Spatiotemporal_input_indir + 'Day_of_the_year/cos_days_of_the_Leap_year_{}{}.npy'.format(MM, DD),
            
        })
    else:
        # Add non-leap year specific files
        inputfiles_dict.update({
            'sin_days' : Spatiotemporal_input_indir + 'Day_of_the_year/sin_days_of_the_year_{}{}.npy'.format(MM, DD),
            'cos_days' : Spatiotemporal_input_indir + 'Day_of_the_year/cos_days_of_the_year_{}{}.npy'.format(MM, DD),
        })
    return inputfiles_dict

def judge_leap_year(year):
    year = int(year)
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False
def getGrg_YYYY_MM_DD(date):
    MONTHs = ['01','02','03','04','05','06','07','08','09','10','11','12']
    DAYs   = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
    YYYY = str(np.floor(date/10000).astype(int))
    MM = MONTHs[int(np.floor((date%10000)/100))-1]
    DD = DAYs[int(date%100)-1]
    return YYYY, MM, DD

def Grg2Jul(gdate):
    """
    Convert a date from YYYYMMDD format to YYYYDDD format.

    Args:
        gdate (int): The date in YYYYMMDD format.

    Returns:
        int: The date in YYYYDDD format.
    """
    date_str = str(gdate)
    date_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
    year = date_obj.year
    day_of_year = date_obj.timetuple().tm_yday
    return int(f"{year}{day_of_year:03d}")

def Jul2Grg(jdate):
    """
    Convert a date from YYYYDDD format to YYYYMMDD format.

    Args:
        jdate (int): The date in YYYYDDD format.

    Returns:
        int: The date in YYYYMMDD format.
    """
    date_str = str(jdate)
    year = int(date_str[:4])
    day_of_year = int(date_str[4:])
    date_obj = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
    return int(date_obj.strftime('%Y%m%d'))

