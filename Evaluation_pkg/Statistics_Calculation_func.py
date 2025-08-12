import numpy as np
import math
from Training_pkg.Statistic_func import *
from Evaluation_pkg.utils import initialize_statistics_recordings,create_date_range,get_YYYY_MM

def calculate_statistics(test_begindates,test_enddates, final_data_recording,obs_data_recording,geo_data_recording,sites_recording,dates_recording,
                         training_final_data_recording,training_obs_data_recording,training_sites_recording,training_dates_recording,Statistics_list):
    Daily_statistics_recording, Monthly_statistics_recording, Annual_statistics_recording = initialize_statistics_recordings(test_begindates,test_enddates,Statistics_list=Statistics_list)
    sites_index = np.unique(sites_recording)
    print('test_begindates type:', type(test_begindates))
    print('test_begindates:', test_begindates)
    dates, dates_series = create_date_range(test_begindates, test_enddates)
    MONTHs = ['01','02','03','04','05','06','07','08','09','10','11','12']
    DAYs   = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
    print('test_begindates type:', type(test_begindates))
    print('test_begindates:', test_begindates)
    YYYY_series_dict,MM_series_dict, total_unique_YYYY = get_YYYY_MM(test_begindates, test_enddates)
    ## Exclude the Nan values from the final_data_recording, obs_data_recording, geo_data_recording, training_final_data_recording, training_obs_data_recording
    nan_indices = np.isnan(obs_data_recording)
    training_nan_indices = np.isnan(training_obs_data_recording)

    final_data_recording = final_data_recording[~nan_indices]
    obs_data_recording = obs_data_recording[~nan_indices]
    geo_data_recording = geo_data_recording[~nan_indices]
    training_final_data_recording = training_final_data_recording[~training_nan_indices]
    training_obs_data_recording = training_obs_data_recording[~training_nan_indices]

    dates_recording = dates_recording[~nan_indices]
    sites_recording = sites_recording[~nan_indices]
    
    training_dates_recording = training_dates_recording[~training_nan_indices]
    training_sites_recording = training_sites_recording[~training_nan_indices]

    # print sizes:
    print('final_data_recording size:', final_data_recording.shape)
    print('obs_data_recording size:', obs_data_recording.shape)
    print('geo_data_recording size:', geo_data_recording.shape)
    print('training_final_data_recording size:', training_final_data_recording.shape)
    print('training_obs_data_recording size:', training_obs_data_recording.shape)
    print('dates_recording size:', dates_recording.shape)
    print('sites_recording size:', sites_recording.shape)
    print('training_dates_recording size:', training_dates_recording.shape)
    print('training_sites_recording size:', training_sites_recording.shape)
    # Daily statistics
    # Purely_Spatial statistics
    for iday in dates_series:
        ## We get the statistics for each day, and then we can average them to get the average statistics for 
        ## purely spatial daily statistics.
        temp_index = np.where(dates_recording == iday)[0]
        temp_final_data = final_data_recording[temp_index].copy()
        temp_obs_data = obs_data_recording[temp_index].copy()
        temp_geo_data = geo_data_recording[temp_index].copy()
        temp_training_index = np.where(training_dates_recording == iday)[0]
        temp_training_final_data = training_final_data_recording[temp_training_index].copy()
        temp_training_obs_data = training_obs_data_recording[temp_training_index].copy()
        
        if len(temp_index) > 0 and len(temp_training_index) > 0:
            temp_test_R2, temp_train_R2, temp_geo_R2, temp_RMSE, temp_NRMSE, intercept, slope = calculation_process_of_statistics(temp_final_data, temp_obs_data, temp_geo_data, temp_training_final_data, temp_training_obs_data)
            print('iday: ', iday, 'temp_test_R2:', temp_test_R2, 'temp_train_R2:', temp_train_R2, 'temp_geo_R2:', temp_geo_R2, 'temp_RMSE:', temp_RMSE, 'temp_NRMSE:', temp_NRMSE)
            if temp_test_R2 > 0:
                Daily_statistics_recording['Purely_Spatial']['test_R2'] = np.concatenate((Daily_statistics_recording['Purely_Spatial'].get('test_R2', []), [temp_test_R2]))
            if temp_RMSE > 0:
                Daily_statistics_recording['Purely_Spatial']['RMSE'] = np.concatenate((Daily_statistics_recording['Purely_Spatial'].get('RMSE', []), [temp_RMSE]))
            if temp_NRMSE > 0:
                Daily_statistics_recording['Purely_Spatial']['NRMSE'] = np.concatenate((Daily_statistics_recording['Purely_Spatial'].get('NRMSE', []), [temp_NRMSE]))
            if slope > 0:
                Daily_statistics_recording['Purely_Spatial']['slope'] = np.concatenate((Daily_statistics_recording['Purely_Spatial'].get('slope', []), [slope]))
            if temp_train_R2 > 0:
                Daily_statistics_recording['Purely_Spatial']['train_R2'] = np.concatenate((Daily_statistics_recording['Purely_Spatial'].get('train_R2', []), [temp_train_R2]))
            if temp_geo_R2 > 0:
                Daily_statistics_recording['Purely_Spatial']['geo_R2'] = np.concatenate((Daily_statistics_recording['Purely_Spatial'].get('geo_R2', []), [temp_geo_R2]))
            
        else:
            continue

    # All_points statistics
    temp_index = np.where((dates_recording >= test_begindates) & (dates_recording <= test_enddates))[0]
    temp_training_index = np.where((training_dates_recording >= test_begindates) & (training_dates_recording <= test_enddates))[0]
    temp_final_data = final_data_recording[temp_index].copy()
    temp_obs_data = obs_data_recording[temp_index].copy()
    temp_geo_data = geo_data_recording[temp_index].copy()
    temp_training_final_data = training_final_data_recording[temp_training_index].copy()
    temp_training_obs_data = training_obs_data_recording[temp_training_index].copy()
    if len(temp_index) > 0 and len(temp_training_index) > 0:
        temp_test_R2, temp_train_R2, temp_geo_R2, temp_RMSE, temp_NRMSE, intercept, slope = calculation_process_of_statistics(temp_final_data, temp_obs_data, temp_geo_data, temp_training_final_data, temp_training_obs_data)
        Daily_statistics_recording['All_points']['test_R2'] = temp_test_R2
        Daily_statistics_recording['All_points']['RMSE'] = temp_RMSE
        Daily_statistics_recording['All_points']['NRMSE'] = temp_NRMSE
        Daily_statistics_recording['All_points']['slope'] = slope
        Daily_statistics_recording['All_points']['train_R2'] = temp_train_R2
        Daily_statistics_recording['All_points']['geo_R2'] = temp_geo_R2
    else:
        None

    # Daily in Monthly_Scale

    for imonth, MM in enumerate(MONTHs):
        temp_final_data = np.array([],dtype=np.float64)
        temp_obs_data = np.array([],dtype=np.float64)
        temp_geo_data = np.array([],dtype=np.float64)
        temp_training_final_data = np.array([],dtype=np.float64)
        temp_training_obs_data = np.array([],dtype=np.float64)
        unique_YYYY = MM_series_dict[MM]['YYYY']
        for iyear,YYYY in enumerate(unique_YYYY):
            start_day = MM_series_dict[MM]['start_day'][YYYY]
            end_day = MM_series_dict[MM]['end_day'][YYYY]
            start_date = int(YYYY) * 10000 + int(MM) * 100 + start_day
            end_date = int(YYYY) * 10000 + int(MM) * 100 + end_day
            temp_index = np.where((dates_recording >= start_date) & (dates_recording <= end_date))[0]
            temp_training_index = np.where((training_dates_recording >= start_date) & (training_dates_recording <= end_date))[0]
            temp_final_data = np.concatenate((temp_final_data, final_data_recording[temp_index].copy()))
            temp_obs_data = np.concatenate((temp_obs_data, obs_data_recording[temp_index].copy()))
            temp_geo_data = np.concatenate((temp_geo_data, geo_data_recording[temp_index].copy()))
            temp_training_final_data = np.concatenate((temp_training_final_data, training_final_data_recording[temp_training_index].copy()))
            temp_training_obs_data = np.concatenate((temp_training_obs_data, training_obs_data_recording[temp_training_index].copy()))
                
        temp_test_R2, temp_train_R2, temp_geo_R2, temp_RMSE, temp_NRMSE, intercept, slope = calculation_process_of_statistics(temp_final_data, temp_obs_data, temp_geo_data, temp_training_final_data, temp_training_obs_data)
        
        Daily_statistics_recording['Monthly_Scale'][MM]['test_R2'] = temp_test_R2
        Daily_statistics_recording['Monthly_Scale'][MM]['RMSE'] = temp_RMSE
        Daily_statistics_recording['Monthly_Scale'][MM]['NRMSE'] = temp_NRMSE
        Daily_statistics_recording['Monthly_Scale'][MM]['slope'] = slope
        Daily_statistics_recording['Monthly_Scale'][MM]['train_R2'] = temp_train_R2
        Daily_statistics_recording['Monthly_Scale'][MM]['geo_R2'] = temp_geo_R2
    
    # Monthly statistics
    # All_points statistics and Purely_Spatial statistics
    Allpoints_AllMonths_temp_final_data = np.array([],dtype=np.float64)
    Allpoints_AllMonths_temp_obs_data = np.array([],dtype=np.float64)
    Allpoints_AllMonths_temp_geo_data = np.array([],dtype=np.float64)
    Allpoints_AllMonths_temp_training_final_data = np.array([],dtype=np.float64)
    Allpoints_AllMonths_temp_training_obs_data = np.array([],dtype=np.float64)

    for imonth, MM in enumerate(MONTHs):
        Allpoints_monthly_temp_final_data = np.array([],dtype=np.float64)
        Allpoints_monthly_temp_obs_data = np.array([],dtype=np.float64)
        Allpoints_monthly_temp_geo_data = np.array([],dtype=np.float64)
        Allpoints_monthly_temp_training_final_data = np.array([],dtype=np.float64)
        Allpoints_monthly_temp_training_obs_data = np.array([],dtype=np.float64)

        unique_YYYY = MM_series_dict[MM]['YYYY']
        for iyear, YYYY in enumerate(unique_YYYY):
            temp_final_data = np.array([],dtype=np.float64)
            temp_obs_data = np.array([],dtype=np.float64)
            temp_geo_data = np.array([],dtype=np.float64)
            temp_training_final_data = np.array([],dtype=np.float64)
            temp_training_obs_data = np.array([],dtype=np.float64)

            start_day = MM_series_dict[MM]['start_day'][YYYY]
            end_day = MM_series_dict[MM]['end_day'][YYYY]
            start_date = int(YYYY) * 10000 + int(MM) * 100 + start_day
            end_date = int(YYYY) * 10000 + int(MM) * 100 + end_day
            temp_index = np.where((dates_recording >= start_date) & (dates_recording <= end_date))[0]
            temp_training_index = np.where((training_dates_recording >= start_date) & (training_dates_recording <= end_date))[0]
            temp_sites = np.unique(sites_recording[temp_index])
            temp_train_sites = np.unique(training_sites_recording[temp_training_index])
            if len(temp_sites) < 1:
                continue
            else:
                for isite in temp_sites:
                    temp_sites_index = np.where(sites_recording[temp_index] == isite)[0]
                    if len(temp_sites_index) < 14:
                        continue
                    else:
                        temp_monthly_final_data =np.mean(final_data_recording[temp_index][temp_sites_index].copy())
                        temp_monthly_obs_data = np.mean(obs_data_recording[temp_index][temp_sites_index].copy())
                        temp_monthly_geo_data = np.mean(geo_data_recording[temp_index][temp_sites_index].copy())
                        temp_final_data = np.concatenate((temp_final_data, [temp_monthly_final_data]))
                        temp_obs_data = np.concatenate((temp_obs_data, [temp_monthly_obs_data]))
                        temp_geo_data = np.concatenate((temp_geo_data, [temp_monthly_geo_data]))
                 
            if len(temp_train_sites) < 1:
                continue
            else:
                for isite in temp_train_sites:
                    temp_train_sites_index = np.where(training_sites_recording[temp_training_index] == isite)[0]
                    if len(temp_train_sites_index) < 14:
                        continue
                    else:
                        temp_monthly_training_final_data = np.mean(training_final_data_recording[temp_training_index][temp_train_sites_index].copy())
                        temp_monthly_training_obs_data = np.mean(training_obs_data_recording[temp_training_index][temp_train_sites_index].copy())
                        temp_training_final_data = np.concatenate((temp_training_final_data, [temp_monthly_training_final_data]))
                        temp_training_obs_data = np.concatenate((temp_training_obs_data, [temp_monthly_training_obs_data]))
            
            temp_test_R2, temp_train_R2, temp_geo_R2, temp_RMSE, temp_NRMSE, intercept, slope = calculation_process_of_statistics(temp_final_data, temp_obs_data, temp_geo_data, temp_training_final_data, temp_training_obs_data)
            print('YYYY:', YYYY, 'MM:', MM, 'temp_test_R2:', temp_test_R2, 'temp_train_R2:', temp_train_R2, 'temp_geo_R2:', temp_geo_R2, 'temp_RMSE:', temp_RMSE, 'temp_NRMSE:', temp_NRMSE)
            if temp_test_R2 > 0:
                Monthly_statistics_recording['Purely_Spatial'][MM]['test_R2'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial'][MM].get('test_R2', []), [temp_test_R2]))
                Monthly_statistics_recording['Purely_Spatial']['AllMonths']['test_R2'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial']['AllMonths'].get('test_R2', []), [temp_test_R2]))
            if temp_RMSE > 0:
                Monthly_statistics_recording['Purely_Spatial'][MM]['RMSE'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial'][MM].get('RMSE', []), [temp_RMSE]))
                Monthly_statistics_recording['Purely_Spatial']['AllMonths']['RMSE'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial']['AllMonths'].get('RMSE', []), [temp_RMSE]))    
            if temp_NRMSE > 0:
                Monthly_statistics_recording['Purely_Spatial'][MM]['NRMSE'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial'][MM].get('NRMSE', []), [temp_NRMSE]))
                Monthly_statistics_recording['Purely_Spatial']['AllMonths']['NRMSE'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial']['AllMonths'].get('NRMSE', []), [temp_NRMSE]))
            if slope > 0:
                Monthly_statistics_recording['Purely_Spatial'][MM]['slope'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial'][MM].get('slope', []), [slope]))
                Monthly_statistics_recording['Purely_Spatial']['AllMonths']['slope'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial']['AllMonths'].get('slope', []), [slope]))  
            if temp_train_R2 > 0:
                Monthly_statistics_recording['Purely_Spatial'][MM]['train_R2'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial'][MM].get('train_R2', []), [temp_train_R2]))
                Monthly_statistics_recording['Purely_Spatial']['AllMonths']['train_R2'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial']['AllMonths'].get('train_R2', []), [temp_train_R2]))    
            if temp_geo_R2 > 0:
                Monthly_statistics_recording['Purely_Spatial'][MM]['geo_R2'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial'][MM].get('geo_R2', []), [temp_geo_R2]))
                Monthly_statistics_recording['Purely_Spatial']['AllMonths']['geo_R2'] = np.concatenate((Monthly_statistics_recording['Purely_Spatial']['AllMonths'].get('geo_R2', []), [temp_geo_R2]))
            

            Allpoints_monthly_temp_final_data = np.concatenate((Allpoints_monthly_temp_final_data, temp_final_data))
            Allpoints_AllMonths_temp_final_data = np.concatenate((Allpoints_AllMonths_temp_final_data, temp_final_data))

            Allpoints_monthly_temp_obs_data = np.concatenate((Allpoints_monthly_temp_obs_data, temp_obs_data))
            Allpoints_AllMonths_temp_obs_data = np.concatenate((Allpoints_AllMonths_temp_obs_data, temp_obs_data))

            Allpoints_monthly_temp_geo_data = np.concatenate((Allpoints_monthly_temp_geo_data, temp_geo_data))
            Allpoints_AllMonths_temp_geo_data = np.concatenate((Allpoints_AllMonths_temp_geo_data, temp_geo_data))

            Allpoints_monthly_temp_training_final_data = np.concatenate((Allpoints_monthly_temp_training_final_data, temp_training_final_data))
            Allpoints_AllMonths_temp_training_final_data = np.concatenate((Allpoints_AllMonths_temp_training_final_data, temp_training_final_data))

            Allpoints_monthly_temp_training_obs_data = np.concatenate((Allpoints_monthly_temp_training_obs_data, temp_training_obs_data))
            Allpoints_AllMonths_temp_training_obs_data = np.concatenate((Allpoints_AllMonths_temp_training_obs_data, temp_training_obs_data))

        Allpoints_Monthly_test_R2, Allpoints_Monthly_train_R2, Allpoints_Monthly_geo_R2, Allpoints_Monthly_RMSE, Allpoints_Monthly_NRMSE, intercept, slope = calculation_process_of_statistics(Allpoints_monthly_temp_final_data, Allpoints_monthly_temp_obs_data, Allpoints_monthly_temp_geo_data, Allpoints_monthly_temp_training_final_data, Allpoints_monthly_temp_training_obs_data)
        Monthly_statistics_recording['All_points'][MM]['test_R2'] = Allpoints_Monthly_test_R2
        Monthly_statistics_recording['All_points'][MM]['RMSE'] = Allpoints_Monthly_RMSE
        Monthly_statistics_recording['All_points'][MM]['NRMSE'] = Allpoints_Monthly_NRMSE
        Monthly_statistics_recording['All_points'][MM]['slope'] = slope
        Monthly_statistics_recording['All_points'][MM]['train_R2'] = Allpoints_Monthly_train_R2
        Monthly_statistics_recording['All_points'][MM]['geo_R2'] = Allpoints_Monthly_geo_R2
        
    Allpoints_Monthly_test_R2, Allpoints_Monthly_train_R2, Allpoints_Monthly_geo_R2, Allpoints_Monthly_RMSE, Allpoints_Monthly_NRMSE, intercept, slope = calculation_process_of_statistics(Allpoints_AllMonths_temp_final_data, Allpoints_AllMonths_temp_obs_data, Allpoints_AllMonths_temp_geo_data, Allpoints_AllMonths_temp_training_final_data, Allpoints_AllMonths_temp_training_obs_data)
    Monthly_statistics_recording['All_points']['AllMonths']['test_R2'] = Allpoints_Monthly_test_R2
    Monthly_statistics_recording['All_points']['AllMonths']['RMSE'] = Allpoints_Monthly_RMSE
    Monthly_statistics_recording['All_points']['AllMonths']['NRMSE'] = Allpoints_Monthly_NRMSE
    Monthly_statistics_recording['All_points']['AllMonths']['slope'] = slope
    Monthly_statistics_recording['All_points']['AllMonths']['train_R2'] = Allpoints_Monthly_train_R2
    Monthly_statistics_recording['All_points']['AllMonths']['geo_R2'] = Allpoints_Monthly_geo_R2


    # Annual statistics
    # All_points statistics and Purely_Spatial statistics 
    Allpoints_annual_final_data = np.array([],dtype=np.float64)
    Allpoints_annual_obs_data = np.array([],dtype=np.float64)
    Allpoints_annual_geo_data = np.array([],dtype=np.float64)
    Allpoints_annual_training_final_data = np.array([],dtype=np.float64)
    Allpoints_annual_training_obs_data = np.array([],dtype=np.float64)

    for iyear, YYYY in enumerate(total_unique_YYYY):

        temp_final_data = np.array([],dtype=np.float64)
        temp_obs_data = np.array([],dtype=np.float64)
        temp_geo_data = np.array([],dtype=np.float64)
        temp_training_final_data = np.array([],dtype=np.float64)
        temp_training_obs_data = np.array([],dtype=np.float64)
        
        temp_start_date = YYYY_series_dict[YYYY]['start_date']
        temp_end_date = YYYY_series_dict[YYYY]['end_date']
        temp_index = np.where((dates_recording >= temp_start_date) & (dates_recording <= temp_end_date))[0]
        temp_training_index = np.where((training_dates_recording >= temp_start_date) & (training_dates_recording <= temp_end_date))[0]
        temp_sites = np.unique(sites_recording[temp_index])
        temp_train_sites = np.unique(training_sites_recording[temp_training_index])
        if len(temp_sites) < 1:
            continue
        else:
            for isite in temp_sites:
                temp_sites_index = np.where(sites_recording[temp_index] == isite)[0]
                if len(temp_sites_index) < 182:
                    continue
                else:
                    temp_annual_final_data = np.mean(final_data_recording[temp_index][temp_sites_index].copy())
                    temp_annual_obs_data = np.mean(obs_data_recording[temp_index][temp_sites_index].copy())
                    temp_annual_geo_data = np.mean(geo_data_recording[temp_index][temp_sites_index].copy())
                    temp_final_data = np.concatenate((temp_final_data, [temp_annual_final_data]))
                    temp_obs_data = np.concatenate((temp_obs_data, [temp_annual_obs_data]))
                    temp_geo_data = np.concatenate((temp_geo_data, [temp_annual_geo_data]))
            Allpoints_annual_final_data = np.concatenate((Allpoints_annual_final_data, temp_final_data))
            Allpoints_annual_obs_data = np.concatenate((Allpoints_annual_obs_data, temp_obs_data))
            Allpoints_annual_geo_data = np.concatenate((Allpoints_annual_geo_data, temp_geo_data))
        if len(temp_train_sites) < 1:
            continue
        else:
            for isite in temp_train_sites:
                temp_train_sites_index = np.where(training_sites_recording[temp_training_index] == isite)[0]
                if len(temp_train_sites_index) < 182:
                    continue
                else:
                    temp_annual_training_final_data = np.mean(training_final_data_recording[temp_training_index][temp_train_sites_index].copy())
                    temp_annual_training_obs_data = np.mean(training_obs_data_recording[temp_training_index][temp_train_sites_index].copy())
                    temp_training_final_data = np.concatenate((temp_training_final_data, [temp_annual_training_final_data]))
                    temp_training_obs_data = np.concatenate((temp_training_obs_data, [temp_annual_training_obs_data]))
            Allpoints_annual_training_final_data = np.concatenate((Allpoints_annual_training_final_data, temp_training_final_data))
            Allpoints_annual_training_obs_data = np.concatenate((Allpoints_annual_training_obs_data, temp_training_obs_data))
        annual_test_R2, annual_train_R2, annual_geo_R2, annual_RMSE, annual_NRMSE, intercept, slope = calculation_process_of_statistics(temp_final_data, temp_obs_data, temp_geo_data, temp_training_final_data, temp_training_obs_data)
        print('YYYY:', YYYY, 'annual_test_R2:', annual_test_R2, 'annual_train_R2:', annual_train_R2, 'annual_geo_R2:', annual_geo_R2, 'annual_RMSE:', annual_RMSE, 'annual_NRMSE:', annual_NRMSE)
        if annual_test_R2 > 0:
            Annual_statistics_recording['Purely_Spatial']['test_R2'] = np.concatenate((Annual_statistics_recording['Purely_Spatial'].get('test_R2', []), [annual_test_R2]))
        if annual_RMSE > 0:
            Annual_statistics_recording['Purely_Spatial']['RMSE'] = np.concatenate((Annual_statistics_recording['Purely_Spatial'].get('RMSE', []), [annual_RMSE]))
        if annual_NRMSE > 0:
            Annual_statistics_recording['Purely_Spatial']['NRMSE'] = np.concatenate((Annual_statistics_recording['Purely_Spatial'].get('NRMSE', []), [annual_NRMSE]))
        if slope > 0:
            Annual_statistics_recording['Purely_Spatial']['slope'] = np.concatenate((Annual_statistics_recording['Purely_Spatial'].get('slope', []), [slope]))
        if annual_train_R2 > 0:
            Annual_statistics_recording['Purely_Spatial']['train_R2'] = np.concatenate((Annual_statistics_recording['Purely_Spatial'].get('train_R2', []), [annual_train_R2]))
        if annual_geo_R2 > 0:
            Annual_statistics_recording['Purely_Spatial']['geo_R2'] = np.concatenate((Annual_statistics_recording['Purely_Spatial'].get('geo_R2', []), [annual_geo_R2]))
    Allpoints_annual_test_R2, Allpoints_annual_train_R2, Allpoints_annual_geo_R2, Allpoints_annual_RMSE, Allpoints_annual_NRMSE, intercept, slope = calculation_process_of_statistics(Allpoints_annual_final_data, Allpoints_annual_obs_data, Allpoints_annual_geo_data, Allpoints_annual_training_final_data, Allpoints_annual_training_obs_data)
    Annual_statistics_recording['All_points']['test_R2'] = Allpoints_annual_test_R2
    Annual_statistics_recording['All_points']['RMSE'] = Allpoints_annual_RMSE
    Annual_statistics_recording['All_points']['NRMSE'] = Allpoints_annual_NRMSE
    Annual_statistics_recording['All_points']['slope'] = slope
    Annual_statistics_recording['All_points']['train_R2'] = Allpoints_annual_train_R2
    Annual_statistics_recording['All_points']['geo_R2'] = Allpoints_annual_geo_R2
    # Return the statistics

    return Daily_statistics_recording, Monthly_statistics_recording, Annual_statistics_recording

def calculation_process_of_statistics(temp_final_data, temp_obs_data, temp_geo_data, temp_training_final_data, temp_training_obs_data):
    # Calculate statistics
    temp_test_R2 = linear_regression(temp_final_data, temp_obs_data)
    temp_train_R2 = linear_regression(temp_training_final_data, temp_training_obs_data)
    temp_geo_R2 = linear_regression(temp_obs_data, temp_geo_data)
    temp_RMSE = Cal_RMSE(temp_final_data, temp_obs_data)
    temp_NRMSE = Cal_NRMSE(temp_final_data, temp_obs_data)
    regression_Dic = regress2(_x=temp_obs_data,_y=temp_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis')
    intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
    return temp_test_R2, temp_train_R2, temp_geo_R2, temp_RMSE, temp_NRMSE, intercept, slope