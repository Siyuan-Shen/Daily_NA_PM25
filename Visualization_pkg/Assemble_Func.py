import numpy as np
import os
from Evaluation_pkg.utils import get_YYYY_MM
from Visualization_pkg.Evaluation_plots import longterm_regression_plot,every_point_regression_plot
from Visualization_pkg.iostream import get_figure_outfile_path
from Training_pkg.utils import *
from Training_pkg.Statistic_func import regress2, linear_regression

def plot_longterm_Annual_Monthly_Daily_Scatter_plots(Evaluation_type,typeName,final_data_recording,obs_data_recording,
                                                     sites_recording, dates_recording,plot_begin_date,plot_end_date,nchannel,**args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)
    
    fig_outdir = figure_outdir + '{}/{}/Figures/Scatter_Plot/{}/'.format(species, version,Evaluation_type)
    if not os.path.exists(fig_outdir):
        os.makedirs(fig_outdir)
    temp_index = temp_index = np.where((dates_recording >= plot_begin_date) & (dates_recording <= plot_end_date))[0]
    YYYY_series_dict,MM_series_dict, total_unique_YYYY = get_YYYY_MM(plot_begin_date, plot_end_date)
    ### First print the all data points daily scatter plot
    print('Plotting the daily scatter plot for {} from {} to {}'.format(species, plot_begin_date, plot_end_date))
    every_daily_point_outfile = get_figure_outfile_path(outdir=fig_outdir,evaluation_type=Evaluation_type,
                                                        figure_type='EveryDailyPoints',typeName=typeName,
                                                        begindate=plot_begin_date,enddate=plot_end_date,
                                                        nchannel=nchannel, width=width, height=height, depth=depth,)
    temp_final_data = final_data_recording[temp_index].copy()
    temp_obs_data = obs_data_recording[temp_index].copy()
    every_point_regression_plot(plot_obs_pm25=temp_obs_data,plot_pre_pm25=temp_final_data,
                    species=species,outfile=every_daily_point_outfile)
    

    ### Then print monthly average scatter plot
    print('Plotting the monthly scatter plot for {} from {} to {}'.format(species, plot_begin_date, plot_end_date))
    MONTHs = ['01','02','03','04','05','06','07','08','09','10','11','12']
    DAYs   = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
    
    
    every_monthly_point_outfile = get_figure_outfile_path(outdir=fig_outdir,evaluation_type=Evaluation_type,
                                                        figure_type='EveryMonthlyPoints',typeName=typeName,
                                                        begindate=plot_begin_date,enddate=plot_end_date,
                                                        nchannel=nchannel, width=width, height=height, depth=depth,)
    
    Allpoints_monthly_temp_final_data = np.array([],dtype=np.float64)
    Allpoints_monthly_temp_obs_data = np.array([],dtype=np.float64)
    for imonth, MM in enumerate(MONTHs):
        unique_YYYY = MM_series_dict[MM]['YYYY']
        for iyear, YYYY in enumerate(unique_YYYY):
            temp_final_data = np.array([],dtype=np.float64)
            temp_obs_data = np.array([],dtype=np.float64)

            start_day = MM_series_dict[MM]['start_day'][YYYY]
            end_day = MM_series_dict[MM]['end_day'][YYYY]
            start_date = int(YYYY) * 10000 + int(MM) * 100 + start_day
            end_date = int(YYYY) * 10000 + int(MM) * 100 + end_day
            temp_index = np.where((dates_recording >= start_date) & (dates_recording <= end_date))[0]
            temp_sites = np.unique(sites_recording[temp_index])
            
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
                        temp_final_data = np.concatenate((temp_final_data, [temp_monthly_final_data]))
                        temp_obs_data = np.concatenate((temp_obs_data, [temp_monthly_obs_data]))
            Allpoints_monthly_temp_final_data = np.concatenate((Allpoints_monthly_temp_final_data, temp_final_data))
            Allpoints_monthly_temp_obs_data = np.concatenate((Allpoints_monthly_temp_obs_data, temp_obs_data))
    
    every_point_regression_plot(plot_obs_pm25=Allpoints_monthly_temp_obs_data,
                                plot_pre_pm25=Allpoints_monthly_temp_final_data,
                                species=species, 
                                outfile=every_monthly_point_outfile)
    
    ### Then print annual average scatter plot
    print('Plotting the annual scatter plot for {} from {} to {}'.format(species, plot_begin_date, plot_end_date))
    every_annual_point_outfile = get_figure_outfile_path(outdir=fig_outdir,evaluation_type=Evaluation_type,
                                                        figure_type='EveryAnnualPoints',typeName=typeName,
                                                        begindate=plot_begin_date,enddate=plot_end_date,
                                                        nchannel=nchannel, width=width, height=height, depth=depth,)
    Allpoints_annual_final_data = np.array([],dtype=np.float64)
    Allpoints_annual_obs_data = np.array([],dtype=np.float64)
    for iyear, YYYY in enumerate(total_unique_YYYY):

        temp_final_data = np.array([],dtype=np.float64)
        temp_obs_data = np.array([],dtype=np.float64)
        temp_start_date = YYYY_series_dict[YYYY]['start_date']
        temp_end_date = YYYY_series_dict[YYYY]['end_date']
        temp_index = np.where((dates_recording >= temp_start_date) & (dates_recording <= temp_end_date))[0]
        temp_sites = np.unique(sites_recording[temp_index])
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
                    temp_final_data = np.concatenate((temp_final_data, [temp_annual_final_data]))
                    temp_obs_data = np.concatenate((temp_obs_data, [temp_annual_obs_data]))
            Allpoints_annual_final_data = np.concatenate((Allpoints_annual_final_data, temp_final_data))
            Allpoints_annual_obs_data = np.concatenate((Allpoints_annual_obs_data, temp_obs_data))
    every_point_regression_plot(plot_obs_pm25=Allpoints_annual_obs_data,
                                plot_pre_pm25=Allpoints_annual_final_data,
                                species=species,
                                outfile=every_annual_point_outfile)
    
    ### Then plot the longterm scatter plot
    print('Plotting the longterm scatter plot for {} from {} to {}'.format(species, plot_begin_date, plot_end_date))
    longterm_final_data = np.array([],dtype=np.float64)
    longterm_obs_data = np.array([],dtype=np.float64)
    temp_index = np.where((dates_recording >= plot_begin_date) & (dates_recording <= plot_end_date))[0]
    temp_sites = np.unique(sites_recording[temp_index])
    longterm_fig_outfile = get_figure_outfile_path(outdir=fig_outdir,evaluation_type=Evaluation_type,
                                                   figure_type='Longterm',typeName=typeName,
                                                   begindate=plot_begin_date,enddate=plot_end_date,
                                                   nchannel=nchannel, width=width, height=height, depth=depth,)
    for isite in temp_sites:
        temp_sites_index = np.where(sites_recording[temp_index] == isite)[0]
        
        temp_longterm_final_data = np.mean(final_data_recording[temp_index][temp_sites_index].copy())
        temp_longterm_obs_data = np.mean(obs_data_recording[temp_index][temp_sites_index].copy())
        longterm_final_data = np.concatenate((longterm_final_data, [temp_longterm_final_data]))
        longterm_obs_data = np.concatenate((longterm_obs_data, [temp_longterm_obs_data]))
    
    longterm_regression_plot(plot_obs_pm25=longterm_obs_data, plot_pre_pm25=longterm_final_data,
                             species=species, outfile=longterm_fig_outfile,)
    return


def plot_timeseries_statistics_plots():
    #### Plot Daily timeseries and monthly whole range R2/rRMSE plots in different regions

    return

def plot_R2_rRMSE_Spatial_Distribution_plots():
    #### Plot R2/rRMSE at each sites on the map

    return