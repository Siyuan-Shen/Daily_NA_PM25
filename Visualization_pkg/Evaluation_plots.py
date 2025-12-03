import os
import shap
import numpy as np
import math
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from sklearn.metrics import mean_squared_error,r2_score
from matplotlib.ticker import ScalarFormatter
from Training_pkg.Statistic_func import linear_regression, regress2
from Visualization_pkg.utils import return_sign
from Training_pkg.utils import *
from Evaluation_pkg.utils import calculate_distance_forArray

nrows = 2
ncols = 2
proj = ccrs.PlateCarree()
aspect = (179)/(60+70)
height = 5.0
width = aspect * height
vpad = 0.03 * height
hpad = 0.02 * width
hlabel = 0.12 * height*2
vlabel = 0.1 * height*2
hmargin = 0.03 * width
vmargin = 0.03 * height*2
cbar_height = 0.48 * height
cbar_width = 0.015 * width
cbar_height_2 = 0.9 * (height*2 - vlabel)
cbar_width_2 = 0.08 * (width + height*2)

figwidth = width + height + hmargin*2 + cbar_width_2
figheight = height*2 + vmargin*2

def shap_value_plot(shap_values_with_feature_names:shap._explanation.Explanation,plot_type:str,outfile:str):
    if plot_type == 'beeswarm':
        
        shap.plots.beeswarm(shap_values_with_feature_names, show=False)

        # Get the current figure and axes for customization
        fig = plt.gcf()

        #cbar = fig.axes[-1]  # The colorbar is usually the last axis in the figure
        #cbar.set_ylim(-1,1)
        

        cbar = fig.get_axes()[-1]  # Retrieve the colorbar axis
        cbar.set_yticks([ 0, 1.0])
        cbar.set_yticklabels(["0","1"])
        plt.xlabel(r'Impact on PM$_{2.5}$ ($\mu g/m^3$)')
        cbar.set_ylabel('Predictor variables values')
        plt.savefig(outfile,format='png',dpi=1000, bbox_inches='tight')
        plt.close()
    return

def every_point_regression_plot(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    species,outfile):
    if species == 'PM25':
        tag = r'$PM_{2.5}$'
    every_point_plot_obs_pm25 = plot_obs_pm25
    every_point_plot_pre_pm25 = plot_pre_pm25
 

    index = np.where(~np.isnan(every_point_plot_obs_pm25) & ~np.isnan(every_point_plot_pre_pm25))
    every_point_plot_obs_pm25 = every_point_plot_obs_pm25[index]
    every_point_plot_pre_pm25 = every_point_plot_pre_pm25[index]
    H, xedges, yedges = np.histogram2d(every_point_plot_obs_pm25, every_point_plot_pre_pm25, bins=100)
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(figwidth, figheight))
    extent = [0, max(xedges), 0, max(xedges)]
    RMSE = np.sqrt(mean_squared_error(every_point_plot_obs_pm25, every_point_plot_pre_pm25))
    RMSE = round(RMSE, 1)

    R2 = linear_regression(every_point_plot_obs_pm25, every_point_plot_pre_pm25)
    R2 = np.round(R2, 2)

    ax = plt.axes([0.1,0.1,0.8,0.8])  # [left, bottom, width, height]
    cbar_ax = plt.axes([0.78,0.2,0.015,0.45])
    regression_Dic = regress2(_x=every_point_plot_obs_pm25,_y=every_point_plot_pre_pm25,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
    b0,b1 = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    b0 = round(b0, 2)
    b1 = round(b1, 2)

    extentlim = 10*np.mean(every_point_plot_obs_pm25)
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(every_point_plot_obs_pm25, every_point_plot_pre_pm25,
                   cmap='cool', norm=colors.LogNorm(vmin=1, vmax=10000), extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([0, extentlim], [0, extentlim], color='black', linestyle='--')
    ax.plot([0, extentlim], [b0, b0 + b1 * extentlim], color='pink', linestyle='-',linewidth=2)
    #ax.set_title('Comparsion of Modeled $PM_{2.5}$ and observations for '+area_name+' '+beginyear+' '+endyear)
    ax.set_xlabel('Observed {} concentration ($\mu g/m^3$)'.format(tag), fontsize=28)
    ax.set_ylabel('Estimated {} concentration ($\mu g/m^3$)'.format(tag), fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)

    ax.text(0, extentlim - 0.05 * extentlim, '$R^2 = $ {}'.format(R2), style='italic', fontsize=28)
    ax.text(0, extentlim - (0.05 + 0.064) * extentlim, '$RMSE = $' + str(RMSE)+'$\mu g/m^3$', style='italic', fontsize=28)
    if b1 > 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = {}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=28)
    elif b1 == 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0), style='italic',
            fontsize=28)
    else:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y=-{}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=28)

    ax.text(0, extentlim - (0.05 + 0.064 * 3) * extentlim, 'N = ' + str(len(every_point_plot_obs_pm25)), style='italic',
            fontsize=28)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, ticks=[1, 10, 100,1000,10000])
    cbar.ax.set_yticklabels(['1', '10', r'$10^2$',r'$10^3$',r'$10^4$'], fontsize=28)
    #cbar.set_label('Number of points', fontsize=28)

    fig.savefig(outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.show()


def longterm_regression_plot(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    species,outfile):
    if species == 'PM25':
        tag = r'$PM_{2.5}$'
    index = np.where((~np.isnan(plot_obs_pm25) & ~np.isnan(plot_pre_pm25) ))
    
    plot_obs_pm25 = plot_obs_pm25[index]
    plot_pre_pm25 = plot_pre_pm25[index]
    H, xedges, yedges = np.histogram2d(plot_obs_pm25, plot_pre_pm25, bins=100)
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(figwidth, figheight))
    extent = [0, max(xedges), 0, max(xedges)]
    RMSE = np.sqrt(mean_squared_error(plot_obs_pm25, plot_pre_pm25))
    RMSE = round(RMSE, 1)

    R2 = linear_regression(plot_obs_pm25, plot_pre_pm25)
    R2 = np.round(R2, 2)

    ax = plt.axes([0.1,0.1,0.8,0.8])  # [left, bottom, width, height]
    cbar_ax = plt.axes([0.78,0.2,0.015,0.45])#plt.axes([0.91,0.2,0.03,0.6])
    regression_Dic = regress2(_x=plot_obs_pm25,_y=plot_pre_pm25,_method_type_1='ordinary least square',_method_type_2='reduced major axis',
    )
    b0,b1 = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    b0 = round(b0, 2)
    b1 = round(b1, 2)

    extentlim = 3*np.mean(plot_obs_pm25)
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(plot_obs_pm25, plot_pre_pm25,
                   cmap='cool', vmin=1, vmax=10, extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([0, extentlim], [0, extentlim], color='black', linestyle='--')
    ax.plot([0, extentlim], [b0, b0 + b1 * extentlim], color='pink', linestyle='-',linewidth=2)
    #ax.set_title('Comparsion of Modeled $PM_{2.5}$ and observations for '+area_name+' '+beginyear+' '+endyear)
    ax.set_xlabel('Observed {} concentration ($\mu g/m^3$)'.format(tag), fontsize=28)
    ax.set_ylabel('Estimated {} concentration ($\mu g/m^3$)'.format(tag), fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=24)

    ax.text(0, extentlim - 0.05 * extentlim, '$R^2 = $ {}'.format(R2), style='italic', fontsize=28)
    ax.text(0, extentlim - (0.05 + 0.064) * extentlim, '$RMSE = $' + str(RMSE)+'$\mu g/m^3$', style='italic', fontsize=28)
    if b1 > 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = {}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=28)
    elif b1 == 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0), style='italic',
            fontsize=28)
    else:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y=-{}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=28)

    ax.text(0, extentlim - (0.05 + 0.064 * 3) * extentlim, 'N = ' + str(len(plot_pre_pm25)), style='italic',
            fontsize=28)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0)
    cbar.set_ticks([])
    cbar.set_ticks([1, 3, 6, 10])  # Customize these based on your needs
    cbar.ax.set_yticklabels(['1', '3', '6', '10'], fontsize=28)
    #cbar.set_label('Number of points', fontsize=28)

    fig.savefig(outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.show()

def plot_R2_rRMSE_timeseries(final_data, obs_data, sites_data,dates_data,outfile):
    ## left y-axis is the R2, right y-axis is the rRMSE
    unique_sites = np.unique(sites_data)
    unique_dates = np.unique(dates_data)
    ## Convert dates from YYYYMMDD to datetime objects for plotting
    # Convert integer date in YYYYMMDD format to datetime64 for plotting
    dates = np.array([np.datetime64(f"{int(date)//10000:04d}-{(int(date)%10000)//100:02d}-{int(date)%100:02d}", 'D') for date in unique_dates])
    r2_array = np.array([np.corrcoef(final_data[dates_data == date], obs_data[dates_data == date])[0, 1]**2 for date in unique_dates])
    rrmse_array = np.array([np.sqrt(np.mean((final_data[dates_data == date] - obs_data[dates_data == date])**2)) / np.mean(obs_data[dates_data == date]) for date in unique_dates])
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(dates, r2_array, 'b-', label='R2', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('R2', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(dates, rrmse_array, 'r-', label='rRMSE', linewidth=2)
    ax2.set_ylabel('rRMSE', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_title('R2 and rRMSE Timeseries for {} at {} Sites'.format(species, len(unique_sites)))
    fig.tight_layout()
    fig.savefig(outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.show()


def plot_final_obs_comparison(final_data, obs_data, sites_data, dates_data,outfile,area='North America',):
    unique_sites = np.unique(sites_data)
    unique_dates = np.unique(dates_data)
    dates = np.array([np.datetime64(f"{int(date)//10000:04d}-{(int(date)%10000)//100:02d}-{int(date)%100:02d}", 'D') for date in unique_dates])
    daily_average_final_data = np.array([np.mean(final_data[dates_data == date]) for date in unique_dates])
    daily_average_obs_data = np.array([np.mean(obs_data[dates_data == date]) for date in unique_dates])
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, daily_average_obs_data, 'r-', label='Observed Data', linewidth=3)
    ax.plot(dates, daily_average_final_data, 'b-', label='Model Derived Data', linewidth=3)
    ax.set_xlabel('Date',fontdict={'fontsize':14})
    ax.set_ylabel(r'$PM_{2.5}$ Concentration', fontdict={'fontsize':14})
    ax.set_title(area + r' Daily Average Final vs Observed $PM_{2.5}$ Concentration Timeseries',fontsize=16)
    ax.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    fig.savefig(outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.show()


import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.patches as mpatches
def plot_BLCO_test_train_buffers(train_index, test_index, excluded_index, sitelat, sitelon, buffer_radius,extent,fig_outfile):
    ax = plt.axes(projection=ccrs.PlateCarree())
    bottom_lat = extent[0]
    left_lon     = extent[2]
    up_lat       = extent[1]
    right_lon   = extent[3]
    extent = [left_lon,right_lon,bottom_lat,up_lat]
    ax.set_extent(extent)
    
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor=cfeat.COLORS['land']))
    ax.add_feature(cfeat.BORDERS, linewidth=0.1)
    ax.add_feature(cfeat.BORDERS, linewidth=0.1)
    ax.add_feature(cfeat.LAKES, linewidth = 0.05)
    nearest_distances = np.array([],dtype=np.float32)
    for isite in range(len(test_index)):
        site_distances = calculate_distance_forArray(site_lat=sitelat[test_index[isite]],site_lon=sitelon[test_index[isite]],SATLAT_MAP=sitelat[train_index],SATLON_MAP=sitelon[train_index])
        nearest_distances = np.append(nearest_distances,np.min(site_distances[np.where(site_distances>0.1)]))
        ax.add_patch(mpatches.Circle(xy=[sitelon[test_index[isite]], sitelat[test_index[isite]]], radius=buffer_radius*0.01, color='white', alpha=0.8, transform=ccrs.PlateCarree(), zorder=6))
    average_neaerest_distance = round(np.average(nearest_distances),1)
            
    plt.scatter(sitelon[test_index], sitelat[test_index], s=10,
            linewidths=0.1, marker='*', edgecolors='red',c='red',
            alpha=0.8,label='Test Sites - {}\n Average Distance {}'.format(len(test_index),average_neaerest_distance),zorder=10)
    plt.scatter(sitelon[train_index], sitelat[train_index], s=3,  
            linewidths=0.1, marker='o', edgecolors='black',c='black',
            alpha=0.8,label='Training Sites - {}'.format(len(train_index)),zorder=8)
    plt.scatter(sitelon[excluded_index], sitelat[excluded_index], s=3,
            linewidths=0.1, marker='X',c='blue',
            alpha=0.5,label='Excluded Sites - {}'.format(len(excluded_index)),zorder=8)
    plt.legend(fontsize='small',markerscale = 3.0,loc=4)
    plt.savefig(fig_outfile, format='png', dpi=2000, transparent=True,bbox_inches='tight')
    plt.close()
    return
