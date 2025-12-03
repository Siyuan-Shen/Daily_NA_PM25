from click import style
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as crt
import numpy as np
#from .Statistic_Func import Calculate_PWA_PM25, linear_regression,linear_slope
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import xarray as xr
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.ticker as tick
import matplotlib.colors as colors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Training_pkg.Statistic_func import Calculate_PWA_PM25
from Visualization_pkg.utils import crop_map_data



def Plot_Species_Map_Figures(PM25_Map:np.array,PM25_LAT:np.array,PM25_LON:np.array,PM25_Sites_LON:np.array, PM25_Sites_LAT:np.array, PM25_Sites:np.array,Population_Map:np.array,
                             population_Lat:np.array, population_Lon:np.array, extent:np.array, outfile:str,YYYY,MM,DD):
    
    PM25_Map[np.where(PM25_Map < 0)] = 0
    PM25_Map = np.nan_to_num(PM25_Map, nan=5.0, posinf=3.0, neginf=2.0)
    Cropped_Population_Map = crop_map_data(Population_Map,population_Lat,population_Lon,extent)
    Croppeed_PM25_Map      = crop_map_data(PM25_Map, PM25_LAT, PM25_LON,Extent=extent)
    PWA_PM25 = Calculate_PWA_PM25(Population_array=Cropped_Population_Map, PM25_array=Croppeed_PM25_Map)
    ax = plt.axes(projection=ccrs.PlateCarree())
    m1 = 0
    m2 = PWA_PM25*2.5
    extent = [extent[2],extent[3],extent[0],extent[1]]
    
    #print('PM25_Lat Dim:',PM25_LAT.shape, 'PM25_Lon Dim:', PM25_LON.shape, 'PM25_Map Dim:', PM25_Map.shape,'PWA: ',PWA_PM25)
    
    print('extent:', extent)
    ax.set_aspect(1.25)
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    #ax.add_feature(cfeat.COASTLINE,linewidth = 0.15) 
    ax.add_feature(cfeat.LAKES, linewidth   = 0.05)
    ax.add_feature(cfeat.BORDERS, linewidth = 0.1)
    pcm = plt.pcolormesh(PM25_LON, PM25_LAT,PM25_Map,transform=ccrs.PlateCarree(),
          cmap = 'YlOrRd',norm=colors.Normalize(vmin = m1, vmax = m2))
    #ax.add_feature(cfeat.OCEAN) 

    #RMSE = round(np.sqrt(mean_squared_error(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],
    #                                      pre_pm25_site[area_index])),2)
    #R2 = round(linear_regression(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],pre_pm25_site[area_index]),2)    
    #ax.text(extent[2], extent[1]-0.1*abs(extent[1]), '$R^2 = $' + str(R2), style='italic', fontsize=12)
    #ax.text(extent[2], extent[1], '$RMSE = $' + str(RMSE), style='italic', fontsize=12)
    
    ax.text(extent[0]+0.01*abs(extent[1]-extent[0]),extent[2]+0.05*abs(extent[3]-extent[2]),'PWM ' + r'$\rm{PM_{2.5} = }$' + str(round(PWA_PM25,1)) +r' $\rm{(\mu g/m^3)}$', style='italic',fontsize = 6)
    ax.text(extent[0]+0.01*abs(extent[1]-extent[0]),extent[2]+0.10*abs(extent[3]-extent[2]),'{} {} {}'.format(YYYY,MM,DD), style='italic',fontsize = 6)

    plt.scatter(PM25_Sites_LON, PM25_Sites_LAT, c=PM25_Sites, s=0.4,
                    linewidths=0.1, marker='o', edgecolors='black', vmin=0, vmax=m2,
                    cmap='YlOrRd',
                   alpha=0.8)
    
    ## Global colorbar parameters fraction=0.35, pad=-1.63, shrink=0.5, aspect=50.0
    cbar = plt.colorbar(pcm, location = 'right',fraction=0.15, shrink=0.5,aspect=40.0, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('PM$_{2.5}$' + '' + r'$\rm{(\mu g/m^3)}$')
    cbar.ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='k', alpha=0.7, linestyle='--')
    ##gl.top_labels = False  ##关闭上侧坐标显示
    #gl.right_labels = False  ##关闭右侧坐标显示
    #gl.xformatter = LONGITUDE_FORMATTER  ##坐标刻度转换为经纬度样式
    #gl.yformatter = LATITUDE_FORMATTER
    #gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1], 10))
    #gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3], 10))
    #gl.xlabel_style = {'size': 3.5}
    #gl.ylabel_style = {'size': 3.5}
    plt.savefig(outfile, format = 'png', dpi= 500, transparent = True,bbox_inches='tight')
    plt.close()

    return
