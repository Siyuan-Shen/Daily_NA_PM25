import numpy as np
import math
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error





def linear_regression(x, y):
    # Convert to numpy arrays if they are not already
    x = np.array(x)
    y = np.array(y)
    
    # Filter out NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    # Check if there are enough data points after filtering
    if len(x) == 0 or len(y) == 0:
        print("No valid data points after removing NaN values.")
        return -999.0
    
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(x)):
        diffXXBar = x[i] - xBar
        diffYYBar = y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = math.sqrt(varX * varY)
    if SST == 0.0:
        rsquared = 0.0
    else:
        print("r: ", SSR / SST, "r-squared: ", (SSR / SST) ** 2)
        rsquared = (SSR / SST) ** 2
    return rsquared


def Cal_RMSE(x,y):
    # Convert to numpy arrays if they are not already
    x = np.array(x)
    y = np.array(y)
    
    # Filter out NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    # Check if there are enough data points after filtering
    if len(x) == 0 or len(y) == 0:
        print("No valid data points after removing NaN values.")
        return -999.0
    RMSE = np.sqrt(mean_squared_error(x, y))
    RMSE = round(RMSE, 2)
    print('RMSE: {}'.format(RMSE))
    return RMSE

def Cal_NRMSE(final_data,obs_data):
    # Convert to numpy arrays if they are not already
    final_data = np.array(final_data)
    obs_data = np.array(obs_data)
    
    # Filter out NaN values
    mask = ~np.isnan(final_data) & ~np.isnan(obs_data)
    final_data = final_data[mask]
    obs_data = obs_data[mask]
    
    # Check if there are enough data points after filtering
    if len(final_data) == 0 or len(obs_data) == 0:
        print("No valid data points after removing NaN values.")
        return -999.0
    RMSE = np.sqrt(mean_squared_error(final_data, obs_data))
    RMSE = round(RMSE, 2)
    NRMSE = RMSE/np.mean(obs_data)
    return NRMSE

def Cal_PWM_rRMSE(x,y,population):
    # Convert to numpy arrays if they are not already
    x = np.array(x)
    y = np.array(y)
    
    # Filter out NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    population = population[mask]
    # Check if there are enough data points after filtering
    if len(x) == 0 or len(y) == 0:
        print("No valid data points after removing NaN values.")
        return -999.0
    Total_Population = np.sum(population)
    Weighted_RMSE = np.sum(population*np.square(x-y)) 
    PWA_RMSE = np.sqrt(Weighted_RMSE/Total_Population)
    PWA_PM = np.sum(population*x)/Total_Population
    PWA_rRMSE = PWA_RMSE/PWA_PM
    PWA_rRMSE = round(PWA_rRMSE, 3)
    return PWA_rRMSE

def Calculate_PWA_PM25(Population_array:np.array, PM25_array:np.array):
    """Calculate the Population Weighted PM2.5
    Args:
        Population_Map (np.array): _description_
        PM25_Map (np.array): _description_

    Returns:
        _type_: _description_
    """
    mask = ~np.isnan(PM25_array)
    PM25_array = PM25_array[mask]
    Population_array = Population_array[mask]

    if Population_array.shape == PM25_array.shape:
        index = np.where(PM25_array>0)
        Total_Population = np.sum(Population_array[index])
        Weighted_PM25 = np.sum(Population_array[index]*PM25_array[index])
        PWA_PM25 = Weighted_PM25/Total_Population 
    else:
        print('PM25_Map Dim: ', PM25_array.shape,'Population Map Dim: ',Population_array.shape)
        print('Error Calculate_PWA_PM25 - The dimensions of Population_Map and PM25_Map are different!')
    return PWA_PM25

def linear_slope(x, y):
    # Convert to numpy arrays if they are not already
    x = np.array(x)
    y = np.array(y)
    
    # Filter out NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    # Check if there are enough data points after filtering
    if len(x) == 0 or len(y) == 0:
        print("No valid data points after removing NaN values.")
        return -999.0
    
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(x)):
        diffXXBar = x[i] - xBar
        diffYYBar = y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    print("r: ", SSR / SST, "r-squared: ", (SSR / SST) ** 2)

    return np.linalg.solve(A, b)



def regress2(_x, _y, _method_type_1 = "ordinary least square",
             _method_type_2 = "reduced major axis",
             _weight_x = [], _weight_y = [], _need_intercept = True):
    # Regression Type II based on statsmodels
    # Type II regressions are recommended if there is variability on both x and y
    # It's computing the linear regression type I for (x,y) and (y,x)
    # and then average relationship with one of the type II methods
    #
    # INPUT:
    #   _x <np.array>
    #   _y <np.array>
    #   _method_type_1 <str> method to use for regression type I:
    #     ordinary least square or OLS <default>
    #     weighted least square or WLS
    #     robust linear model or RLM
    #   _method_type_2 <str> method to use for regression type II:
    #     major axis
    #     reduced major axis <default> (also known as geometric mean)
    #     arithmetic mean
    #   _need_intercept <bool>
    #     True <default> add a constant to relation (y = a x + b)
    #     False force relation by 0 (y = a x)
    #   _weight_x <np.array> containing the weigth of x
    #   _weigth_y <np.array> containing the weigth of y
    #
    # OUTPUT:
    #   slope
    #   intercept
    #   r
    #   r_square
    #   std_slope
    #   std_intercept
    #   predict
    #
    # REQUIRE:
    #   numpy
    #   statsmodels
    #
    # The code is based on the matlab function of MBARI.
    # AUTHOR: Nils Haentjens
    # REFERENCE: https://www.mbari.org/products/research-software/matlab-scripts-linear-regressions/
    # Convert to numpy arrays if they are not already
    _x = np.array(_x)
    _y = np.array(_y)
    
    # Filter out NaN values
    mask = ~np.isnan(_x) & ~np.isnan(_y)
    _x = _x[mask]
    _y = _y[mask]
    
    # Check if there are enough data points after filtering
    if len(_x) <= 1 or len(_y) <= 1:
        print("No valid data points after removing NaN values.")
        return {"slope": float(-999.0), "intercept": -999.0, "r": -999.0, 'r_square': -999.0,
            "std_slope": -999.0, "std_intercept": -999.0,
            "predict": -999.0}
    # Check input
    if _method_type_2 != "reduced major axis" and _method_type_1 != "ordinary least square":
        raise ValueError("'" + _method_type_2 + "' only supports '" + _method_type_1 + "' method as type 1.")

    # Set x, y depending on intercept requirement
    if _need_intercept:
        x_intercept = sm.add_constant(_x)
        y_intercept = sm.add_constant(_y)

    # Compute Regression Type I (if type II requires it)
    if (_method_type_2 == "reduced major axis" or
        _method_type_2 == "geometric mean"):
        if _method_type_1 == "OLS" or _method_type_1 == "ordinary least square":
            if _need_intercept:
                
                [intercept_a, slope_a] = sm.OLS(_y, x_intercept).fit().params
                [intercept_b, slope_b] = sm.OLS(_x, y_intercept).fit().params
            else:
                slope_a = sm.OLS(_y, _x).fit().params
                slope_b = sm.OLS(_x, _y).fit().params
        elif _method_type_1 == "WLS" or _method_type_1 == "weighted least square":
            if _need_intercept:
                [intercept_a, slope_a] = sm.WLS(
                    _y, x_intercept, weights=1. / _weight_y).fit().params
                [intercept_b, slope_b] = sm.WLS(
                    _x, y_intercept, weights=1. / _weight_x).fit().params
            else:
                slope_a = sm.WLS(_y, _x, weights=1. / _weight_y).fit().params
                slope_b = sm.WLS(_x, _y, weights=1. / _weight_x).fit().params
        elif _method_type_1 == "RLM" or _method_type_1 == "robust linear model":
            if _need_intercept:
                [intercept_a, slope_a] = sm.RLM(_y, x_intercept).fit().params
                [intercept_b, slope_b] = sm.RLM(_x, y_intercept).fit().params
            else:
                slope_a = sm.RLM(_y, _x).fit().params
                slope_b = sm.RLM(_x, _y).fit().params
        else:
            raise ValueError("Invalid literal for _method_type_1: " + _method_type_1)

    # Compute Regression Type II
    if (_method_type_2 == "reduced major axis" or
        _method_type_2 == "geometric mean"):
        # Transpose coefficients
        if _need_intercept:
            intercept_b = -intercept_b / slope_b
        slope_b = 1 / slope_b
        # Check if correlated in same direction
        if np.sign(slope_a) != np.sign(slope_b):
            raise RuntimeError('Type I regressions of opposite sign.')
        # Compute Reduced Major Axis Slope
        slope = np.sign(slope_a) * np.sqrt(slope_a * slope_b)
        if _need_intercept:
            # Compute Intercept (use mean for least square)
            if _method_type_1 == "OLS" or _method_type_1 == "ordinary least square":
                intercept = np.mean(_y) - slope * np.mean(_x)
            else:
                intercept = np.median(_y) - slope * np.median(_x)
        else:
            intercept = 0
        # Compute r
        r = np.sign(slope_a) * np.sqrt(slope_a / slope_b)
        # Compute predicted values
        predict = slope * _x + intercept
        # Compute standard deviation of the slope and the intercept
        n = len(_x)
        diff = _y - predict
        Sx2 = np.sum(np.multiply(_x, _x))
        den = n * Sx2 - np.sum(_x) ** 2
        s2 = np.sum(np.multiply(diff, diff)) / (n - 2)
        std_slope = np.sqrt(n * s2 / den)
        if _need_intercept:
            std_intercept = np.sqrt(Sx2 * s2 / den)
        else:
            std_intercept = 0
    elif (_method_type_2 == "Pearson's major axis" or
          _method_type_2 == "major axis"):
        if not _need_intercept:
            raise ValueError("Invalid value for _need_intercept: " + str(_need_intercept))
        xm = np.mean(_x)
        ym = np.mean(_y)
        xp = _x - xm
        yp = _y - ym
        sumx2 = np.sum(np.multiply(xp, xp))
        sumy2 = np.sum(np.multiply(yp, yp))
        sumxy = np.sum(np.multiply(xp, yp))
        slope = ((sumy2 - sumx2 + np.sqrt((sumy2 - sumx2)**2 + 4 * sumxy**2)) /
                 (2 * sumxy))
        intercept = ym - slope * xm
        # Compute r
        r = sumxy / np.sqrt(sumx2 * sumy2)
        # Compute standard deviation of the slope and the intercept
        n = len(_x)
        std_slope = (slope / r) * np.sqrt((1 - r ** 2) / n)
        sigx = np.sqrt(sumx2 / (n - 1))
        sigy = np.sqrt(sumy2 / (n - 1))
        std_i1 = (sigy - sigx * slope) ** 2
        std_i2 = (2 * sigx * sigy) + ((xm ** 2 * slope * (1 + r)) / r ** 2)
        std_intercept = np.sqrt((std_i1 + ((1 - r) * slope * std_i2)) / n)
        # Compute predicted values
        predict = slope * _x + intercept
    elif _method_type_2 == "arithmetic mean":
        if not _need_intercept:
            raise ValueError("Invalid value for _need_intercept: " + str(_need_intercept))
        n = len(_x)
        sg = np.floor(n / 2)
        # Sort x and y in order of x
        sorted_index = sorted(range(len(_x)), key=lambda i: _x[i])
        x_w = np.array([_x[i] for i in sorted_index])
        y_w = np.array([_y[i] for i in sorted_index])
        x1 = x_w[1:sg + 1]
        x2 = x_w[sg:n]
        y1 = y_w[1:sg + 1]
        y2 = y_w[sg:n]
        x1m = np.mean(x1)
        x2m = np.mean(x2)
        y1m = np.mean(y1)
        y2m = np.mean(y2)
        xm = (x1m + x2m) / 2
        ym = (y1m + y2m) / 2
        slope = (x2m - x1m) / (y2m - y1m)
        intercept = ym - xm * slope
        # r (to verify)
        r = []
        # Compute predicted values
        predict = slope * _x + intercept
        # Compute standard deviation of the slope and the intercept
        std_slope = []
        std_intercept = []
    r_square = r*r
    # Return all that
    return {"slope": float(slope), "intercept": intercept, "r": r, 'r_square': r_square,
            "std_slope": std_slope, "std_intercept": std_intercept,
            "predict": predict}