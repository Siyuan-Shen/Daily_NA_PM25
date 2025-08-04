import numpy as np
import os
from Evaluation_pkg.utils import get_nearest_point_index

def find_masked_latlon(mask_map,mask_lat,mask_lon,test_lat,test_lon):
    index_lon,index_lat = get_nearest_point_index(test_lon,test_lat,mask_lon,mask_lat)
    masked_obs_array = mask_map[index_lat,index_lon]
    masked_array_index = np.where(masked_obs_array == 1)
    return masked_array_index[0]

def Get_final_output(Validation_Prediction, validation_geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std  ):
    """This function is used to convert the model estimation to absolute PM species concentration and to compare with the 
    observed PM species.

    Args:
        Validation_Prediction (_type_): _description_
        geophysical_species (_type_): _description_
        SPECIES_OBS (_type_): _description_
        bias (_type_): _description_
        normalize_species (_type_): _description_
        absolute_species (_type_): _description_
        log_species (_type_): _description_
        Y_Testing_index (_type_): _description_

    Returns:
        _type_: _description_
    """
    if bias == True:
        final_data = Validation_Prediction + validation_geophysical_species
    elif normalize_bias == True:
        final_data = Validation_Prediction * std + mean + validation_geophysical_species
    elif normalize_species == True:
        final_data = Validation_Prediction * std + mean
    elif absolute_species == True:
        final_data = Validation_Prediction
    elif log_species == True:
        final_data = np.exp(Validation_Prediction) - 1
    return final_data


def Split_Datasets_based_site_index(train_site_index,test_site_index,total_trainingdatasets, total_true_input, total_sites_index, total_dates):
    print('total_sites_index: ',total_sites_index)
    
    print('train_site_index: ',train_site_index)
    
    print('test_site_index: ',test_site_index)
    print('total_sites_index.shape: ',total_sites_index.shape)
    print('train_site_index.shape: ',train_site_index.shape)
    print('test_site_index.shape: ',test_site_index.shape)
    train_datasets_index = np.where(np.isin(total_sites_index, train_site_index))[0]
    test_datasets_index = np.where(np.isin(total_sites_index, test_site_index))[0]
    X_train = total_trainingdatasets[train_datasets_index, :]
    y_train = total_true_input[train_datasets_index]
    X_test = total_trainingdatasets[test_datasets_index, :]
    y_test = total_true_input[test_datasets_index]
    dates_train = total_dates[train_datasets_index]
    dates_test = total_dates[test_datasets_index]
    sites_train = total_sites_index[train_datasets_index]
    sites_test = total_sites_index[test_datasets_index]
    return X_train, y_train, X_test, y_test, dates_train, dates_test, sites_train, sites_test,train_datasets_index, test_datasets_index

def randomly_select_training_testing_indices(sites_index, training_portion):
    if training_portion > 0.0 and training_portion < 1.0:
        num_elements_to_select = int(training_portion * len(sites_index))
    else:
        raise ValueError("training_portion must be between 0.0 and 1.0")
    np.random.seed(19980130)
    training_selected_indices = np.random.choice(len(sites_index), num_elements_to_select, replace=False)
    training_selected_sites = sites_index[training_selected_indices]
    
    testing_selected_indices = np.setdiff1d(np.arange(len(sites_index)), training_selected_indices)
    testing_selected_sites = sites_index[testing_selected_indices]
    return training_selected_sites, testing_selected_sites

