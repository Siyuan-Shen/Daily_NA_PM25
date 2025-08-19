from Training_pkg.data_func import TransformerInputDatasets
from Training_pkg.TrainingModule import Transformer_train
from Training_pkg.utils import channel_names
import numpy as np
from Evaluation_pkg.data_func import randomly_select_training_testing_indices,Split_Datasets_based_site_index
from Model_Structure_pkg.Transformer_Model.model.transformer import Transformer
from Model_Structure_pkg.utils import Transformer_max_len, Transformer_spin_up_len
import numpy as np
import torch
import torch.nn as nn


print('Loading Transformer datasets...')
Init_Transformer_Datasets = TransformerInputDatasets(species='PM25', total_channel_names=channel_names, bias=False, normalize_bias=False, normalize_species=True, absolute_species=False, datapoints_threshold=100)
total_sites_number = Init_Transformer_Datasets.total_sites_number
true_input_mean, true_input_std = Init_Transformer_Datasets.true_input_mean, Init_Transformer_Datasets.true_input_std
TrainingDatasets_mean, TrainingDatasets_std = Init_Transformer_Datasets.TrainingDatasets_mean, Init_Transformer_Datasets.TrainingDatasets_std
sites_lat, sites_lon = Init_Transformer_Datasets.sites_lat, Init_Transformer_Datasets.sites_lon

true_input = Init_Transformer_Datasets.true_input
true_array = np.concatenate([true_input[str(site)]['PM25'] for site in true_input.keys()])
print('Max true input:', np.max(true_array),'Min true input:', np.min(true_array),
              'true input mean:', Init_Transformer_Datasets.true_input_mean, 'true input std:', Init_Transformer_Datasets.true_input_std)
# Get the initial true_input and training datasets for the current model (within the desired time range)
print('1...')
desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data = Init_Transformer_Datasets.get_desired_range_inputdatasets(start_date=20220101,
                                                                                end_date=20231231,max_len=Transformer_max_len,spinup_len=Transformer_spin_up_len) # initial datasets
# Normalize the training datasets
print('2...')
normalized_TrainingDatasets  = Init_Transformer_Datasets.normalize_trainingdatasets(desired_trainingdatasets=desired_trainingdatasets)

# Concatenate the training datasets and true input for the current model for training and tetsing purposes
print('3...')
cctnd_trainingdatasets, cctnd_true_input,cctnd_ground_observation_data,cctnd_geophysical_species_data, cctnd_sites_index, cctnd_dates = Init_Transformer_Datasets.concatenate_trainingdatasets(desired_true_input=desired_true_input, 
                                                                                                                        desired_normalized_trainingdatasets=normalized_TrainingDatasets,
                                                                                                                        desired_ground_observation_data=desired_ground_observation_data,
                                                                                                                        desired_geophysical_species_data=desired_geophysical_species_data)

print('4...')
training_selected_sites, testing_selected_sites = randomly_select_training_testing_indices(sites_index=np.arange(total_sites_number), training_portion=0.8)
print('training_selected_sites: ',training_selected_sites.shape)
print('testing_selected_sites: ',testing_selected_sites.shape) 

X_train, y_train, X_test, y_test, GGdates_train, dates_test, sites_train, sites_test, train_datasets_index, test_datasets_index = Split_Datasets_based_site_index(train_site_index=training_selected_sites,
                                                                                                                test_site_index=testing_selected_sites,
                                                                                                                total_trainingdatasets=cctnd_trainingdatasets,
                                                                                                                total_true_input=cctnd_true_input,
                                                                                                                total_sites_index=cctnd_sites_index,
                                                                                                                total_dates=cctnd_dates)
print('cctnd_ground_observation_data[test_datasets_index]: ', cctnd_ground_observation_data[test_datasets_index])
print('cctnd_geophysical_species_data[test_datasets_index]: ', cctnd_geophysical_species_data[test_datasets_index])
print('test_datasets_index: ', test_datasets_index)


from multiprocessing import Manager
manager = Manager()
run_id_container = manager.dict() 
Transformer_train(rank=0,world_size=1,temp_sweep_config=None,sweep_mode=False,
                  sweep_id=None, run_id_container=run_id_container,init_total_channel_names=channel_names,
                  X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                  input_mean=TrainingDatasets_mean, input_std=TrainingDatasets_std,
                  evaluation_type='Transformer',
                  typeName='Normalized_obs',begindates=20220101, enddates=20221231,ifold=0)