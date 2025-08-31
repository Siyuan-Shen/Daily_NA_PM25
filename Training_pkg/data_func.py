import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from Training_pkg.utils import *
from Evaluation_pkg.utils import create_date_range
from Model_Structure_pkg.utils import Transformer_trg_dim
import time
import copy
from datetime import datetime, timedelta


class TransformerInputDatasets():
    def __init__(self, species, total_channel_names, bias, normalize_bias, normalize_species, absolute_species,datapoints_threshold):
        self.datapoints_threshold = datapoints_threshold

        self.species = species
        self.bias = bias
        self.normalize_bias = normalize_bias
        self.normalize_species = normalize_species
        self.absolute_species = absolute_species

        self.ground_observation_data = self._load_daily_PM25_data()
        self.geophysical_species_data,self.bias_data = self._load_geophyscial_PMandBias_data()
        self.ground_observation_data, self.geophysical_species_data, self.bias_data,self.delete_all_data_sites, self.delete_specific_data_sites, self.delete_specific_data_indices = self._get_nonan_and_threshold_sites()
        self.true_input, self.true_input_mean, self.true_input_std = self._Learning_objective()
        self.total_sites_number = len(self.true_input.keys())
        self.sites_lat, self.sites_lon = self._get_lat_lon_arrays()

        start_time = time.time()
        print('Start loading and aggregating the training datasets...')
        self.trainingdatasets = self._load_aggregate_TrainingDatasets(total_channel_names, self.total_sites_number)
        print('The time of loading and aggregating the training datasets is {} seconds'.format(time.time()-start_time))
        
        self.ground_observation_data, self.geophysical_species_data, self.bias_data, self.trainingdatasets, self.true_input = self._rename_key_in_dict()

        print('Start deriving the training datasets normalization matrix...')
        start_time = time.time()
        self.TrainingDatasets_mean, self.TrainingDatasets_std = self._derive_trainingdatasets_normalization_matrix()
        print('The time of deriving the training datasets normalization matrix is {} seconds'.format(time.time()-start_time))


    def _get_nonan_and_threshold_sites(self):
        temp_observation_data = copy.deepcopy(self.ground_observation_data)
        temp_geophysical_species_data = copy.deepcopy(self.geophysical_species_data)
        temp_bias_data = copy.deepcopy(self.bias_data)

        delete_all_data_sites = []
        delete_specific_data_sites = []
        delete_specific_data_indices = []

        for isite in self.ground_observation_data.keys():
            site = str(isite)
            
            if np.isnan(self.geophysical_species_data[site]['geoPM25']).any():
                #print('Warning: Site {} has NaN values in PM2.5 data!'.format(site))
                if np.isnan(self.geophysical_species_data[site]['geoPM25']).all():
                    del temp_observation_data[site]
                    del temp_geophysical_species_data[site]
                    del temp_bias_data[site]
                    delete_all_data_sites.append(site)
                '''
                else:
                    indice = np.where(np.isnan(self.geophysical_species_data[site]['geoPM25']))[0]
                    delete_specific_data_sites.append(site)
                    delete_specific_data_indices.append(indice)
                    temp_observation_data[site]['PM25'] = np.delete(temp_observation_data[site]['PM25'], indice, axis=0)
                    temp_observation_data[site]['dates'] = np.delete(temp_observation_data[site]['dates'], indice, axis=0)
                    temp_geophysical_species_data[site]['geoPM25'] = np.delete(temp_geophysical_species_data[site]['geoPM25'], indice, axis=0)
                    temp_geophysical_species_data[site]['dates'] = np.delete(temp_geophysical_species_data[site]['dates'], indice, axis=0)
                    temp_bias_data[site]['geobias'] = np.delete(temp_bias_data[site]['geobias'], indice, axis=0)
                    temp_bias_data[site]['dates'] = np.delete(temp_bias_data[site]['dates'], indice, axis=0)
                '''
            elif len(self.ground_observation_data[site]['PM25']) < self.datapoints_threshold:
                    #print('Warning: Site {} has less than {} data points!'.format(site, self.datapoints_threshold))
                    del temp_observation_data[site]
                    del temp_geophysical_species_data[site]
                    del temp_bias_data[site]
                    delete_all_data_sites.append(site)
            else:
                None
                #print('Site {} has no NaN values in PM2.5 data.'.format(site))
        return temp_observation_data, temp_geophysical_species_data, temp_bias_data, delete_all_data_sites, delete_specific_data_sites, delete_specific_data_indices


    def _load_daily_PM25_data(self):
        '''
        Load daily PM2.5 data from the directory of obs_indir. 
        return: data - it is a dictionary with keys as the site loc, date, and values as the daily PM2.5 data.
        '''
        infile = ground_observation_data_dir + ground_observation_data_infile
        if not os.path.exists(infile):
            raise ValueError('Observation file - The {} file does not exist!'.format(infile))
        data = np.load(infile,allow_pickle=True).item()

        return data
    
    def _load_geophyscial_PMandBias_data(self):
        PM_infile = geophysical_species_data_dir + geophysical_species_data_infile
        Bias_infile = geophysical_biases_data_dir + geophysical_biases_data_infile
        if not os.path.exists(PM_infile):
            raise ValueError('Geophysical species file - The {} file does not exist!'.format(PM_infile))
        if not os.path.exists(Bias_infile):
            raise ValueError('Geophysical biases file - The {} file does not exist!'.format(Bias_infile))
        PM_data = np.load(PM_infile,allow_pickle=True).item()
        Bias_data = np.load(Bias_infile,allow_pickle=True).item()
        return PM_data, Bias_data

    def _Learning_objective(self):
        if self.bias == True:
            PM_data, Bias_data = copy.deepcopy(self.geophysical_species_data),copy.deepcopy(self.bias_data)

            true_input = Bias_data
            mean = 0
            std = 1
            return true_input, mean, std
        
        elif self.normalize_bias:
            PM_data, Bias_data = copy.deepcopy(self.geophysical_species_data),copy.deepcopy(self.bias_data)
            for index, isite in enumerate(Bias_data.keys()):
                if index == 0:
                    total_bias_data = Bias_data[str(isite)]['geobias']
                else:
                    total_bias_data = np.concatenate((total_bias_data, Bias_data[str(isite)]['geobias']), axis=0)
            if normalize_type == 'Gaussian':
                bias_mean = np.mean(total_bias_data)
                bias_std = np.std(total_bias_data)
            elif normalize_type == 'MinMax':
                bias_mean = np.min(total_bias_data)
                bias_std = np.max(total_bias_data) - np.min(total_bias_data)
            elif normalize_type == 'Robust':
                bias_mean = np.median(total_bias_data)
                bias_std = np.percentile(total_bias_data, 75) - np.percentile(total_bias_data, 25)

            for index, isite in enumerate(Bias_data.keys()):
                Bias_data[str(isite)]['geobias'] = (Bias_data[str(isite)]['geobias'] - bias_mean) / bias_std
            
            true_input = Bias_data
            return true_input, bias_mean, bias_std
        
        elif self.absolute_species:
            PM_data = copy.deepcopy(self.ground_observation_data)

            true_input = PM_data
            mean = 0
            std = 1
            return true_input, mean, std
        
        elif self.normalize_species:
            PM_data = copy.deepcopy(self.ground_observation_data)

            for index,isite in enumerate(PM_data.keys()):
                if index == 0:
                    total_PM_data = PM_data[str(isite)]['PM25']
                else:
                    total_PM_data = np.concatenate((total_PM_data, PM_data[str(isite)]['PM25']), axis=0)
            if normalize_type == 'Gaussian':
                PM_mean = np.mean(total_PM_data)
                PM_std = np.std(total_PM_data)
            elif normalize_type == 'MinMax':
                PM_mean = np.min(total_PM_data)
                PM_std = np.max(total_PM_data) - np.min(total_PM_data)
            elif normalize_type == 'Robust':
                PM_mean = np.median(total_PM_data)
                PM_std = np.percentile(total_PM_data, 75) - np.percentile(total_PM_data, 25)
            for index,isite in enumerate(PM_data.keys()):
                PM_data[str(isite)]['PM25'] = (PM_data[str(isite)]['PM25'] - PM_mean) / PM_std
            true_input = PM_data
            return true_input, PM_mean, PM_std
    
    def _get_lat_lon_arrays(self):
        lat_array = np.zeros(self.total_sites_number,dtype=np.float32)
        lon_array = np.zeros(self.total_sites_number,dtype=np.float32)

        for i,isite in enumerate(self.ground_observation_data.keys()):
            site = str(isite)
            lat_array[i] = self.ground_observation_data[site]['lat']
            lon_array[i] = self.ground_observation_data[site]['lon']
        return lat_array, lon_array
    
    def _process_concatenate_site_data(self, data, temp_data, isite):
        site = str(isite)
        data[site]['data'] = np.concatenate((data[site]['data'], temp_data[site]['data']), axis=1)
        return data
    
    def _load_aggregate_TrainingDatasets(self, Training_Channels, sites_number):
        '''
        Here I compared the differences of time consuming between the old and new 
        version of loading and aggregating the training datasets.

        >>> data = load_aggregate_TrainingDatasets(channel_names,1684)
        The time of loading and aggregating the training datasets is 11.369443893432617 seconds
        >>> data = old_load_aggregate_TrainingDatasets(channel_names,1684)
        The time of old loading and aggregating the training datasets is 26.89457082748413 seconds 

        def old_load_aggregate_TrainingDatasets(Training_Channels, sites_number):
            start_time = time.time()
            
            for i, channel_name in enumerate(Training_Channels):
                infile = CNN_Training_infiles.format(channel_name)
                if os.path.exists(infile):
                    temp_data = np.load(infile, allow_pickle=True).item()
                else:
                    raise ValueError('The {} file does not exist!'.format(infile))
                
                if i == 0:
                    data = temp_data
                else:
                    for isite in range(sites_number):
                        site = str(isite)
                        data[site]['data'] = np.concatenate((data[site]['data'], temp_data[site]['data']), axis=1)
            end_time = time.time()
            print('The time of old loading and aggregating the training datasets is {} seconds'.format(end_time-start_time))
            return data'
        '''

        
        for i, channel_name in enumerate(Training_Channels):
            infile = Transformer_Training_infiles.format(channel_name)
            if os.path.exists(infile):
                temp_data = np.load(infile, allow_pickle=True).item()
            else:
                raise ValueError('Training file - The {} file does not exist!'.format(infile))
            
            if i == 0:
                data = temp_data
                print('Init Training Data Shape: ' ,data['0']['data'].shape)
                for isite in self.ground_observation_data.keys():
                    data[str(isite)]['dates'] = np.array(data[str(isite)]['dates'], dtype=np.int32)
            else:
                with ThreadPoolExecutor() as executor:
                    print('channel_name: ', channel_name)
                    futures = [executor.submit(self._process_concatenate_site_data, data, temp_data, isite) for isite in self.ground_observation_data.keys()]
                    for future in futures:
                        future.result()
        print('Training Data Shape: ' ,data['0']['data'].shape)
        for isite in self.delete_all_data_sites:
            if str(isite) in data.keys():
                del data[str(isite)]
                print('Site {} has been deleted from the training datasets.'.format(isite))
            else:
                print('Site {} is not in the training datasets.'.format(isite))
        '''
        for i, isite,in enumerate(self.delete_specific_data_sites):
            if str(isite) in data.keys():
                indices = self.delete_specific_data_indices[i]
                data[str(isite)]['data'] = np.delete(data[str(isite)]['data'], indices, axis=0)
                data[str(isite)]['dates'] = np.delete(data[str(isite)]['dates'], indices, axis=0)
                print('Site {} has been deleted from the training datasets at indices {}.'.format(isite, indices))
            else:
                print('Site {} is not in the training datasets.'.format(isite))
        '''
        return data
    
    def _derive_trainingdatasets_normalization_matrix(self):
       
    
        # Collect all site data in a list
        datasets = [self.trainingdatasets[str(isite)]['data'] for isite in self.ground_observation_data.keys()]
        
        # Concatenate all at once (much faster than incremental concatenation)
        total_trainingdatasets = np.concatenate(datasets, axis=0)
        # Compute mean and std along the feature axis
        if training_data_normalization_type == 'Gaussian':
            TrainingDatasets_mean = np.mean(total_trainingdatasets, axis=0)
            TrainingDatasets_std = np.std(total_trainingdatasets, axis=0)
        elif training_data_normalization_type == 'MinMax':
            TrainingDatasets_mean = np.min(total_trainingdatasets, axis=0)
            TrainingDatasets_std = np.max(total_trainingdatasets, axis=0) - np.min(total_trainingdatasets, axis=0)
        elif training_data_normalization_type == 'Robust':
            TrainingDatasets_mean = np.median(total_trainingdatasets, axis=0)
            TrainingDatasets_std = np.percentile(total_trainingdatasets, 75, axis=0) - np.percentile(total_trainingdatasets, 25, axis=0)

        return TrainingDatasets_mean, TrainingDatasets_std
    
    def _rename_key_in_dict(self):
        temp_observation_data = {}
        temp_geophysical_species_data = {}
        temp_bias_data = {}
        temp_trainingdatasets = {}
        temp_true_input = {}
        for isite, site in enumerate(self.ground_observation_data.keys()):
            temp_observation_data[str(isite)] = self.ground_observation_data[site]
            temp_geophysical_species_data[str(isite)] = self.geophysical_species_data[site]
            temp_bias_data[str(isite)] = self.bias_data[site]
            temp_trainingdatasets[str(isite)] = self.trainingdatasets[site]
            temp_true_input[str(isite)] = self.true_input[site]
        return temp_observation_data, temp_geophysical_species_data, temp_bias_data, temp_trainingdatasets, temp_true_input
    
    def get_desired_range_inputdatasets(self,start_date,end_date,max_len=365,spinup_len=30):
        '''
        For getting the desired range of input datasets based on the start_date and end_date in transformer,
        I also consider max_len and spinup_len.

        '''

        ## Get all dates range from start_date to end_date in YYYYMMDD format
        Alldates_str, AllDates_range = create_date_range(start_date, end_date)
        len_AllDates_range = len(AllDates_range)
        number_of_batch_each_site = int(np.ceil(len_AllDates_range / max_len))
        if number_of_batch_each_site == 0:
            raise ValueError('The start_date and end_date are too close, please check the input dates!')
        
        ## This is used to get the desired range of input datasets
        desired_true_input = copy.deepcopy(self.true_input)
        desired_trainingdatasets = copy.deepcopy(self.trainingdatasets)
        desired_ground_observation_data = copy.deepcopy(self.ground_observation_data)
        desired_geophysical_species_data = copy.deepcopy(self.geophysical_species_data)

        ## Create a total_dates_array to store the dates for each batch
        ## The shape of total_dates_array is (number_of_batch_each_site, max_len+spinup_len)
        ## The first spinup_len days are used for the spin-up period,
        ## and the rest max_len days are used for the actual training period.
        ## The dates in total_dates_array are in YYYYMMDD format.
        total_dates_array = np.zeros((number_of_batch_each_site, max_len+spinup_len), dtype=np.int32)
        str_start_date = datetime.strptime(str(start_date), '%Y%m%d')
        str_end_date = datetime.strptime(str(end_date), '%Y%m%d')


        ## Here we need to get the desired range of input datasets based on the start_date and end_date
        for isite in self.ground_observation_data.keys():
            temp_desired_observation_data = np.full((number_of_batch_each_site, max_len+spinup_len, Transformer_trg_dim), np.nan, dtype=np.float32)
            temp_desired_true_input = np.full((number_of_batch_each_site, max_len+spinup_len, Transformer_trg_dim), np.nan, dtype=np.float32)
            temp_desired_geophysical_species_data = np.full((number_of_batch_each_site, max_len+spinup_len, Transformer_trg_dim), np.nan, dtype=np.float32)
            temp_desired_trainingdatasets = np.full((number_of_batch_each_site, max_len+ spinup_len, self.trainingdatasets[str(isite)]['data'].shape[1]), 0.0, dtype=np.float32)

            
            for i in range(number_of_batch_each_site):
                site = str(isite)
                ## temp_start_date is the start date + i * max_len - spinup_len
                temp_start_date = str_start_date + timedelta(days=i * max_len)
                temp_end_date   = temp_start_date + timedelta(days=max_len - 1)
                temp_start_date_with_spinup = temp_start_date - timedelta(days=spinup_len)

                temp_start_date = int(temp_start_date.strftime('%Y%m%d'))
                temp_start_date_with_spinup = int(temp_start_date_with_spinup.strftime('%Y%m%d'))
                temp_end_date = int(temp_end_date.strftime('%Y%m%d'))

                #print('Processing site {}: Batch {} with start date {} and end date {}, and spin-up period start date {}'.format(site, i, temp_start_date, temp_end_date, temp_start_date_with_spinup))


                ## create a date range for the current batch with spin-up period
                temp_dates_str, temp_dates_range = create_date_range(temp_start_date_with_spinup, temp_end_date)
                total_dates_array[i, :len(temp_dates_range)] = temp_dates_range

                ## Find the indices of the dates in the desired_true_input and desired_trainingdatasets for the current batch
                #print('desired_true_input[site][\'dates\']: ', desired_true_input[site]['dates'])
                
                if len(np.where(desired_true_input[site]['dates'] >= temp_start_date)[0]) == 0 or len(np.where(desired_true_input[site]['dates'] <= temp_end_date)[0]) == 0:
                    #print('Warning: Site {} has no data in the desired range of input datasets!'.format(site))
                    continue
                else:
                    start_index = np.where(desired_true_input[site]['dates'] >= temp_start_date)[0][0]
                    end_index = np.where(desired_true_input[site]['dates'] <= temp_end_date)[0][-1]
                
                    trainingdatasets_start_index = np.where(desired_trainingdatasets[site]['dates'] >= temp_start_date_with_spinup)[0][0]
                    trainingdatasets_end_index = np.where(desired_trainingdatasets[site]['dates'] <= temp_end_date)[0][-1]

                    ## find the indices of the dates in total_dates_array for the current batch
                    batch_dates = total_dates_array[i, :]
                    batch_indices = np.where(np.isin(batch_dates,desired_true_input[site]['dates'][start_index:end_index+1]))[0]
                    training_batch_indices = np.where(np.isin(batch_dates,desired_trainingdatasets[site]['dates'][trainingdatasets_start_index:trainingdatasets_end_index+1]))[0]

                    ## Get the desired range of input datasets for the current batch
                    
                    temp_desired_observation_data[i,batch_indices] = np.expand_dims(self.ground_observation_data[site]['PM25'][start_index:end_index+1], axis=-1)
                    if self.bias == True or self.normalize_bias == True:
                        temp_desired_true_input[i,batch_indices] = np.expand_dims(self.true_input[site]['geobias'][start_index:end_index+1],axis=-1)
                    else:
                        temp_desired_true_input[i,batch_indices] = np.expand_dims(self.true_input[site]['PM25'][start_index:end_index+1], axis=-1)
                    temp_desired_geophysical_species_data[i,batch_indices] = np.expand_dims(self.geophysical_species_data[site]['geoPM25'][start_index:end_index+1], axis=-1)
                    temp_desired_trainingdatasets[i,training_batch_indices,:] = self.trainingdatasets[site]['data'][trainingdatasets_start_index:trainingdatasets_end_index+1,:]
                    for ichannel in range(self.trainingdatasets[site]['data'].shape[1]):
                        temp_desired_trainingdatasets_nan_value_indices = np.where(np.isnan(temp_desired_trainingdatasets[i,training_batch_indices,ichannel]))
                        temp_desired_trainingdatasets[i,training_batch_indices,ichannel][temp_desired_trainingdatasets_nan_value_indices] = np.nanmean(temp_desired_trainingdatasets[i,training_batch_indices,ichannel])
            ## Assign the desired range of input datasets to the desired dictionaries
            desired_ground_observation_data[site]['PM25'] = temp_desired_observation_data
            desired_true_input[site]['data'] = temp_desired_true_input
            desired_trainingdatasets[site]['data'] = temp_desired_trainingdatasets
            desired_geophysical_species_data[site]['geoPM25'] = temp_desired_geophysical_species_data

            desired_trainingdatasets[site]['dates'] = total_dates_array        
            desired_true_input[site]['dates'] = total_dates_array
            desired_ground_observation_data[site]['dates'] = total_dates_array
            desired_geophysical_species_data[site]['dates'] = total_dates_array

        return desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data
    
    def normalize_trainingdatasets(self,desired_trainingdatasets):

        ## We already calculated the mean and std of the training datasets based on the whole training datasets.
        ## Now we need to normalize the desired training datasets based on the mean and std matrix.
        desired_normalized_trainingdatasets = copy.deepcopy(desired_trainingdatasets)
        for isite in self.ground_observation_data.keys():
            desired_normalized_trainingdatasets[str(isite)]['data'] = (desired_normalized_trainingdatasets[str(isite)]['data'] - self.TrainingDatasets_mean) / self.TrainingDatasets_std
        return desired_normalized_trainingdatasets
    
    def concatenate_trainingdatasets(self,desired_true_input, desired_normalized_trainingdatasets,desired_ground_observation_data, desired_geophysical_species_data):
        ## We first get the desired range of input datasets based on the start_date and end_date
        ## Then we normalize the desired training datasets based on the mean and std matrix.
        ## Here we need to concatenate the normalized training datasets and true inputs, respectively.
        
        
        datasets = [desired_normalized_trainingdatasets[str(isite)]['data'] for isite in self.ground_observation_data.keys()]
        total_trainingdatasets = np.concatenate(datasets, axis=0).astype(np.float32)

        datasets = [desired_true_input[str(isite)]['data'] for isite in self.ground_observation_data.keys()]
        total_true_input = np.concatenate(datasets, axis=0).astype(np.float32)

        datasets = [desired_ground_observation_data[str(isite)]['PM25'] for isite in self.ground_observation_data.keys()]
        total_ground_observation_data = np.concatenate(datasets, axis=0).astype(np.float32)
        
        datasets = [desired_geophysical_species_data[str(isite)]['geoPM25'] for isite in self.ground_observation_data.keys()]
        total_geophysical_species_data = np.concatenate(datasets, axis=0).astype(np.float32)
        
        datasets = [desired_normalized_trainingdatasets[str(isite)]['dates'] for isite in self.ground_observation_data.keys()]
        total_dates = np.concatenate(datasets, axis=0).astype(np.float32)

        datasets = [int(isite)*np.ones(len(desired_normalized_trainingdatasets[str(isite)]['dates']),dtype=int) for isite in self.ground_observation_data.keys()]
        total_sites_index = np.concatenate(datasets, axis=0)

        ### Remove the NaN batch in the total_trainingdatasets, total_true_input, total_ground_observation_data, total_geophysical_species_data
        nan_indices = np.isnan(total_true_input).all(axis=(1, 2))
        ## Remove the NaN batches
        total_trainingdatasets = total_trainingdatasets[~nan_indices]
        total_true_input = total_true_input[~nan_indices]
        total_ground_observation_data = total_ground_observation_data[~nan_indices]
        total_geophysical_species_data = total_geophysical_species_data[~nan_indices]
        total_dates = total_dates[~nan_indices]
        total_sites_index = total_sites_index[~nan_indices]

        #print('total_sites_index.shape: ',total_sites_index.shape)
        return total_trainingdatasets, total_true_input, total_ground_observation_data,total_geophysical_species_data, total_sites_index, total_dates
    

class CNNInputDatasets():
    def __init__(self,species,total_channel_names,bias,normalize_bias,normalize_species,absolute_species,datapoints_threshold):
        self.datapoints_threshold = datapoints_threshold
        self.species = species
        self.bias = bias
        self.normalize_bias = normalize_bias
        self.normalize_species = normalize_species
        self.absolute_species = absolute_species
        self.total_channel_names = total_channel_names

        self.ground_observation_data = self._load_daily_PM25_data() 
        ## First time examin the ground_observation_data, wether it has negative values
        for isite in self.ground_observation_data.keys():
            temp_data = self.ground_observation_data[isite]['PM25']
            index = np.where(temp_data < 0)[0]
            if len(index) > 0:
                #print('First check - Warning: Site {} has negative values in PM2.5 data!'.format(isite))
               # print('Negative values: ', temp_data[index])
                self.ground_observation_data[isite]['PM25'][index] = np.nan
            else:
                None
                #print('Site {} has no negative values in PM2.5 data.'.format(isite))

        self.geophysical_species_data,self.bias_data = self._load_geophyscial_PMandBias_data()
        self.ground_observation_data, self.geophysical_species_data, self.bias_data,self.delete_all_data_sites, self.delete_specific_data_sites, self.delete_specific_data_indices = self._get_nonan_and_threshold_sites()

        # Second time examin the ground_observation_data, wether it has negative values
        for isite in self.ground_observation_data.keys():
            temp_data = self.ground_observation_data[isite]['PM25']
            index = np.where(temp_data < 0)[0]
            if len(index) > 0:
                #print('Second check - Warning: Site {} has negative values in PM2.5 data!'.format(isite))
                #print('Negative values: ', temp_data[index])
                self.ground_observation_data[isite]['PM25'][index] = np.nan
            else:
                None
                #print('Site {} has no negative values in PM2.5 data.'.format(isite))

        self.true_input, self.true_input_mean, self.true_input_std = self._Learning_objective()
        self.total_sites_number = len(self.true_input.keys())
        self.sites_lat, self.sites_lon = self._get_lat_lon_arrays()

        start_time = time.time()
        print('Start loading and aggregating the training datasets...')
        self.trainingdatasets = self._load_aggregate_TrainingDatasets(total_channel_names, self.total_sites_number)
        print('The time of loading and aggregating the training datasets is {} seconds'.format(time.time()-start_time))
        
        self.ground_observation_data, self.geophysical_species_data, self.bias_data, self.trainingdatasets, self.true_input = self._rename_key_in_dict()

        # Third time examin the ground_observation_data, wether it has negative values
        for isite in self.ground_observation_data.keys():
            temp_data = self.ground_observation_data[isite]['PM25']
            index = np.where(temp_data < 0)[0]
            if len(index) > 0:
                #print('Third check - Warning: Site {} has negative values in PM2.5 data!'.format(isite))
                #print('Negative values: ', temp_data[index])
                self.ground_observation_data[isite]['PM25'][index] = np.nan
            else:
                None
                #print('Site {} has no negative values in PM2.5 data.'.format(isite))
        print('Start deriving the training datasets normalization matrix...')
        start_time = time.time()
        self.TrainingDatasets_mean, self.TrainingDatasets_std = self._derive_trainingdatasets_normalization_matrix()
        print('The time of deriving the training datasets normalization matrix is {} seconds'.format(time.time()-start_time))

        self.width, self.height = self.TrainingDatasets_mean.shape[1], self.TrainingDatasets_mean.shape[2]

    def _get_nonan_and_threshold_sites(self):
        temp_observation_data = copy.deepcopy(self.ground_observation_data)
        temp_geophysical_species_data = copy.deepcopy(self.geophysical_species_data)
        temp_bias_data = copy.deepcopy(self.bias_data)

        delete_all_data_sites = []
        delete_specific_data_sites = []
        delete_specific_data_indices = []

        for isite in self.ground_observation_data.keys():
            site = str(isite)
            
            if np.isnan(self.geophysical_species_data[site]['geoPM25']).any():
                #print('Warning: Site {} has NaN values in PM2.5 data!'.format(site))
                if np.isnan(self.geophysical_species_data[site]['geoPM25']).all():
                    del temp_observation_data[site]
                    del temp_geophysical_species_data[site]
                    del temp_bias_data[site]
                    delete_all_data_sites.append(site)
                else:
                    indice = np.where(np.isnan(self.geophysical_species_data[site]['geoPM25']))[0]
                    delete_specific_data_sites.append(site)
                    delete_specific_data_indices.append(indice)
                    temp_observation_data[site]['PM25'] = np.delete(temp_observation_data[site]['PM25'], indice, axis=0)
                    temp_observation_data[site]['dates'] = np.delete(temp_observation_data[site]['dates'], indice, axis=0)
                    temp_geophysical_species_data[site]['geoPM25'] = np.delete(temp_geophysical_species_data[site]['geoPM25'], indice, axis=0)
                    temp_geophysical_species_data[site]['dates'] = np.delete(temp_geophysical_species_data[site]['dates'], indice, axis=0)
                    temp_bias_data[site]['geobias'] = np.delete(temp_bias_data[site]['geobias'], indice, axis=0)
                    temp_bias_data[site]['dates'] = np.delete(temp_bias_data[site]['dates'], indice, axis=0)

            elif len(self.ground_observation_data[site]['PM25']) < self.datapoints_threshold:
                    #print('Warning: Site {} has less than {} data points!'.format(site, self.datapoints_threshold))
                    del temp_observation_data[site]
                    del temp_geophysical_species_data[site]
                    del temp_bias_data[site]
                    delete_all_data_sites.append(site)
            else:
                None
                #print('Site {} has no NaN values in PM2.5 data.'.format(site))
        return temp_observation_data, temp_geophysical_species_data, temp_bias_data, delete_all_data_sites, delete_specific_data_sites, delete_specific_data_indices

    def _load_daily_PM25_data(self):
        '''
        Load daily PM2.5 data from the directory of obs_indir. 
        return: data - it is a dictionary with keys as the site loc, date, and values as the daily PM2.5 data.
        '''
        infile = ground_observation_data_dir + ground_observation_data_infile
        if not os.path.exists(infile):
            raise ValueError('Observation file - The {} file does not exist!'.format(infile))
        data = np.load(infile,allow_pickle=True).item()
        return data

    def _load_geophyscial_PMandBias_data(self):
        PM_infile = geophysical_species_data_dir + geophysical_species_data_infile
        Bias_infile = geophysical_biases_data_dir + geophysical_biases_data_infile
        if not os.path.exists(PM_infile):
            raise ValueError('Geophysical species file - The {} file does not exist!'.format(PM_infile))
        if not os.path.exists(Bias_infile):
            raise ValueError('Geophysical biases file - The {} file does not exist!'.format(Bias_infile))
        PM_data = np.load(PM_infile,allow_pickle=True).item()
        Bias_data = np.load(Bias_infile,allow_pickle=True).item()
        return PM_data, Bias_data
    
    def _Learning_objective(self):
        if self.bias == True:
            PM_data, Bias_data = copy.deepcopy(self.geophysical_species_data),copy.deepcopy(self.bias_data)

            true_input = Bias_data
            mean = 0
            std = 1
            return true_input, mean, std
        
        elif self.normalize_bias:
            PM_data, Bias_data = copy.deepcopy(self.geophysical_species_data),copy.deepcopy(self.bias_data)
            for index, isite in enumerate(Bias_data.keys()):
                if index == 0:
                    total_bias_data = Bias_data[str(isite)]['geobias']
                else:
                    total_bias_data = np.concatenate((total_bias_data, Bias_data[str(isite)]['geobias']), axis=0)
            if normalize_type == 'Gaussian':
                bias_mean = np.mean(total_bias_data)
                bias_std = np.std(total_bias_data)
            elif normalize_type == 'MinMax':
                bias_mean = np.min(total_bias_data)
                bias_std = np.max(total_bias_data) - np.min(total_bias_data)
            elif normalize_type == 'Robust':
                bias_mean = np.median(total_bias_data)
                bias_std = np.percentile(total_bias_data, 75) - np.percentile(total_bias_data, 25)

            for index, isite in enumerate(Bias_data.keys()):
                Bias_data[str(isite)]['geobias'] = (Bias_data[str(isite)]['geobias'] - bias_mean) / bias_std
            
            true_input = Bias_data
            return true_input, bias_mean, bias_std
        
        elif self.absolute_species:
            PM_data = copy.deepcopy(self.ground_observation_data)

            true_input = PM_data
            mean = 0
            std = 1
            return true_input, mean, std
        
        elif self.normalize_species:
            PM_data = copy.deepcopy(self.ground_observation_data)

            for index,isite in enumerate(PM_data.keys()):
                if index == 0:
                    total_PM_data = PM_data[str(isite)]['PM25']
                else:
                    total_PM_data = np.concatenate((total_PM_data, PM_data[str(isite)]['PM25']), axis=0)
            if normalize_type == 'Gaussian':
                PM_mean = np.mean(total_PM_data)
                PM_std = np.std(total_PM_data)
            elif normalize_type == 'MinMax':
                PM_mean = np.min(total_PM_data)
                PM_std = np.max(total_PM_data) - np.min(total_PM_data)
            elif normalize_type == 'Robust':
                PM_mean = np.median(total_PM_data)
                PM_std = np.percentile(total_PM_data, 75) - np.percentile(total_PM_data, 25)
            for index,isite in enumerate(PM_data.keys()):
                PM_data[str(isite)]['PM25'] = (PM_data[str(isite)]['PM25'] - PM_mean) / PM_std
            true_input = PM_data
            return true_input, PM_mean, PM_std

    def _get_lat_lon_arrays(self):
        lat_array = np.zeros(self.total_sites_number,dtype=np.float32)
        lon_array = np.zeros(self.total_sites_number,dtype=np.float32)

        for i,isite in enumerate(self.ground_observation_data.keys()):
            site = str(isite)
            lat_array[i] = self.ground_observation_data[site]['lat']
            lon_array[i] = self.ground_observation_data[site]['lon']
        return lat_array, lon_array
    
    def _process_concatenate_site_data(self, data, temp_data, isite):
        site = str(isite)
        data[site]['data'] = np.concatenate((data[site]['data'], temp_data[site]['data']), axis=1)
        return data

    def _load_aggregate_TrainingDatasets(self, Training_Channels, sites_number):
        '''
        Here I compared the differences of time consuming between the old and new 
        version of loading and aggregating the training datasets.

        >>> data = load_aggregate_TrainingDatasets(channel_names,1684)
        The time of loading and aggregating the training datasets is 11.369443893432617 seconds
        >>> data = old_load_aggregate_TrainingDatasets(channel_names,1684)
        The time of old loading and aggregating the training datasets is 26.89457082748413 seconds 

        def old_load_aggregate_TrainingDatasets(Training_Channels, sites_number):
            start_time = time.time()
            
            for i, channel_name in enumerate(Training_Channels):
                infile = CNN_Training_infiles.format(channel_name)
                if os.path.exists(infile):
                    temp_data = np.load(infile, allow_pickle=True).item()
                else:
                    raise ValueError('The {} file does not exist!'.format(infile))
                
                if i == 0:
                    data = temp_data
                else:
                    for isite in range(sites_number):
                        site = str(isite)
                        data[site]['data'] = np.concatenate((data[site]['data'], temp_data[site]['data']), axis=1)
            end_time = time.time()
            print('The time of old loading and aggregating the training datasets is {} seconds'.format(end_time-start_time))
            return data'
        '''

        
        for i, channel_name in enumerate(Training_Channels):
            infile = CNN_Training_infiles.format(channel_name)
            if os.path.exists(infile):
                temp_data = np.load(infile, allow_pickle=True).item()
            else:
                raise ValueError('Training file - The {} file does not exist!'.format(infile))
            
            if i == 0:
                
                data = temp_data
                print('Init Training Data Shape: ' ,data['0']['data'].shape)
            else:
                with ThreadPoolExecutor() as executor:
                    print('channel_name: ', channel_name)
                    futures = [executor.submit(self._process_concatenate_site_data, data, temp_data, isite) for isite in self.ground_observation_data.keys()]
                    for future in futures:
                        future.result()
        print('Training Data Shape: ' ,data['0']['data'].shape)
        for isite in self.delete_all_data_sites:
            if str(isite) in data.keys():
                del data[str(isite)]
                #print('Site {} has been deleted from the training datasets.'.format(isite))
            else:
                None

                #print('Site {} is not in the training datasets.'.format(isite))

        for i, isite,in enumerate(self.delete_specific_data_sites):
            if str(isite) in data.keys():
                indices = self.delete_specific_data_indices[i]
                data[str(isite)]['data'] = np.delete(data[str(isite)]['data'], indices, axis=0)
                data[str(isite)]['dates'] = np.delete(data[str(isite)]['dates'], indices, axis=0)
                #print('Site {} has been deleted from the training datasets at indices {}.'.format(isite, indices))
            else:
                None
                #print('Site {} is not in the training datasets.'.format(isite))
        return data
    
    def _derive_trainingdatasets_normalization_matrix(self):
       
    
        # Collect all site data in a list
        datasets = [self.trainingdatasets[str(isite)]['data'] for isite in self.ground_observation_data.keys()]
        
        # Concatenate all at once (much faster than incremental concatenation)
        total_trainingdatasets = np.concatenate(datasets, axis=0)

        # Compute mean and std along the feature axis
        if training_data_normalization_type == 'Gaussian':
            TrainingDatasets_mean = np.mean(total_trainingdatasets, axis=0)
            TrainingDatasets_std = np.std(total_trainingdatasets, axis=0)
        elif training_data_normalization_type == 'MinMax':
            TrainingDatasets_mean = np.min(total_trainingdatasets, axis=0)
            TrainingDatasets_std = np.max(total_trainingdatasets, axis=0) - np.min(total_trainingdatasets, axis=0)
        elif training_data_normalization_type == 'Robust':
            TrainingDatasets_mean = np.median(total_trainingdatasets, axis=0)
            TrainingDatasets_std = np.percentile(total_trainingdatasets, 75, axis=0) - np.percentile(total_trainingdatasets, 25, axis=0)

        return TrainingDatasets_mean, TrainingDatasets_std
    
    def _rename_key_in_dict(self):
        temp_observation_data = {}
        temp_geophysical_species_data = {}
        temp_bias_data = {}
        temp_trainingdatasets = {}
        temp_true_input = {}
        for isite, site in enumerate(self.ground_observation_data.keys()):
            temp_observation_data[str(isite)] = self.ground_observation_data[site]
            temp_geophysical_species_data[str(isite)] = self.geophysical_species_data[site]
            temp_bias_data[str(isite)] = self.bias_data[site]
            temp_trainingdatasets[str(isite)] = self.trainingdatasets[site]
            temp_true_input[str(isite)] = self.true_input[site]
        return temp_observation_data, temp_geophysical_species_data, temp_bias_data, temp_trainingdatasets, temp_true_input
    
    def get_desired_range_inputdatasets(self,start_date,end_date):

        ## This is used to get the desired range of input datasets
        desired_true_input = copy.deepcopy(self.true_input)
        desired_trainingdatasets = copy.deepcopy(self.trainingdatasets)
        desired_ground_observation_data = copy.deepcopy(self.ground_observation_data)
        desired_geophysical_species_data = copy.deepcopy(self.geophysical_species_data)
        ## Here we need to get the desired range of input datasets based on the start_date and end_date
        for isite in self.ground_observation_data.keys():
            site = str(isite)
            if len(np.where(desired_true_input[site]['dates'] >= start_date)[0]) == 0 or len(np.where(desired_true_input[site]['dates'] <= end_date)[0]) == 0:
                #print('The start_date or end_date is out of range for site {}!'.format(site))
                desired_trainingdatasets[site]['data'] = np.empty((0, len(self.total_channel_names), self.height, self.width), dtype=np.float32)
                if self.bias == True or self.normalize_bias == True:
                    desired_true_input[site]['geobias'] = np.array([])
                else:
                    desired_true_input[site]['PM25'] = np.array([])
                desired_ground_observation_data[site]['PM25'] = np.array([])
                desired_geophysical_species_data[site]['geoPM25'] = np.array([])
                desired_trainingdatasets[site]['dates'] = np.array([])
                desired_true_input[site]['dates'] = np.array([])
                desired_ground_observation_data[site]['dates'] = np.array([])
                desired_geophysical_species_data[site]['dates'] = np.array([])
            else:
                start_index = np.where(desired_true_input[site]['dates'] >= start_date)[0][0]
                end_index = np.where(desired_true_input[site]['dates'] <= end_date)[0][-1]
                trainingdatasets_start_index = np.where(desired_trainingdatasets[site]['dates'] >= start_date)[0][0]
                trainingdatasets_end_index = np.where(desired_trainingdatasets[site]['dates'] <= end_date)[0][-1]

                desired_trainingdatasets[site]['data'] = self.trainingdatasets[site]['data'][trainingdatasets_start_index:trainingdatasets_end_index+1]
                if self.bias == True or self.normalize_bias == True:
                    desired_true_input[site]['geobias'] = self.true_input[site]['geobias'][start_index:end_index+1]
                else:
                    desired_true_input[site]['PM25'] = self.true_input[site]['PM25'][start_index:end_index+1]
                desired_ground_observation_data[site]['PM25'] = self.ground_observation_data[site]['PM25'][start_index:end_index+1]
                desired_geophysical_species_data[site]['geoPM25'] = self.geophysical_species_data[site]['geoPM25'][start_index:end_index+1]

                desired_trainingdatasets[site]['dates'] = self.trainingdatasets[site]['dates'][trainingdatasets_start_index:trainingdatasets_end_index+1]            
                desired_true_input[site]['dates'] = self.true_input[site]['dates'][start_index:end_index+1]
                desired_ground_observation_data[site]['dates'] = self.ground_observation_data[site]['dates'][start_index:end_index+1]
                desired_geophysical_species_data[site]['dates'] = self.geophysical_species_data[site]['dates'][start_index:end_index+1]
                
        return desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data

    def normalize_trainingdatasets(self,desired_trainingdatasets):

        ## We already calculated the mean and std of the training datasets based on the whole training datasets.
        ## Now we need to normalize the desired training datasets based on the mean and std matrix.
        desired_normalized_trainingdatasets = copy.deepcopy(desired_trainingdatasets)
        for isite in self.ground_observation_data.keys():
            desired_normalized_trainingdatasets[str(isite)]['data'] = (desired_normalized_trainingdatasets[str(isite)]['data'] - self.TrainingDatasets_mean) / self.TrainingDatasets_std
        return desired_normalized_trainingdatasets
    
    def concatenate_trainingdatasets(self,desired_true_input, desired_normalized_trainingdatasets,desired_ground_observation_data, desired_geophysical_species_data):
        ## We first get the desired range of input datasets based on the start_date and end_date
        ## Then we normalize the desired training datasets based on the mean and std matrix.
        ## Here we need to concatenate the normalized training datasets and true inputs, respectively.
        
        
        datasets = [desired_normalized_trainingdatasets[str(isite)]['data'] for isite in self.ground_observation_data.keys()]
        total_trainingdatasets = np.concatenate(datasets, axis=0)
        
        if self.bias == True or self.normalize_bias == True:
            datasets = [desired_true_input[str(isite)]['geobias'] for isite in self.ground_observation_data.keys()]
            total_true_input = np.concatenate(datasets, axis=0)
        else:
            datasets = [desired_true_input[str(isite)]['PM25'] for isite in self.ground_observation_data.keys()]
            total_true_input = np.concatenate(datasets, axis=0)
        
        datasets = [desired_ground_observation_data[str(isite)]['PM25'] for isite in self.ground_observation_data.keys()]
        total_ground_observation_data = np.concatenate(datasets, axis=0)
        
        datasets = [desired_geophysical_species_data[str(isite)]['geoPM25'] for isite in self.ground_observation_data.keys()]
        total_geophysical_species_data = np.concatenate(datasets, axis=0)
        
        datasets = [desired_normalized_trainingdatasets[str(isite)]['dates'] for isite in self.ground_observation_data.keys()]
        total_dates = np.concatenate(datasets, axis=0)

        datasets = [int(isite)*np.ones(len(desired_normalized_trainingdatasets[str(isite)]['dates']),dtype=int) for isite in self.ground_observation_data.keys()]
        total_sites_index = np.concatenate(datasets, axis=0)
        print('total_sites_index.shape: ',total_sites_index.shape)
        return total_trainingdatasets, total_true_input, total_ground_observation_data,total_geophysical_species_data, total_sites_index, total_dates
    


class CNN3DInputDatasets():
    def __init__(self,species,total_channel_names,bias,normalize_bias,normalize_species,absolute_species,datapoints_threshold):
        self.datapoints_threshold = datapoints_threshold
        self.species = species
        self.bias = bias
        self.normalize_bias = normalize_bias
        self.normalize_species = normalize_species
        self.absolute_species = absolute_species
        self.total_channel_names = total_channel_names


        self.ground_observation_data = self._load_daily_PM25_data() 
        self.geophysical_species_data,self.bias_data = self._load_geophyscial_PMandBias_data()
        self.ground_observation_data, self.geophysical_species_data, self.bias_data,self.delete_all_data_sites, self.delete_specific_data_sites, self.delete_specific_data_indices = self._get_nonan_and_threshold_sites()

        self.true_input, self.true_input_mean, self.true_input_std = self._Learning_objective()
        self.total_sites_number = len(self.true_input.keys())
        self.sites_lat, self.sites_lon = self._get_lat_lon_arrays()
        self.depth = ResNet3D_depth
        start_time = time.time()
        print('Start loading and aggregating the 3D training datasets...')
        self.trainingdatasets = self._load_aggregate_3DTrainingDatasets(total_channel_names, self.total_sites_number)

        print('The time of loading and aggregating the training datasets is {} seconds'.format(time.time()-start_time))
        self.ground_observation_data, self.geophysical_species_data, self.bias_data, self.trainingdatasets, self.true_input = self._rename_key_in_dict()
        
        print('Start deriving the training datasets normalization matrix...')
        start_time = time.time()
        self.TrainingDatasets_mean, self.TrainingDatasets_std = self._derive_trainingdatasets_normalization_matrix()
        print('The time of deriving the training datasets normalization matrix is {} seconds'.format(time.time()-start_time))

        self.height, self.width = self.TrainingDatasets_mean.shape[2], self.TrainingDatasets_mean.shape[3]
    def _load_daily_PM25_data(self):
        '''
        Load daily PM2.5 data from the directory of obs_indir. 
        return: data - it is a dictionary with keys as the site loc, date, and values as the daily PM2.5 data.
        '''
        infile = ground_observation_data_dir + ground_observation_data_infile
        if not os.path.exists(infile):
            raise ValueError('Observation file - The {} file does not exist!'.format(infile))
        data = np.load(infile,allow_pickle=True).item()

        return data

    def _load_geophyscial_PMandBias_data(self):
        PM_infile = geophysical_species_data_dir + geophysical_species_data_infile
        Bias_infile = geophysical_biases_data_dir + geophysical_biases_data_infile
        if not os.path.exists(PM_infile):
            raise ValueError('Geophysical species file - The {} file does not exist!'.format(PM_infile))
        if not os.path.exists(Bias_infile):
            raise ValueError('Geophysical biases file - The {} file does not exist!'.format(Bias_infile))
        PM_data = np.load(PM_infile,allow_pickle=True).item()
        Bias_data = np.load(Bias_infile,allow_pickle=True).item()
        return PM_data, Bias_data
    
    def _get_nonan_and_threshold_sites(self):
        temp_observation_data = copy.deepcopy(self.ground_observation_data)
        temp_geophysical_species_data = copy.deepcopy(self.geophysical_species_data)
        temp_bias_data = copy.deepcopy(self.bias_data)

        delete_all_data_sites = []
        delete_specific_data_sites = []
        delete_specific_data_indices = []

        for isite in self.ground_observation_data.keys():
            site = str(isite)
            if np.isnan(self.geophysical_species_data[site]['geoPM25']).any():
                
                #print('Warning: Site {} has NaN values in PM2.5 data!'.format(site))
                if np.isnan(self.geophysical_species_data[site]['geoPM25']).all():
                    del temp_observation_data[site]
                    del temp_geophysical_species_data[site]
                    del temp_bias_data[site]
                    delete_all_data_sites.append(site)
                else:
                    indice = np.where(np.isnan(self.geophysical_species_data[site]['geoPM25']))[0]
                    delete_specific_data_sites.append(site)
                    delete_specific_data_indices.append(indice)
                    temp_observation_data[site]['PM25'] = np.delete(temp_observation_data[site]['PM25'], indice, axis=0)
                    temp_observation_data[site]['dates'] = np.delete(temp_observation_data[site]['dates'], indice, axis=0)
                    temp_geophysical_species_data[site]['geoPM25'] = np.delete(temp_geophysical_species_data[site]['geoPM25'], indice, axis=0)
                    temp_geophysical_species_data[site]['dates'] = np.delete(temp_geophysical_species_data[site]['dates'], indice, axis=0)
                    temp_bias_data[site]['geobias'] = np.delete(temp_bias_data[site]['geobias'], indice, axis=0)
                    temp_bias_data[site]['dates'] = np.delete(temp_bias_data[site]['dates'], indice, axis=0)
            elif len(self.ground_observation_data[site]['PM25']) < self.datapoints_threshold:
                    #print('Warning: Site {} has less than {} data points!'.format(site, self.datapoints_threshold))
                    del temp_observation_data[site]
                    del temp_geophysical_species_data[site]
                    del temp_bias_data[site]
                    delete_all_data_sites.append(site)
            else:
                None
                #print('Site {} has no NaN values in PM2.5 data.'.format(site))
        return temp_observation_data, temp_geophysical_species_data, temp_bias_data, delete_all_data_sites, delete_specific_data_sites, delete_specific_data_indices


    def _Learning_objective(self):
        if self.bias == True:
            PM_data, Bias_data = copy.deepcopy(self.geophysical_species_data),copy.deepcopy(self.bias_data)
            true_input = Bias_data
            mean = 0
            std = 1
            return true_input, mean, std
        
        elif self.normalize_bias:
            PM_data, Bias_data = copy.deepcopy(self.geophysical_species_data),copy.deepcopy(self.bias_data)
            for index,isite in enumerate(Bias_data.keys()):
                if index == 0:
                    total_bias_data = Bias_data[str(isite)]['geobias']
                else:
                    total_bias_data = np.concatenate((total_bias_data, Bias_data[str(isite)]['geobias']), axis=0)
            if normalize_type == 'Gaussian':
                bias_mean = np.mean(total_bias_data)
                bias_std = np.std(total_bias_data)
            elif normalize_type == 'MinMax':
                bias_mean = np.min(total_bias_data)
                bias_std = np.max(total_bias_data) - np.min(total_bias_data)
            elif normalize_type == 'Robust':
                bias_mean = np.median(total_bias_data)
                bias_std = np.percentile(total_bias_data, 75) - np.percentile(total_bias_data, 25)

            for isite in Bias_data.keys():
                Bias_data[str(isite)]['geobias'] = (Bias_data[str(isite)]['geobias'] - bias_mean) / bias_std
            
            true_input = Bias_data
            return true_input, bias_mean, bias_std
        
        elif self.absolute_species:
            PM_data = copy.deepcopy(self.ground_observation_data)
            true_input = PM_data
            mean = 0
            std = 1
            return true_input, mean, std
        
        elif self.normalize_species:
            PM_data = copy.deepcopy(self.ground_observation_data)
            for index,isite in enumerate(PM_data.keys()):
                if index == 0:
                    total_PM_data = PM_data[str(isite)]['PM25']
                else:
                    total_PM_data = np.concatenate((total_PM_data, PM_data[str(isite)]['PM25']), axis=0)
            if normalize_type == 'Gaussian':
                PM_mean = np.mean(total_PM_data)
                PM_std = np.std(total_PM_data)
            elif normalize_type == 'MinMax':    
                PM_mean = np.min(total_PM_data)
                PM_std = np.max(total_PM_data) - np.min(total_PM_data)
            elif normalize_type == 'Robust':
                PM_mean = np.median(total_PM_data)
                PM_std = np.percentile(total_PM_data, 75) - np.percentile(total_PM_data, 25)
            for index,isite in enumerate(PM_data.keys()):
                PM_data[str(isite)]['PM25'] = (PM_data[str(isite)]['PM25'] - PM_mean) / PM_std
            true_input = PM_data
            return true_input, PM_mean, PM_std

    def _get_lat_lon_arrays(self):
        lat_array = np.zeros(self.total_sites_number,dtype=np.float32)
        lon_array = np.zeros(self.total_sites_number,dtype=np.float32)

        for i,isite in enumerate(self.ground_observation_data.keys()):
            site = str(isite)
            lat_array[i] = self.ground_observation_data[site]['lat']
            lon_array[i] = self.ground_observation_data[site]['lon']
        return lat_array, lon_array
    
    def _process_concatenate_site_data(self, data, temp_data, isite):
        site = str(isite)
        data[site]['data'] = np.concatenate((data[site]['data'], temp_data[site]['data']), axis=1)
        return data
    
    def _convert_to_3Dinputdatasets(self, data, istie):
        site = str(istie)
        indices = [np.where(data[site]['dates_withdepth'] == date)[0][0] for date in data[site]['dates']]
        #print('sites: {}, length of indices: '.format(site), len(indices))
        #print('sites: {}, length of data[site][\'dates\']'.format(site),(len(data[site]['dates'])))
        temp_3D_data = np.zeros((len(data[site]['dates']),len(self.total_channel_names), self.depth, self.height, self.width), dtype=np.float32)
        for i, index in enumerate(indices):
            temp_3D_data[i,:,:,:,:] = data[site]['data'][(index+1-self.depth):index+1,:,:,:].swapaxes(0,1)
        data[site]['data'] = temp_3D_data
        return data
    def _load_aggregate_3DTrainingDatasets(self, Training_Channels, sites_number):
        for i, channel_name in enumerate(Training_Channels):
            infile = CNN3D_Training_infiles.format(channel_name)
            if os.path.exists(infile):
                temp_data = np.load(infile, allow_pickle=True).item()
            else:
                raise ValueError('Training file - The {} file does not exist!'.format(infile))
            
            if i == 0:
                
                data = temp_data
                print('Init 3D Training Data Shape: ' ,data['0']['data'].shape)
                self.width, self.height = data['0']['data'].shape[3], data['0']['data'].shape[2]
            else:
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self._process_concatenate_site_data, data, temp_data, isite) for isite in self.ground_observation_data.keys()]
                    for future in futures:
                        future.result()

        for isite in self.delete_all_data_sites:
            if str(isite) in data.keys():
                del data[str(isite)]
                #print('Site {} has been deleted from the training datasets.'.format(isite))
            else:
                #print('Site {} is not in the training datasets.'.format(isite))
                None


        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._convert_to_3Dinputdatasets, data, isite) for isite in self.ground_observation_data.keys()]
            for future in futures:
                future.result()
        for i, isite,in enumerate(self.delete_specific_data_sites):
            if str(isite) in data.keys():
                indices = self.delete_specific_data_indices[i]
                data[str(isite)]['data'] = np.delete(data[str(isite)]['data'], indices, axis=0)
                data[str(isite)]['dates'] = np.delete(data[str(isite)]['dates'], indices, axis=0)
                #print('Site {} has been deleted from the training datasets at indices {}.'.format(isite, indices))
            else:
                #print('Site {} is not in the training datasets.'.format(isite))
                None

        print('Training Data Shape: ' ,data['0']['data'].shape)

        return data
    def _rename_key_in_dict(self):
        temp_observation_data = {}
        temp_geophysical_species_data = {}
        temp_bias_data = {}
        temp_trainingdatasets = {}
        temp_true_input = {}
        for isite, site in enumerate(self.ground_observation_data.keys()):
            temp_observation_data[str(isite)] = self.ground_observation_data[site]
            temp_geophysical_species_data[str(isite)] = self.geophysical_species_data[site]
            temp_bias_data[str(isite)] = self.bias_data[site]
            temp_trainingdatasets[str(isite)] = self.trainingdatasets[site]
            temp_true_input[str(isite)] = self.true_input[site]
        return temp_observation_data, temp_geophysical_species_data, temp_bias_data, temp_trainingdatasets, temp_true_input
    

    def _derive_trainingdatasets_normalization_matrix(self):
       
    
        # Collect all site data in a list
        datasets = [self.trainingdatasets[str(isite)]['data'] for isite in self.ground_observation_data.keys()]
        
        # Concatenate all at once (much faster than incremental concatenation)
        total_trainingdatasets = np.concatenate(datasets, axis=0)

        # Compute mean and std along the feature axis
        if training_data_normalization_type == 'Gaussian':
            TrainingDatasets_mean = np.mean(total_trainingdatasets, axis=0)
            TrainingDatasets_std = np.std(total_trainingdatasets, axis=0)
        elif training_data_normalization_type == 'MinMax':  
            TrainingDatasets_mean = np.min(total_trainingdatasets, axis=0)
            TrainingDatasets_std = np.max(total_trainingdatasets, axis=0) - np.min(total_trainingdatasets, axis=0)
        elif training_data_normalization_type == 'Robust':
            TrainingDatasets_mean = np.median(total_trainingdatasets, axis=0)
            TrainingDatasets_std = np.percentile(total_trainingdatasets, 75, axis=0) - np.percentile(total_trainingdatasets, 25, axis=0)
        return TrainingDatasets_mean, TrainingDatasets_std
    
    def get_desired_range_inputdatasets(self,start_date,end_date):

        ## This is used to get the desired range of input datasets
        desired_true_input = copy.deepcopy(self.true_input)
        desired_trainingdatasets = copy.deepcopy(self.trainingdatasets)
        desired_ground_observation_data = copy.deepcopy(self.ground_observation_data)
        desired_geophysical_species_data = copy.deepcopy(self.geophysical_species_data)
        ## Here we need to get the desired range of input datasets based on the start_date and end_date
        for isite in self.ground_observation_data.keys():
            site = str(isite)
            #print('isite: ',isite, 'start_date: ',start_date, 'end_date: ',end_date,'dates: ',desired_true_input[site]['dates'][0],' - ', desired_true_input[site]['dates'][-1])
            if len(np.where(desired_true_input[site]['dates'] >= start_date)[0]) == 0 or len(np.where(desired_true_input[site]['dates'] <= end_date)[0]) == 0:

                #print('The start_date or end_date is out of range for site {}!'.format(site))
                desired_trainingdatasets[site]['data'] = np.empty((0, len(self.total_channel_names), self.depth, self.height, self.width), dtype=np.float32)
                if self.bias == True or self.normalize_bias == True:
                    desired_true_input[site]['geobias'] = np.array([])
                else:
                    desired_true_input[site]['PM25'] = np.array([])
                desired_ground_observation_data[site]['PM25'] = np.array([])
                desired_geophysical_species_data[site]['geoPM25'] = np.array([])
                desired_trainingdatasets[site]['dates'] = np.array([])
                desired_true_input[site]['dates'] = np.array([])
                desired_ground_observation_data[site]['dates'] = np.array([])
                desired_geophysical_species_data[site]['dates'] = np.array([])

            else:
                start_index = np.where(desired_true_input[site]['dates'] >= start_date)[0][0]
                end_index = np.where(desired_true_input[site]['dates'] <= end_date)[0][-1]
                trainingdatasets_start_index = np.where(desired_trainingdatasets[site]['dates'] >= start_date)[0][0]
                trainingdatasets_end_index = np.where(desired_trainingdatasets[site]['dates'] <= end_date)[0][-1]

                desired_trainingdatasets[site]['data'] = self.trainingdatasets[site]['data'][trainingdatasets_start_index:trainingdatasets_end_index+1]
                if self.bias == True or self.normalize_bias == True:
                    desired_true_input[site]['geobias'] = self.true_input[site]['geobias'][start_index:end_index+1]
                else:
                    desired_true_input[site]['PM25'] = self.true_input[site]['PM25'][start_index:end_index+1]
                desired_ground_observation_data[site]['PM25'] = self.ground_observation_data[site]['PM25'][start_index:end_index+1]
                desired_geophysical_species_data[site]['geoPM25'] = self.geophysical_species_data[site]['geoPM25'][start_index:end_index+1]

                desired_trainingdatasets[site]['dates'] = self.trainingdatasets[site]['dates'][trainingdatasets_start_index:trainingdatasets_end_index+1]            
                desired_true_input[site]['dates'] = self.true_input[site]['dates'][start_index:end_index+1]
                desired_ground_observation_data[site]['dates'] = self.ground_observation_data[site]['dates'][start_index:end_index+1]
                desired_geophysical_species_data[site]['dates'] = self.geophysical_species_data[site]['dates'][start_index:end_index+1]
            
        return desired_trainingdatasets, desired_true_input,  desired_ground_observation_data, desired_geophysical_species_data

    def normalize_trainingdatasets(self,desired_trainingdatasets):

        ## We already calculated the mean and std of the training datasets based on the whole training datasets.
        ## Now we need to normalize the desired training datasets based on the mean and std matrix.
        desired_normalized_trainingdatasets = copy.deepcopy(desired_trainingdatasets)
        for isite in self.ground_observation_data.keys():
            desired_normalized_trainingdatasets[str(isite)]['data'] = (desired_normalized_trainingdatasets[str(isite)]['data'] - self.TrainingDatasets_mean) / self.TrainingDatasets_std
        return desired_normalized_trainingdatasets
    
    def concatenate_trainingdatasets(self,desired_true_input, desired_normalized_trainingdatasets,desired_ground_observation_data, desired_geophysical_species_data):
        ## We first get the desired range of input datasets based on the start_date and end_date
        ## Then we normalize the desired training datasets based on the mean and std matrix.
        ## Here we need to concatenate the normalized training datasets and true inputs, respectively.

        print('ground_observation_data.keys() :', self.ground_observation_data.keys())
        datasets = [desired_normalized_trainingdatasets[str(isite)]['data'] for isite in self.ground_observation_data.keys()]
        total_trainingdatasets = np.concatenate(datasets, axis=0).astype(np.float32)

        if self.bias == True or self.normalize_bias == True:
            datasets = [desired_true_input[str(isite)]['geobias'] for isite in self.ground_observation_data.keys()]
            total_true_input = np.concatenate(datasets, axis=0).astype(np.float32)
        else:
            datasets = [desired_true_input[str(isite)]['PM25'] for isite in self.ground_observation_data.keys()]
            total_true_input = np.concatenate(datasets, axis=0).astype(np.float32)

        datasets = [desired_ground_observation_data[str(isite)]['PM25'] for isite in self.ground_observation_data.keys()]
        total_ground_observation_data = np.concatenate(datasets, axis=0).astype(np.float32)

        datasets = [desired_geophysical_species_data[str(isite)]['geoPM25'] for isite in self.ground_observation_data.keys()]
        total_geophysical_species_data = np.concatenate(datasets, axis=0).astype(np.float32)
        
        datasets = [desired_normalized_trainingdatasets[str(isite)]['dates'] for isite in self.ground_observation_data.keys()]
        total_dates = np.concatenate(datasets, axis=0).astype(np.int32)

        datasets = [int(isite)*np.ones(len(desired_normalized_trainingdatasets[str(isite)]['dates']),dtype=int) for isite in self.ground_observation_data.keys()]
        total_sites_index = np.concatenate(datasets, axis=0)
        print('total_sites_index.shape: ',total_sites_index.shape)
        return total_trainingdatasets, total_true_input, total_ground_observation_data,total_geophysical_species_data, total_sites_index, total_dates
    

