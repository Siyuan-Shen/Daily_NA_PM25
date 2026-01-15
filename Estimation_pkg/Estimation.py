from Estimation_pkg.training_func import Train_Model_forEstimation
from Estimation_pkg.utils import *
from Estimation_pkg.predict_func import *
from Evaluation_pkg.utils import Get_typeName
import gc
import time

from Training_pkg.iostream import load_daily_datesbased_model


def Estimation_Func(total_channel_names, main_stream_channel_names,
                             side_stream_channel_names,sweep_id=None,):

    ### Get the number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Number of available GPUs: {world_size}")

    ### Training
    if Estimation_Train_model_Switch:
        Train_Model_forEstimation(total_channel_names, main_stream_channel_names,
                                   side_stream_channel_names, sweep_id=None,)

    ### Estimation
    if Map_estimation_Switch:
        Evaluation_type = 'Estimation'
        typeName = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=False, species=species)
        
        if Apply_CNN_architecture:
            ### Initialize the CNN datasets
            Model_structure_type = 'CNNModel'
            print('Init_CNN_Datasets starting...')
            start_time = time.time()
            Init_CNN_Datasets = CNNInputDatasets(species=species, total_channel_names=total_channel_names,bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,datapoints_threshold=observation_datapoints_threshold)
            print('Init_CNN_Datasets finished, time elapsed: ', time.time() - start_time)
            total_sites_number = Init_CNN_Datasets.total_sites_number
            true_input_mean, true_input_std = Init_CNN_Datasets.true_input_mean, Init_CNN_Datasets.true_input_std
            print('true_input_mean:', true_input_mean)
            print('true_input_std:', true_input_std)
            TrainingDatasets_mean, TrainingDatasets_std = Init_CNN_Datasets.TrainingDatasets_mean, Init_CNN_Datasets.TrainingDatasets_std
            width, height = Init_CNN_Datasets.width, Init_CNN_Datasets.height
            sites_lat, sites_lon = Init_CNN_Datasets.sites_lat, Init_CNN_Datasets.sites_lon
            del Init_CNN_Datasets
            gc.collect()
        elif Apply_3D_CNN_architecture:
            if MoE_Settings:
                Model_structure_type = '3DCNN_MoE_{}Experts_Model'.format(MoE_num_experts)
            elif MoCE_Settings:
                Model_structure_type = '3DCNN_MoCE_{}Experts_Model'.format(MoCE_num_experts)
            else:
                Model_structure_type = '3DCNNModel'
            print('Init_CNN_Datasets starting...')
            start_time = time.time()
            Init_CNN_Datasets = CNN3DInputDatasets(species=species, total_channel_names=total_channel_names,bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,datapoints_threshold=observation_datapoints_threshold)
            print('Init_CNN_Datasets finished, time elapsed: ', time.time() - start_time)
            total_sites_number = Init_CNN_Datasets.total_sites_number

            true_input_mean, true_input_std = Init_CNN_Datasets.true_input_mean, Init_CNN_Datasets.true_input_std
            print('true_input_mean:', true_input_mean)
            print('true_input_std:', true_input_std)
            TrainingDatasets_mean, TrainingDatasets_std = Init_CNN_Datasets.TrainingDatasets_mean, Init_CNN_Datasets.TrainingDatasets_std
            depth, width, height = Init_CNN_Datasets.depth,Init_CNN_Datasets.width, Init_CNN_Datasets.height
            sites_lat, sites_lon = Init_CNN_Datasets.sites_lat, Init_CNN_Datasets.sites_lon

            del Init_CNN_Datasets
            gc.collect()
        elif Apply_Transformer_architecture:
            Model_structure_type = 'TransformerModel'
            print('Init_Transformer_Datasets starting...')
            start_time = time.time()
            Init_Transformer_Datasets = TransformerInputDatasets(species=species, total_channel_names=total_channel_names,bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species,datapoints_threshold=observation_datapoints_threshold)
            print('Init_Transformer_Datasets finished, time elapsed: ', time.time() - start_time)

            total_sites_number = Init_Transformer_Datasets.total_sites_number

            true_input_mean, true_input_std = Init_Transformer_Datasets.true_input_mean, Init_Transformer_Datasets.true_input_std
            print('true_input_mean:', true_input_mean)
            print('true_input_std:', true_input_std)
            TrainingDatasets_mean, TrainingDatasets_std = Init_Transformer_Datasets.TrainingDatasets_mean, Init_Transformer_Datasets.TrainingDatasets_std
            sites_lat, sites_lon = Init_Transformer_Datasets.sites_lat, Init_Transformer_Datasets.sites_lon

            del Init_Transformer_Datasets
            gc.collect()
        
        for imodel in range(len(Estimation_trained_begin_dates)):
            if Apply_CNN_architecture:
                model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, 
                                                 begindates=Estimation_trained_begin_dates[imodel], enddates=Estimation_trained_end_dates[imodel], 
                                                 nchannel=len(total_channel_names), version=version,species=species, special_name=description, ifold=0, 
                                                 width=width, height=height,)
            elif Apply_3D_CNN_architecture:
                model = load_daily_datesbased_model(evaluation_type=Evaluation_type, typeName=typeName, 
                                                 begindates=Estimation_trained_begin_dates[imodel], enddates=Estimation_trained_end_dates[imodel], 
                                                 nchannel=len(total_channel_names), version=version,species=species, special_name=description, ifold=0, 
                                                 depth=depth, width=width, height=height,)
            for irange in range(len(Estimation_begindates[imodel])):
                predict_begindate = Estimation_begindates[imodel][irange]
                predict_endate = Estimation_enddates[imodel][irange]
                print(f'Starting the map estimation from {predict_begindate} to {predict_endate} ...')
                ## Create the output recording directory
                if Apply_CNN_architecture:
                    if world_size >1:
                        mp.spawn(cnn_mapdata_predict_func,
                                 args=(world_size,model,predict_begindate,predict_endate,total_channel_names,
                                       Evaluation_type, typeName,Estimation_Area,Extent,
                                       TrainingDatasets_mean, TrainingDatasets_std, true_input_mean,true_input_std,width,height,),
                                 nprocs=world_size,
                                 join=True)
                    elif world_size <=1:
                        cnn_mapdata_predict_func(rank=0, world_size=world_size, model=model,
                                                   predict_begindate=predict_begindate,predict_endate=predict_endate,
                                                   total_channel_names=total_channel_names,
                                                   Evaluation_type=Evaluation_type, typeName=typeName,Area=Estimation_Area,Extent=Extent,
                                                   train_mean=TrainingDatasets_mean, train_std=TrainingDatasets_std,true_mean=true_input_mean,true_std=true_input_std,
                                                   width=width,height=height)
                elif Apply_3D_CNN_architecture:
                    if world_size >1:
                        mp.spawn(cnn3D_mapdata_predict_func,
                                 args=(world_size,model,predict_begindate,predict_endate,total_channel_names,
                                       Evaluation_type, typeName,Estimation_Area,Extent,
                                       TrainingDatasets_mean, TrainingDatasets_std,true_input_mean,true_input_std, width,height,depth,),
                                 nprocs=world_size,
                                 join=True)
                    elif world_size <=1:
                        cnn3D_mapdata_predict_func(rank=0, world_size=world_size, model=model,
                                                   predict_begindate=predict_begindate,predict_endate=predict_endate,
                                                   total_channel_names=total_channel_names,
                                                   Evaluation_type=Evaluation_type, typeName=typeName,Area=Estimation_Area,Extent=Extent,
                                                   train_mean=TrainingDatasets_mean, train_std=TrainingDatasets_std,true_mean=true_input_mean,true_std=true_input_std,
                                                   width=width,height=height,depth=depth,)

                

    return