import os
import csv
import torch 
import numpy as np
import netCDF4 as nc
import wandb

from Model_Structure_pkg.utils import *
from Training_pkg.utils import *
from Evaluation_pkg.utils import csv_outdir,model_outdir,data_recording_outdir, HSV_Apply_wandb_sweep_Switch,Hyperparameters_Search_Validation_Switch

def get_data_recording_filenname(outdir,evaluation_type, file_target,typeName,begindate,enddate, nchannel, **args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)

    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'    
        if HSV_Apply_wandb_sweep_Switch and Hyperparameters_Search_Validation_Switch:
            api = wandb.Api()
            sweep = api.sweep(f"/{wandb.run.entity}/{wandb.run.project}/{wandb.run.sweep_id}")
            outdir = outdir + 'sweep-{}/'.format(sweep.name)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            outfile = outdir + '{}_{}_{}_{}_{}_{}x{}_{}-{}_{}Channel_sweep-{}.npy'.format(Model_structure_type, evaluation_type,file_target,typeName, species, width, height, begindate,enddate,nchannel,wandb.run.name)
        outfile = outdir + '{}_{}_{}_{}_{}_{}x{}_{}-{}_{}Channel{}.npy'.format(Model_structure_type, evaluation_type,file_target,typeName, species, width, height, begindate,enddate,nchannel,description)
    elif Apply_3D_CNN_architecture:
        Model_structure_type = 'CNN3DModel'
        if HSV_Apply_wandb_sweep_Switch and Hyperparameters_Search_Validation_Switch:
            api = wandb.Api()
            sweep = api.sweep(f"/{wandb.run.entity}/{wandb.run.project}/{wandb.run.sweep_id}")
            outdir = outdir + 'sweep-{}/'.format(sweep.name)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            outfile = outdir + '{}_{}_{}_{}_{}_{}x{}x{}_{}-{}_{}Channel_sweep-{}.npy'.format(Model_structure_type, evaluation_type,file_target,typeName, species, depth,width, height, begindate,enddate,nchannel,wandb.run.name)
    return outfile

def save_loss_accuracy_recording(loss,accuracy,valid_loss,valid_accuracy,species,version,evaluation_type,begindate,enddate,typeName,nchannel,width=11,height=11,):
    outdir = data_recording_outdir + '{}/{}/Results/results-LossAccuracy/{}/'.format(species, version,evaluation_type)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    loss_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='loss',typeName=typeName,nchannel=nchannel,width=width,height=height,begindate=begindate,enddate=enddate)
    accuracy_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='accuracy',typeName=typeName,nchannel=nchannel,width=width,height=height,begindate=begindate,enddate=enddate)
    valid_loss_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='valid_loss',typeName=typeName,nchannel=nchannel,width=width,height=height,begindate=begindate,enddate=enddate)
    valid_accuracy_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='valid_accuracy',typeName=typeName,nchannel=nchannel,width=width,height=height,begindate=begindate,enddate=enddate)
    
    np.save(loss_outfile, loss)
    np.save(accuracy_outfile, accuracy)
    np.save(valid_loss_outfile, valid_loss)
    np.save(valid_accuracy_outfile, valid_accuracy)
    return

def load_loss_accuracy_recording(species,version,evaluation_type,typeName,begindate,enddate,nchannel,width=11,height=11,):
    indir = data_recording_outdir + '{}/{}/Results/results-LossAccuracy/{}/'.format(species, version,evaluation_type)
    if not os.path.isdir(indir):
        raise ValueError('The {} directory does not exist!'.format(indir))
    loss_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='loss',typeName=typeName,nchannel=nchannel,width=width,height=height,begindate=begindate,enddate=enddate)
    accuracy_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='accuracy',typeName=typeName,nchannel=nchannel,width=width,height=height,begindate=begindate,enddate=enddate)
    valid_loss_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='valid_loss',typeName=typeName,nchannel=nchannel,width=width,height=height,begindate=begindate,enddate=enddate)
    valid_accuracy_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='valid_accuracy',typeName=typeName,nchannel=nchannel,width=width,height=height,begindate=begindate,enddate=enddate)

    if not os.path.isfile(loss_infile):
        raise ValueError('The {} file does not exist!'.format(loss_infile))
    if not os.path.isfile(accuracy_infile):
        raise ValueError('The {} file does not exist!'.format(accuracy_infile))
    if not os.path.isfile(valid_loss_infile):
        raise ValueError('The {} file does not exist!'.format(valid_loss_infile))
    if not os.path.isfile(valid_accuracy_infile):
        raise ValueError('The {} file does not exist!'.format(valid_accuracy_infile))
    loss = np.load(loss_infile)
    accuracy = np.load(accuracy_infile)
    valid_loss = np.load(valid_loss_infile)
    valid_accuracy = np.load(valid_accuracy_infile)
    return loss, accuracy, valid_loss, valid_accuracy



def output_csv(outfile:str,status:str,Area,test_begindate,test_enddate,Daily_statistics_recording,Monthly_statistics_recording,Annual_statistics_recording):
    MONTH = ['Annual','01','02','03','04','05','06','07','08','09','10','11','12']
    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        # first write the head
        writer.writerow(['Time Range', 'Area', 'Evaluation Type', 
                         'Test R2 - Mean', 'Test geo R2 - Mean','Train R2 - Mean',
                         'Test RMSE - Mean','Test NRMSE - Mean','Test slope - Mean',  
                         'Test R2 - Max', 'Test geo R2 - Max','Train R2 - Max', 
                         'Test RMSE - Max', 'Test NRMSE - Max','Test slope - Max',
                         'Test R2 - Min', 'Test geo R2 - Min', 'Train R2 - Min',
                         'Test RMSE - Min', 'Test NRMSE - Min','Test slope - Min',
                         'Test R2 - Std', 'Test geo R2 - Std', 'Train R2 - Std',
                         'Test RMSE - Std', 'Test NRMSE - Std','Test slope - Std'])
        ## Write the All points for Daily
       
        writer.writerow(['{}-{}'.format(test_begindate,test_enddate), Area ,'Daily - Allpoints',
                        str(np.round(Daily_statistics_recording['All_points']['test_R2'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['geo_R2'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['train_R2'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['RMSE'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['NRMSE'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['slope'], 4)),

                        str(np.round(Daily_statistics_recording['All_points']['test_R2'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['geo_R2'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['train_R2'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['RMSE'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['NRMSE'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['slope'], 4)),
                        
                        str(np.round(Daily_statistics_recording['All_points']['test_R2'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['geo_R2'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['train_R2'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['RMSE'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['NRMSE'], 4)),
                        str(np.round(Daily_statistics_recording['All_points']['slope'], 4)),

                        0.0, 0.0, 0.0, 0.0,0.0,0.0])
        ## Write the Purely Spatial for Daily 
        

        writer.writerow(['{}-{}'.format(test_begindate,test_enddate),  Area ,'Daily - Purely Spatial',
                        
                        str(np.round(np.mean(Daily_statistics_recording['Purely_Spatial']['test_R2']), 4)),
                        str(np.round(np.mean(Daily_statistics_recording['Purely_Spatial']['geo_R2']), 4)),
                        str(np.round(np.mean(Daily_statistics_recording['Purely_Spatial']['train_R2']), 4)),
                        str(np.round(np.mean(Daily_statistics_recording['Purely_Spatial']['RMSE']), 4)),
                        str(np.round(np.mean(Daily_statistics_recording['Purely_Spatial']['NRMSE']), 4)),
                        str(np.round(np.mean(Daily_statistics_recording['Purely_Spatial']['slope']), 4)),


                        str(np.round(np.max(Daily_statistics_recording['Purely_Spatial']['test_R2']), 4)),
                        str(np.round(np.max(Daily_statistics_recording['Purely_Spatial']['geo_R2']), 4)),
                        str(np.round(np.max(Daily_statistics_recording['Purely_Spatial']['train_R2']), 4)),
                        str(np.round(np.max(Daily_statistics_recording['Purely_Spatial']['RMSE']), 4)),
                        str(np.round(np.max(Daily_statistics_recording['Purely_Spatial']['NRMSE']), 4)),
                        str(np.round(np.max(Daily_statistics_recording['Purely_Spatial']['slope']), 4)),

                        str(np.round(np.min(Daily_statistics_recording['Purely_Spatial']['test_R2']), 4)),
                        str(np.round(np.min(Daily_statistics_recording['Purely_Spatial']['geo_R2']), 4)),
                        str(np.round(np.min(Daily_statistics_recording['Purely_Spatial']['train_R2']), 4)),
                        str(np.round(np.min(Daily_statistics_recording['Purely_Spatial']['RMSE']), 4)),
                        str(np.round(np.min(Daily_statistics_recording['Purely_Spatial']['NRMSE']), 4)),
                        str(np.round(np.min(Daily_statistics_recording['Purely_Spatial']['slope']), 4)),

                        str(np.round(np.std(Daily_statistics_recording['Purely_Spatial']['test_R2']), 4)),
                        str(np.round(np.std(Daily_statistics_recording['Purely_Spatial']['geo_R2']), 4)),
                        str(np.round(np.std(Daily_statistics_recording['Purely_Spatial']['train_R2']), 4)),
                        str(np.round(np.std(Daily_statistics_recording['Purely_Spatial']['RMSE']), 4)),
                        str(np.round(np.std(Daily_statistics_recording['Purely_Spatial']['NRMSE']), 4)),
                        str(np.round(np.std(Daily_statistics_recording['Purely_Spatial']['slope']), 4)),
                        ])
        
        ## Write the Monthly Sacle for Daily data

        for imonth in MONTH[1:]:
            writer.writerow(['{}-{}'.format(test_begindate,test_enddate), Area , 'Daily - Monthly Scale - {}'.format(imonth),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['test_R2'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['geo_R2'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['train_R2'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['RMSE'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['NRMSE'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['slope'], 4)),

                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['test_R2'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['geo_R2'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['train_R2'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['RMSE'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['NRMSE'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['slope'], 4)),

                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['test_R2'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['geo_R2'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['train_R2'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['RMSE'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['NRMSE'], 4)),
                            str(np.round(Daily_statistics_recording['Monthly_Scale'][imonth]['slope'], 4)),

                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])
        ## Write the Monthly All Points Statistics

        for imonth in MONTH[1:]:
            writer.writerow(['{}-{}'.format(test_begindate,test_enddate), Area , 'Monthly - Allpoints - {}'.format(imonth),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['test_R2'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['geo_R2'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['train_R2'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['RMSE'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['NRMSE'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['slope'], 4)),

                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['test_R2'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['geo_R2'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['train_R2'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['RMSE'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['NRMSE'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['slope'], 4)),

                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['test_R2'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['geo_R2'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['train_R2'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['RMSE'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['NRMSE'], 4)),
                            str(np.round(Monthly_statistics_recording['All_points'][imonth]['slope'], 4)),

                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])
        
        ## Write the Monthly Purely Spatial Statistics
        for imonth in MONTH[1:]:
            writer.writerow(['{}-{}'.format(test_begindate,test_enddate), Area , 'Monthly - Purely Spatial - {}'.format(imonth),
                                str(np.round(np.mean(Monthly_statistics_recording['Purely_Spatial'][imonth]['test_R2']), 4)),
                                str(np.round(np.mean(Monthly_statistics_recording['Purely_Spatial'][imonth]['geo_R2']), 4)),
                                str(np.round(np.mean(Monthly_statistics_recording['Purely_Spatial'][imonth]['train_R2']), 4)),
                                str(np.round(np.mean(Monthly_statistics_recording['Purely_Spatial'][imonth]['RMSE']), 4)),
                                str(np.round(np.mean(Monthly_statistics_recording['Purely_Spatial'][imonth]['NRMSE']), 4)),
                                str(np.round(np.mean(Monthly_statistics_recording['Purely_Spatial'][imonth]['slope']), 4)),

                                str(np.round(np.max(Monthly_statistics_recording['Purely_Spatial'][imonth]['test_R2']), 4)),
                                str(np.round(np.max(Monthly_statistics_recording['Purely_Spatial'][imonth]['geo_R2']), 4)),
                                str(np.round(np.max(Monthly_statistics_recording['Purely_Spatial'][imonth]['train_R2']), 4)),
                                str(np.round(np.max(Monthly_statistics_recording['Purely_Spatial'][imonth]['RMSE']), 4)),
                                str(np.round(np.max(Monthly_statistics_recording['Purely_Spatial'][imonth]['NRMSE']), 4)),
                                str(np.round(np.max(Monthly_statistics_recording['Purely_Spatial'][imonth]['slope']), 4)),

                                str(np.round(np.min(Monthly_statistics_recording['Purely_Spatial'][imonth]['test_R2']), 4)),
                                str(np.round(np.min(Monthly_statistics_recording['Purely_Spatial'][imonth]['geo_R2']), 4)),
                                str(np.round(np.min(Monthly_statistics_recording['Purely_Spatial'][imonth]['train_R2']), 4)),
                                str(np.round(np.min(Monthly_statistics_recording['Purely_Spatial'][imonth]['RMSE']), 4)),
                                str(np.round(np.min(Monthly_statistics_recording['Purely_Spatial'][imonth]['NRMSE']), 4)),
                                str(np.round(np.min(Monthly_statistics_recording['Purely_Spatial'][imonth]['slope']), 4)),

                                str(np.round(np.std(Monthly_statistics_recording['Purely_Spatial'][imonth]['test_R2']), 4)),
                                str(np.round(np.std(Monthly_statistics_recording['Purely_Spatial'][imonth]['geo_R2']), 4)),
                                str(np.round(np.std(Monthly_statistics_recording['Purely_Spatial'][imonth]['train_R2']), 4)),
                                str(np.round(np.std(Monthly_statistics_recording['Purely_Spatial'][imonth]['RMSE']), 4)),
                                str(np.round(np.std(Monthly_statistics_recording['Purely_Spatial'][imonth]['NRMSE']), 4)),
                                str(np.round(np.std(Monthly_statistics_recording['Purely_Spatial'][imonth]['slope']), 4)),

                                ])
        
        ## Write the Annual All Points Statistics
        writer.writerow(['{}-{}'.format(test_begindate,test_enddate), Area , 'Annual - Allpoints',
                        str(np.round(Annual_statistics_recording['All_points']['test_R2'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['geo_R2'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['train_R2'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['RMSE'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['NRMSE'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['slope'], 4)),

                        str(np.round(Annual_statistics_recording['All_points']['test_R2'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['geo_R2'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['train_R2'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['RMSE'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['NRMSE'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['slope'], 4)),
 
                        str(np.round(Annual_statistics_recording['All_points']['test_R2'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['geo_R2'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['train_R2'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['RMSE'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['NRMSE'], 4)),
                        str(np.round(Annual_statistics_recording['All_points']['slope'], 4)),

                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])
        ## Write the Annual Purely Spatial Statistics
        writer.writerow(['{}-{}'.format(test_begindate,test_enddate), Area , 'Annual - Purely Spatial',
                        str(np.round(np.mean(Annual_statistics_recording['Purely_Spatial']['test_R2']), 4)),
                        str(np.round(np.mean(Annual_statistics_recording['Purely_Spatial']['geo_R2']), 4)),
                        str(np.round(np.mean(Annual_statistics_recording['Purely_Spatial']['train_R2']), 4)),
                        str(np.round(np.mean(Annual_statistics_recording['Purely_Spatial']['RMSE']), 4)),
                        str(np.round(np.mean(Annual_statistics_recording['Purely_Spatial']['NRMSE']), 4)),
                        str(np.round(np.mean(Annual_statistics_recording['Purely_Spatial']['slope']), 4)),


                        str(np.round(np.max(Annual_statistics_recording['Purely_Spatial']['test_R2']), 4)),
                        str(np.round(np.max(Annual_statistics_recording['Purely_Spatial']['geo_R2']), 4)),
                        str(np.round(np.max(Annual_statistics_recording['Purely_Spatial']['train_R2']), 4)),
                        str(np.round(np.max(Annual_statistics_recording['Purely_Spatial']['RMSE']), 4)),
                        str(np.round(np.max(Annual_statistics_recording['Purely_Spatial']['NRMSE']), 4)),
                        str(np.round(np.max(Annual_statistics_recording['Purely_Spatial']['slope']), 4)),


                        str(np.round(np.min(Annual_statistics_recording['Purely_Spatial']['test_R2']), 4)),
                        str(np.round(np.min(Annual_statistics_recording['Purely_Spatial']['geo_R2']), 4)),
                        str(np.round(np.min(Annual_statistics_recording['Purely_Spatial']['train_R2']), 4)),
                        str(np.round(np.min(Annual_statistics_recording['Purely_Spatial']['RMSE']), 4)),
                        str(np.round(np.min(Annual_statistics_recording['Purely_Spatial']['NRMSE']), 4)),
                        str(np.round(np.min(Annual_statistics_recording['Purely_Spatial']['slope']), 4)),


                        str(np.round(np.std(Annual_statistics_recording['Purely_Spatial']['test_R2']), 4)),
                        str(np.round(np.std(Annual_statistics_recording['Purely_Spatial']['geo_R2']), 4)),
                        str(np.round(np.std(Annual_statistics_recording['Purely_Spatial']['train_R2']), 4)),
                        str(np.round(np.std(Annual_statistics_recording['Purely_Spatial']['RMSE']), 4)),
                        str(np.round(np.std(Annual_statistics_recording['Purely_Spatial']['NRMSE']), 4)),
                        str(np.round(np.std(Annual_statistics_recording['Purely_Spatial']['slope']), 4)),
                        ])

    return
        
def save_data_recording(final_data_recording,obs_data_recording,geo_data_recording,sites_recording,dates_recording,
                        training_final_data_recording,training_obs_data_recording,training_sites_recording,training_dates_recording,
                        species, version, begindates,enddates, evaluation_type, typeName,nchannel,**args):
    """This is for saving the data recording files for the evaluation of the model.
    The hyperparameter search validation, spatial crossvalidation and temporal crossvalidation

    Args:
        final_data_recording (_type_): _description_
        obs_data_recording (_type_): _description_
        geo_data_recording (_type_): _description_
        sites_recording (_type_): _description_
        dates_recording (_type_): _description_
        training_final_data_recording (_type_): _description_
        training_obs_data_recording (_type_): _description_
        training_sites_recording (_type_): _description_
        training_dates_recording (_type_): _description_
        species (_type_): _description_
        version (_type_): _description_
        begindates (_type_): _description_
        enddates (_type_): _description_
        evaluation_type (_type_): _description_
        typeName (_type_): _description_
        width (_type_): _description_
        height (_type_): _description_
        nchannel (_type_): _description_
        special_name (_type_): _description_
    """
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)

    outdir = csv_outdir + '{}/{}/Results/results-DataRecording/{}/'.format(species, version,evaluation_type)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    obs_data_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='ObsDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    final_data_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='FinalDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    geo_data_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='GeoDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    sites_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='SitesRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    dates_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='DatesRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    training_obs_data_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='TrainingObsDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    training_final_data_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='TrainingFinalDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    training_sites_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='TrainingSitesRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    training_dates_outfile = get_data_recording_filenname(outdir=outdir,evaluation_type=evaluation_type,file_target='TrainingDatesRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)

    np.save(obs_data_outfile, obs_data_recording.data)
    np.save(final_data_outfile, final_data_recording.data)
    np.save(geo_data_outfile, geo_data_recording.data)
    np.save(sites_outfile, sites_recording.data)
    np.save(dates_outfile, dates_recording.data)
    np.save(training_obs_data_outfile, training_obs_data_recording.data)
    np.save(training_final_data_outfile, training_final_data_recording.data)
    np.save(training_sites_outfile, training_sites_recording.data)
    np.save(training_dates_outfile, training_dates_recording.data)
    print('Data recording files saved in: {}'.format(outdir))

    print('Obs data recording file: {}'.format(obs_data_outfile))
    print('Final data recording file: {}'.format(final_data_outfile))
    print('Geo data recording file: {}'.format(geo_data_outfile))
    print('Sites recording file: {}'.format(sites_outfile))
    print('Dates recording file: {}'.format(dates_outfile))
    print('Training Obs data recording file: {}'.format(training_obs_data_outfile))
    print('Training Final data recording file: {}'.format(training_final_data_outfile))
    print('Training Sites recording file: {}'.format(training_sites_outfile))
    print('Training Dates recording file: {}'.format(training_dates_outfile))
    return

def load_data_recording(species, version, begindates,enddates, evaluation_type, typeName, nchannel,special_name,**args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)
    indir = csv_outdir + '{}/{}/Results/results-DataRecording/{}/'.format(species, version,evaluation_type)
    if not os.path.isdir(indir):
        raise ValueError('The {} directory does not exist!'.format(indir))
    obs_data_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='ObsDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    final_data_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='FinalDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    geo_data_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='GeoDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    sites_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='SitesRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    dates_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='DatesRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    training_obs_data_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='TrainingObsDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    training_final_data_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='TrainingFinalDataRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    training_sites_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='TrainingSitesRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)
    training_dates_infile = get_data_recording_filenname(outdir=indir,evaluation_type=evaluation_type,file_target='TrainingDatesRecording',typeName=typeName,nchannel=nchannel,width=width,height=height,depth=depth,begindate=begindates,enddate=enddates)

    if not os.path.isfile(obs_data_infile):
        raise ValueError('The {} file does not exist!'.format(obs_data_infile))
    if not os.path.isfile(final_data_infile):
        raise ValueError('The {} file does not exist!'.format(final_data_infile))
    if not os.path.isfile(geo_data_infile):
        raise ValueError('The {} file does not exist!'.format(geo_data_infile))
    if not os.path.isfile(sites_infile):
        raise ValueError('The {} file does not exist!'.format(sites_infile))
    if not os.path.isfile(dates_infile):
        raise ValueError('The {} file does not exist!'.format(dates_infile))
    if not os.path.isfile(training_obs_data_infile):
        raise ValueError('The {} file does not exist!'.format(training_obs_data_infile))
    if not os.path.isfile(training_final_data_infile):
        raise ValueError('The {} file does not exist!'.format(training_final_data_infile))
    if not os.path.isfile(training_sites_infile):
        raise ValueError('The {} file does not exist!'.format(training_sites_infile))
    if not os.path.isfile(training_dates_infile):
        raise ValueError('The {} file does not exist!'.format(training_dates_infile))
    obs_data_recording = np.load(obs_data_infile)
    final_data_recording = np.load(final_data_infile)
    geo_data_recording = np.load(geo_data_infile)
    sites_recording = np.load(sites_infile)
    dates_recording = np.load(dates_infile)
    training_obs_data_recording = np.load(training_obs_data_infile)
    training_final_data_recording = np.load(training_final_data_infile)
    training_sites_recording = np.load(training_sites_infile)
    training_dates_recording = np.load(training_dates_infile)
    
    return final_data_recording, obs_data_recording, geo_data_recording, sites_recording, dates_recording, training_final_data_recording, training_obs_data_recording, training_sites_recording, training_dates_recording

def load_NA_Mask_data(region_name):
    NA_Mask_indir = '/my-projects/mask/NA_Masks/Cropped_NA_Masks/'
    try:
        dataset = nc.Dataset(NA_Mask_indir+'Cropped_PROVMASK-{}.nc'.format(region_name))
        mask_map = dataset.variables['provmask'][:]
        lat = dataset.variables['lat'][:]
        lon = dataset.variables['lon'][:]
    except:
        print('Not in PROV')
    try:
        dataset = nc.Dataset(NA_Mask_indir+'Cropped_REGIONMASK-{}.nc'.format(region_name))
        mask_map = dataset.variables['regionmask'][:]
        lat = dataset.variables['lat'][:]
        lon = dataset.variables['lon'][:]
    except:
        print('Not in Region')
    try:
        dataset = nc.Dataset(NA_Mask_indir+'Cropped_STATEMASK-{}.nc'.format(region_name))
        mask_map = dataset.variables['statemask'][:]
        lat = dataset.variables['lat'][:]
        lon = dataset.variables['lon'][:]
    except:
        print('Not in STATE')
    return mask_map, lat, lon

def save_daily_datesbased_model(model,evaluation_type, typeName, begindates,enddates, version, species, nchannel, special_name, ifold, **args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)

    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/{}/'.format(species, version,evaluation_type)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'
        model_outfile = outdir +  '{}_{}_{}_{}_{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, evaluation_type, typeName, species, width,height, begindates,enddates,nchannel,special_name, ifold)
        torch.save(model, model_outfile)
    elif Apply_3D_CNN_architecture:
        Model_structure_type = '3DCNNModel'
        model_outfile = outdir +  '{}_{}_{}_{}_{}x{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, evaluation_type, typeName, species, depth, width,height, begindates,enddates,nchannel,special_name, ifold)
        torch.save(model, model_outfile)
    return

def load_daily_datesbased_model(evaluation_type, typeName, begindates,enddates, version, species, nchannel, special_name, ifold,**args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)
    indir = model_outdir + '{}/{}/Results/results-Trained_Models/{}/'.format(species, version,evaluation_type)
    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'
        model_infile = indir + '{}_{}_{}_{}_{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, species, version,evaluation_type, evaluation_type, typeName, species, width,height, begindates,enddates,nchannel,special_name, ifold)
        
        if not os.path.isfile(model_infile):
            raise ValueError('The {} file does not exist!'.format(model_infile))
        
        model = torch.load(model_infile)
    elif Apply_3D_CNN_architecture:
        Model_structure_type = '3DCNNModel'
        model_infile = indir + '{}_{}_{}_{}_{}x{}x{}_{}-{}_{}Channel{}_No{}.pt'.format(Model_structure_type, species, version,evaluation_type, evaluation_type, typeName, species, depth, width,height, begindates,enddates,nchannel,special_name, ifold)
        
        if not os.path.isfile(model_infile):
            raise ValueError('The {} file does not exist!'.format(model_infile))
        
        model = torch.load(model_infile)
    return model