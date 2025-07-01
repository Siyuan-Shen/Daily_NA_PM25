import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.utils.data import DataLoader
from Training_pkg.utils import *
from Training_pkg.Statistic_func import linear_regression
from Training_pkg.TensorData_func import Dataset,Dataset_Val
from Training_pkg.Loss_func import SelfDesigned_LossFunction
from Training_pkg.iostream import save_daily_datesbased_model
import torch.nn.functional as F
from Model_Structure_pkg.CNN_Module import initial_cnn_network
from Model_Structure_pkg.ResCNN3D_Module import initial_3dcnn_net
from Model_Structure_pkg.utils import *
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from wandb_config import *
def ddp_setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def CNN3D_train(rank,world_size,temp_sweep_config,init_total_channel_names,X_train, y_train,X_test,y_test,input_mean, input_std,width,height,depth,
              evaluation_type,typeName,begindates,enddates,ifold=0):


    try:
        print(f"[Rank {rank}] Starting CNN_train")
        # Your original CNN_train logic goes here...
    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    print(f"Rank {rank} process started.")

    run_config = wandb_run_config()
    if rank == 0 and ifold == 0:
        wandb_initialize(run_config,rank)
    if temp_sweep_config is not None:
        wandb_config = temp_sweep_config
    else:
        wandb_config = run_config

    BATCH_SIZE, learning_rate, TOTAL_EPOCHS = wandb_parameters_return(wandb_config=wandb_config)
    try:
        channels_to_exclude = wandb_config.get("channel_to_exclude", [])
    except AttributeError:
        channels_to_exclude = []
    total_channel_names, main_stream_channel_names, side_stream_channel_names = Get_channel_names(channels_to_exclude=channels_to_exclude)


    index_of_main_stream_channels_of_initial = [init_total_channel_names.index(channel) for channel in main_stream_channel_names]

    X_train = X_train[:,index_of_main_stream_channels_of_initial,:,:,:]
    X_test  = X_test[:,index_of_main_stream_channels_of_initial,:,:,:]


    print('X_train shape: ',X_train.shape)
    print('y_train shape: ',y_train.shape)
    print('X_test shape: ',X_test.shape)
    print('y_test shape: ',y_test.shape)

    if rank != 0:
        os.environ['WANDB_MODE'] = 'disabled'
    
    if world_size == 1:
        Daily_Model = initial_3dcnn_net(main_stream_nchannel=len(main_stream_channel_names))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Daily_Model.to(device)
        torch.manual_seed(21)
        train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(Dataset(X_test, y_test), 2000, shuffle=True)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        Daily_Model = initial_3dcnn_net(main_stream_nchannel=len(main_stream_channel_names))
        device = rank
        Daily_Model.to(device)
        torch.manual_seed(21 + rank)
        Daily_Model = DDP(Daily_Model, device_ids=[device])
        train_dataset = Dataset(X_train, y_train)
        validation_dataset = Dataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False,sampler=DistributedSampler(train_dataset))
        validation_loader = DataLoader(validation_dataset, 2000, shuffle=False,sampler=DistributedSampler(validation_dataset))
    

    print('*' * 25, type(train_loader), '*' * 25)
    losses = []
    valid_losses = []
    train_acc = []
    test_acc  = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = SelfDesigned_LossFunction(losstype=Regression_loss_type)
    optimizer = optimizer_lookup(model_parameters=Daily_Model.parameters(), learning_rate=learning_rate)
    scheduler = lr_strategy_lookup_table(optimizer=optimizer)

    GeoSpecies_index = channel_names.index('tSATPM25')

    for epoch in range(TOTAL_EPOCHS):
        correct = 0
        counts = 0
        temp_losses = []
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        else:
            train_loader.sampler = None

        for i, (images, labels) in enumerate(train_loader):
            
            Daily_Model.train()
            images = images.to(device)
            labels = torch.squeeze(labels.type(torch.FloatTensor))
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = Daily_Model(images)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, labels, images[:,GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)])
            loss.backward()  ## backward
            optimizer.step()
            temp_losses.append(loss.item())

            # Calculate R2
            y_hat = outputs.cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()
            R2 = linear_regression(y_hat, y_true)
            R2 = np.round(R2, 4)
            correct += R2
            counts  += 1
            if (i + 1) % 10 == 0 and rank == 0 :
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train) // BATCH_SIZE,
                                                                loss.item()))
        losses.append(np.mean(temp_losses))
        valid_correct = 0
        valid_counts  = 0
        temp_losses = []
        scheduler.step()

        for i, (valid_images, valid_labels) in enumerate(validation_loader):
            Daily_Model.eval()
            valid_images = valid_images.to(device)
            valid_labels = valid_labels.to(device)
            valid_output = Daily_Model(valid_images)
            valid_output = torch.squeeze(valid_output)
            valid_loss   = criterion(valid_output, valid_labels, valid_images[:,GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)])
            temp_losses.append(valid_loss.item())
            test_y_hat   = valid_output.cpu().detach().numpy()
            test_y_true  = valid_labels.cpu().detach().numpy()
            Valid_R2 = linear_regression(test_y_hat, test_y_true)
            Valid_R2 = np.round(Valid_R2, 4)
            valid_correct += Valid_R2
            valid_counts  += 1    
            if rank == 0:  # Only print from the main process
                print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train) // BATCH_SIZE,
                                                                valid_loss.item(), Valid_R2))
        valid_losses.append(np.mean(temp_losses))
        accuracy = correct / counts
        test_accuracy = valid_correct / valid_counts
        print('Epoch: ',epoch, ', Training Loss: ', loss.item(),', Training accuracy:',accuracy, ', \nTesting Loss:', valid_loss.item(),', Testing accuracy:', test_accuracy)
        if wandb.run is not None and rank == 0 and ifold == 0:  # Log only from the main process
            wandb.log({
                'epoch': epoch,
                'learning_rates': optimizer.param_groups[0]['lr'],
                'train_loss': losses[-1],
                'valid_loss': valid_losses[-1],
                'train_accuracy': accuracy,
                'valid_accuracy': test_accuracy
            })
        train_acc.append(accuracy)
        test_acc.append(test_accuracy)
        print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])
    raw_model = Daily_Model.module if world_size > 1 else Daily_Model  # Get the underlying model if using DDP
    if world_size > 1:
        destroy_process_group()
    save_daily_datesbased_model(model=raw_model,evaluation_type=evaluation_type,typeName=typeName,
                                                begindates=begindates,enddates=enddates,
                                                version=version,species=species,nchannel=len(main_stream_channel_names),width=width,height=height,depth=depth,
                                                special_name=description,ifold=ifold)


def CNN_train(rank,world_size,temp_sweep_config,init_total_channel_names,X_train, y_train,X_test,y_test,input_mean, input_std,width,height,
              evaluation_type,typeName,begindates,enddates,ifold=0):
    try:
        print(f"[Rank {rank}] Starting CNN_train")
        # Your original CNN_train logic goes here...
    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    print(f"Rank {rank} process started.")

    run_config = wandb_run_config()
    if rank == 0 and ifold == 0:
        wandb_initialize(run_config,rank)

    if temp_sweep_config is not None:
        wandb_config = temp_sweep_config
    else:
        wandb_config = run_config
    
    BATCH_SIZE, learning_rate, TOTAL_EPOCHS = wandb_parameters_return(wandb_config=wandb_config)
    try:
        channels_to_exclude = wandb_config.get("channel_to_exclude", [])
    except AttributeError:
        channels_to_exclude = []

    total_channel_names, main_stream_channel_names, side_stream_channel_names = Get_channel_names(channels_to_exclude=channels_to_exclude)


    index_of_main_stream_channels_of_initial = [init_total_channel_names.index(channel) for channel in main_stream_channel_names]

    X_train = X_train[:,index_of_main_stream_channels_of_initial,:,:]
    X_test  = X_test[:,index_of_main_stream_channels_of_initial,:,:]

    print('X_train shape: ',X_train.shape)
    print('y_train shape: ',y_train.shape)
    print('X_test shape: ',X_test.shape)
    print('y_test shape: ',y_test.shape)

    if rank != 0:
        os.environ['WANDB_MODE'] = 'disabled'

    if world_size == 1:
        Daily_Model = initial_cnn_network(width=width, main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_channel_names))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Daily_Model.to(device)
        torch.manual_seed(21)
        train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(Dataset(X_test, y_test), 2000, shuffle=True)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        Daily_Model = initial_cnn_network(width=width, main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_channel_names))
        device = rank
        Daily_Model.to(device)
        torch.manual_seed(21 + rank)
        Daily_Model = DDP(Daily_Model, device_ids=[device])
        train_dataset = Dataset(X_train, y_train)
        validation_dataset = Dataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False,sampler=DistributedSampler(train_dataset))
        validation_loader = DataLoader(validation_dataset, 2000, shuffle=False,sampler=DistributedSampler(validation_dataset))
    

    
    print('*' * 25, type(train_loader), '*' * 25)
    losses = []
    valid_losses = []
    train_acc = []
    test_acc  = []

    
    criterion = SelfDesigned_LossFunction(losstype=Regression_loss_type)
    #optimizer = torch.optim.Adam(params=model.parameters(),betas=(), lr=learning_rate)
    optimizer = optimizer_lookup(model_parameters=Daily_Model.parameters(),learning_rate=learning_rate)
    scheduler = lr_strategy_lookup_table(optimizer=optimizer)

    GeoSpecies_index = total_channel_names.index('tSATPM25')

    for epoch in range(TOTAL_EPOCHS):
        correct = 0
        counts = 0
        temp_losses = []
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        else:
            train_loader.sampler = None
        for i, (images, labels) in enumerate(train_loader):
            
            Daily_Model.train()
            images = images.to(device)
            labels = torch.squeeze(labels.type(torch.FloatTensor))
            labels = labels.to(device)
            optimizer.zero_grad()  # Set grads to zero
            outputs = Daily_Model(images) #dimension: Nx1
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, labels, images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
            loss.backward()  ## backward
            optimizer.step()  ## refresh training parameters
            temp_losses.append(loss.item())

            # Calculate R2
            y_hat = outputs.cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()

            
            #torch.cuda.empty_cache()
            #print('y_hat:', y_hat)
            R2 = linear_regression(y_hat,y_true)
            R2 = np.round(R2, 4)
            #pred = y_hat.max(1, keepdim=True)[1] # 得到最大值及索引，a.max[0]为最大值，a.max[1]为最大值的索引
            correct += R2
            counts  += 1
            if (i + 1) % 100 == 0 and rank == 0:  # Only print from the main process
            # 每100个batches打印一次loss
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train) // BATCH_SIZE,
                                                                loss.item()))
        losses.append(np.mean(temp_losses))
        valid_correct = 0
        valid_counts  = 0
        scheduler.step() 
        temp_losses = []
        for i, (valid_images, valid_labels) in enumerate(validation_loader):
            Daily_Model.eval()
            valid_images = valid_images.to(device)
            valid_labels = valid_labels.to(device)
            valid_output = Daily_Model(valid_images)
            valid_output = torch.squeeze(valid_output)
            valid_loss   = criterion(valid_output, valid_labels, valid_images[:,GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,int((width-1)/2),int((height-1)/2)])
            temp_losses.append(valid_loss.item())
            test_y_hat   = valid_output.cpu().detach().numpy()
            test_y_true  = valid_labels.cpu().detach().numpy()
            #print('test_y_hat size: {}'.format(test_y_hat.shape),'test_y_true size: {}'.format(test_y_true.shape))
            Valid_R2 = linear_regression(test_y_hat,test_y_true)
            Valid_R2 = np.round(Valid_R2, 4)
            valid_correct += Valid_R2
            valid_counts  += 1    
            if rank == 0:  # Only print from the main process
                print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train) // BATCH_SIZE,
                                                                valid_loss.item(), Valid_R2)) 
        valid_losses.append(np.mean(temp_losses))
        accuracy = correct / counts
        test_accuracy = valid_correct / valid_counts
        print('Epoch: ',epoch, ', Training Loss: ', loss.item(),', Training accuracy:',accuracy, ', \nTesting Loss:', valid_loss.item(),', Testing accuracy:', test_accuracy)

        if wandb.run is not None and rank == 0 and ifold == 0:  # Log only from the main process
            wandb.log({
                'epoch': epoch,
                'learning_rates': optimizer.param_groups[0]['lr'],
                'train_loss': losses[-1],
                'valid_loss': valid_losses[-1],
                'train_accuracy': accuracy,
                'valid_accuracy': test_accuracy
            })

        train_acc.append(accuracy)
        test_acc.append(test_accuracy)
        print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])

    raw_model = Daily_Model.module if world_size > 1 else Daily_Model  # Get the underlying model if using DDP
    if world_size > 1:
        destroy_process_group()
    save_daily_datesbased_model(model=raw_model,evaluation_type=evaluation_type,typeName=typeName,
                                                begindates=begindates,enddates=enddates,
                                                version=version,species=species,nchannel=len(main_stream_channel_names),width=width,height=height,
                                                special_name=description,ifold=ifold)

def cnn_predict_3D(inputarray, model, batchsize,initial_channel_names,mainstream_channel_names,sidestream_channel_names):
    """
    This function is used to predict the PM2.5 concentration using a 3D CNN model.
    
    Args:
        inputarray (numpy.ndarray): The input data for prediction.
        model (torch.nn.Module): The trained CNN model.
        batchsize (int): The batch size for prediction.
        initial_channel_names (list): List of initial channel names.
        mainstream_channel_names (list): List of mainstream channel names.
        sidestream_channel_names (list): List of sidestream channel names.
        
    Returns:
        numpy.ndarray: The predicted PM2.5 concentrations.
    """
    model.eval()
    final_output = []
    final_output = np.array(final_output)
    predictinput = DataLoader(Dataset_Val(inputarray), batch_size= batchsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for i, image in enumerate(predictinput):
            image = image.to(device)
            output = model(image).cpu().detach().numpy()
            final_output = np.append(final_output,output)
    
    return final_output

def cnn_predict(inputarray, model, batchsize,initial_channel_names,mainstream_channel_names,sidestream_channel_names):

    model.eval()
    final_output = []
    final_output = np.array(final_output)
    predictinput = DataLoader(Dataset_Val(inputarray), batch_size= batchsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ResNet_Settings or ResNet_MLP_Settings:
        with torch.no_grad():
            for i, image in enumerate(predictinput):
                image = image.to(device)
                output = model(image).cpu().detach().numpy()
                final_output = np.append(final_output,output)
    return final_output