import torch
import torch.nn as nn
import wandb
import os 
import time
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from Training_pkg.utils import *
from Training_pkg.Statistic_func import linear_regression
from Training_pkg.TensorData_func import Dataset,Dataset_Val,CNN_Transformer_Dataset,CNN_Transformer_Dataset_Val
from Training_pkg.Loss_func import SelfDesigned_LossFunction
from Training_pkg.iostream import save_daily_datesbased_model
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
import torch.nn.functional as F
from Model_Structure_pkg.Transformer_Model.model.transformer import Transformer
from Model_Structure_pkg.CNN_Transformer_Model.model.CNN_transformer import CNN_Transformer
from Model_Structure_pkg.CNN_Module import initial_cnn_network
from Model_Structure_pkg.ResCNN3D_Module import initial_3dcnn_net
from Model_Structure_pkg.utils import *
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from wandb_config import *
from datetime import timedelta
from torch.profiler import profile, record_function, ProfilerActivity


def ddp_setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size,timeout=timedelta(hours=2))

def CNN_Transformer_train(rank, world_size, temp_sweep_config, sweep_mode, sweep_id, run_id_container,CNN_total_channel_names, Transformer_total_channel_names,
                           X_train_CNN, X_test_CNN,X_train_Transformer, X_test_Transformer,
                           y_train, y_test,input_mean_Transformer, input_std_Transformer, width,height,
                           evaluation_type,typeName,begindates, enddates, ifold=0):
    print('fold {} is starting...'.format(ifold))
    print('world_size: {}'.format(world_size))
    try:
        print(f"[Rank {rank}] Starting Transformer_train")
        # Your original CNN_train logic goes here...
    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    print(f"Rank {rank} process started.")

    run_config = wandb_run_config()
    if rank == 0 and ifold == 0:
        if sweep_mode:
            wandb_initialize(temp_sweep_config,rank,sweep_mode,sweep_id)
        else:
            wandb_initialize(run_config,rank,sweep_mode,sweep_id)
        run_id_container["run_id"] = wandb.run.id
        run_id_container["run_name"] = wandb.run.name
    if sweep_mode:
        wandb_config = temp_sweep_config
    else:
        wandb_config = run_config
    BATCH_SIZE, learning_rate, TOTAL_EPOCHS, CNN_blocks_num, CNN_output_channels, d_model, n_head, ffn_hidden, num_layers, max_len, spin_up_len, drop_prob = wandb_parameters_return(wandb_config=wandb_config)
    print('BATCH_SIZE: ', BATCH_SIZE, ' learning_rate: ', learning_rate, 'TOTAL_EPOCHS: ', TOTAL_EPOCHS,
            ' CNN_blocks_num: ', CNN_blocks_num, ' CNN_output_channels: ', CNN_output_channels,
            ' d_model: ', d_model, ' n_head: ', n_head, ' ffn_hidden: ', ffn_hidden, ' num_layers: ', num_layers,
            ' max_len: ', max_len, ' spin_up_len: ', spin_up_len, ' drop_prob: ', drop_prob)
    try:
        CNN_channels_to_exclude = wandb_config.get("CNN_channel_to_exclude", [])
        Transformer_channels_to_exclude = wandb_config.get("Transformer_channel_to_exclude", [])
    except AttributeError:
        CNN_channels_to_exclude = []
        Transformer_channels_to_exclude = []
    CNN_total_channel_names, CNN_main_stream_channel_names, CNN_side_stream_channel_names = Get_channel_names(channels_to_exclude=CNN_channels_to_exclude,init_channels=CNN_total_channel_names)
    Transformer_total_channel_names, Transformer_main_stream_channel_names, Transformer_side_stream_channel_names = Get_channel_names(channels_to_exclude=Transformer_channels_to_exclude,init_channels=Transformer_total_channel_names)

    CNN_index_of_main_stream_channels_of_initial = [CNN_total_channel_names.index(channel) for channel in CNN_main_stream_channel_names]
    Transformer_index_of_main_stream_channels_of_initial = [Transformer_total_channel_names.index(channel) for channel in Transformer_main_stream_channel_names]

    print('X_train_CNN:', X_train_CNN[0,:])
    X_train_CNN = X_train_CNN[:, :,CNN_index_of_main_stream_channels_of_initial]
    X_test_CNN = X_test_CNN[:, :,CNN_index_of_main_stream_channels_of_initial]
    X_train_Transformer = X_train_Transformer[:, :,Transformer_index_of_main_stream_channels_of_initial]
    X_test_Transformer = X_test_Transformer[:, :,Transformer_index_of_main_stream_channels_of_initial]

    print('CNN-Transformer Architecture X_train_CNN shape: ',X_train_CNN.shape)
    print('CNN-Transformer Architecture X_train_Transformer shape: ',X_train_Transformer.shape)
    print('CNN-Transformer Architecture y_train shape: ',y_train.shape)
    print('CNN-Transformer Architecture X_test_CNN shape: ',X_test_CNN.shape)
    print('CNN-Transformer Architecture X_test_Transformer shape: ',X_test_Transformer.shape)
    print('CNN-Transformer Architecture y_test shape: ',y_test.shape)

    if rank != 0:
        os.environ['WANDB_MODE'] = 'disabled'
    
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Daily_Model = CNN_Transformer(CNN_input_dim=X_train_CNN.shape[2], transformer_input_dim=X_train_Transformer.shape[2], trg_dim=CNN_Transformer_trg_dim, d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, num_layers=num_layers, max_len=max_len+spin_up_len, drop_prob=drop_prob,device=device)
        Daily_Model.to(device)
        torch.manual_seed(21)
        train_loader = DataLoader(CNN_Transformer_Dataset(X_train_CNN,X_train_Transformer, y_train), BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(CNN_Transformer_Dataset(X_test_CNN,X_test_Transformer,y_test), 100, shuffle=False)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        device = rank
        Daily_Model = CNN_Transformer(CNN_input_dim=X_train_CNN.shape[2], transformer_input_dim=X_train_Transformer.shape[2], trg_dim=CNN_Transformer_trg_dim, d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, num_layers=num_layers, max_len=max_len+spin_up_len, drop_prob=drop_prob,device=device)
        Daily_Model.to(device)
        torch.manual_seed(21)
        Daily_Model = DDP(Daily_Model, device_ids=[device],output_device=rank,find_unused_parameters=True)
        train_dataset = CNN_Transformer_Dataset(X_train_CNN,X_train_Transformer, y_train)
        validation_dataset = CNN_Transformer_Dataset(X_test_CNN,X_test_Transformer,y_test)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False,sampler=DistributedSampler(train_dataset))
        validation_loader = DataLoader(validation_dataset, 100, shuffle=False,sampler=DistributedSampler(validation_dataset))
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # INFO also works
        for i, (n, p) in enumerate(Daily_Model.module.named_parameters()):
            if p.requires_grad:
                print(i, n)

    print('*' * 25, type(train_loader), '*' * 25)
    losses = []
    valid_losses = []
    train_acc = []
    test_acc  = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = SelfDesigned_LossFunction(losstype=Regression_loss_type)
    optimizer = optimizer_lookup(model_parameters=Daily_Model.parameters(), learning_rate=learning_rate)
    scheduler = lr_strategy_lookup_table(optimizer=optimizer)

    if 'tSATPM25' in Transformer_total_channel_names:
        GeoSpecies_index = Transformer_total_channel_names.index('tSATPM25')
    else:
        GeoSpecies_index = 0
        

    
    for epoch in range(TOTAL_EPOCHS):
        correct = 0
        counts = 0
        temp_losses = []
        
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        for i, (CNN_images, Transformer_images, labels) in enumerate(train_loader):
            Daily_Model.train()
            CNN_images = CNN_images.to(device)
            Transformer_images = Transformer_images.to(device)
            if torch.isnan(CNN_images).any():
                print(f"NaN values found in CNN_images at epoch {epoch}, iteration {i}.")
                nan_indices = torch.isnan(CNN_images).nonzero(as_tuple=True)
                print(f"NaN indices in CNN_images:", CNN_images[nan_indices], ' and the labels are: ', labels[nan_indices[0],:])
            if torch.isnan(Transformer_images).any():
                print(f"NaN values found in Transformer_images at epoch {epoch}, iteration {i}.")
                nan_indices = torch.isnan(Transformer_images).nonzero(as_tuple=True)
                print(f"NaN indices in Transformer_images:", Transformer_images[nan_indices], ' and the labels are: ', labels[nan_indices[0],:])
            
            mask = ~torch.isnan(labels)  # Create a mask for valid target data (not NaN)
            filled_labels = torch.nan_to_num(labels, nan=0.0)  # Fill NaN values in target data with 0.0
            filled_labels = filled_labels.to(device)  # Ensure filled_labels has the correct shape
            optimizer.zero_grad()
            outputs = Daily_Model(CNN_images,Transformer_images,filled_labels, teacher_forcing=True)
            loss = criterion(outputs, filled_labels, Transformer_images[:,:,GeoSpecies_index],input_mean_Transformer[GeoSpecies_index],input_std_Transformer[GeoSpecies_index],mask=mask)
            loss.backward()
            optimizer.step()
            
            
            torch.cuda.empty_cache()
            temp_losses.append(loss.item())

            # Calculate R2
            y_hat = np.squeeze(outputs.cpu().detach().numpy())[np.where(mask.squeeze().cpu().detach().numpy())]
            y_true = np.squeeze(filled_labels.cpu().detach().numpy())[np.where(mask.squeeze().cpu().detach().numpy())]
            
            R2 = linear_regression(y_hat, y_true)
            R2 = np.round(R2, 4)
            #print('size of outputs:', outputs.shape, 'size of mask', mask.shape, 'size of y_hat:', y_hat.shape, 'size of y_true:', y_true.shape,'size of filled_labels:', filled_labels.shape,
            #'size of images:', images.shape)
            #print('y_hat:', y_hat[0:20])
            #print('y_true:', y_true[0:20])
            print('R2:', R2)
            correct += R2
            counts  += 1    
            if (i+1)%10 == 0 and rank == 0:
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train_Transformer) // BATCH_SIZE,
                                                                loss.item()))
        losses.append(np.mean(temp_losses))
        valid_correct = 0
        valid_counts  = 0
        temp_losses = []
        scheduler.step()
        total_valid_y_hat = []
        total_valid_y_true = []

        with torch.no_grad():
            for i, (CNN_valid_images, Transformer_valid_images,valid_labels) in enumerate(validation_loader):
                Daily_Model.eval()
                CNN_valid_images = CNN_valid_images.to(device)
                Transformer_valid_images = Transformer_valid_images.to(device)
                valid_mask = ~torch.isnan(valid_labels)
                valid_filled_labels = torch.nan_to_num(valid_labels, nan=0.0)
                valid_filled_labels = valid_filled_labels.to(device)
                valid_output = Daily_Model(CNN_valid_images, Transformer_valid_images, valid_filled_labels, teacher_forcing=False)
                valid_loss = criterion(valid_output, valid_filled_labels, Transformer_valid_images[:,:,GeoSpecies_index],input_mean_Transformer[GeoSpecies_index],input_std_Transformer[GeoSpecies_index],mask=valid_mask)
                temp_losses.append(valid_loss.item())
                test_y_hat = valid_output.cpu().detach().numpy()
                test_y_true = valid_filled_labels.cpu().detach().numpy()
                test_y_hat = np.squeeze(test_y_hat)[np.where(valid_mask.squeeze().cpu().detach().numpy())]
                test_y_true = np.squeeze(test_y_true)[np.where(valid_mask.squeeze().cpu().detach().numpy())]

                valid_R2 = linear_regression(test_y_hat, test_y_true)
                valid_R2 = np.round(valid_R2, 4)
                #print('test_y_hat:', test_y_hat[0:200])
                #print('test_y_true:', test_y_true[0:200])
                valid_correct += valid_R2
                valid_counts += 1
                #print('valid_R2:', valid_R2)
                total_valid_y_hat.append(test_y_hat)
                total_valid_y_true.append(test_y_true)
                if rank == 0:  # Only print from the main process
                        print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                        i + 1, len(X_train_CNN) // BATCH_SIZE,
                                                                        valid_loss.item(), valid_R2))
        valid_losses.append(np.mean(temp_losses))
        accuracy = correct / counts
        valid_accuracy = np.round(linear_regression(np.concatenate(total_valid_y_hat), np.concatenate(total_valid_y_true)), 4)
        print('Epoch : %d/%d, Train Loss: %.4f, Train R2: %.4f' % (epoch + 1, TOTAL_EPOCHS, np.mean(temp_losses), accuracy))
        if rank == 0 and ifold == 0:
            wandb.log({
                'epoch': epoch,
                'learning_rates': optimizer.param_groups[0]['lr'],
                'train_loss': losses[-1],
                'valid_loss': valid_losses[-1],
                'train_accuracy': accuracy,
                'valid_accuracy': valid_accuracy
            })

        train_acc.append(accuracy)
        test_acc.append(valid_accuracy)
        print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])
    raw_model = Daily_Model.module if world_size > 1 else Daily_Model  # Get the underlying model if using DDP

    if world_size > 1:
        dist.barrier()  # ← very important
        # synchronize all ranks before any one finishes

    if rank == 0:
        save_daily_datesbased_model(model=raw_model,evaluation_type=evaluation_type,typeName=typeName,
                                                begindates=begindates,enddates=enddates,
                                                version=version,species=species,nchannel=len(CNN_total_channel_names),d_model=d_model
                                                ,width=width,height=height,n_head=n_head,ffn_hidden=ffn_hidden,num_layers=num_layers,max_len=max_len+spin_up_len,
                                                special_name=description,ifold=ifold,CNN_nchannel=len(CNN_total_channel_names),Transformer_nchannel=len(Transformer_total_channel_names))
    if rank == 0 and ifold == 0:
        wandb.finish()  # only finalize logging after all training done

    if world_size > 1:
        destroy_process_group()

                        
def Transformer_train(rank, world_size, temp_sweep_config, sweep_mode, sweep_id, run_id_container,init_total_channel_names, X_train, y_train, X_test, y_test, input_mean, input_std,
                       evaluation_type, typeName, begindates, enddates, ifold=0):
    print('fold {} is starting...'.format(ifold))
    print('world_size: {}'.format(world_size))
    try:
        print(f"[Rank {rank}] Starting Transformer_train")
        # Your original CNN_train logic goes here...
    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    print(f"Rank {rank} process started.")

    run_config = wandb_run_config()
    if rank == 0 and ifold == 0:
        if sweep_mode:
            wandb_initialize(temp_sweep_config,rank,sweep_mode,sweep_id)
        else:
            wandb_initialize(run_config,rank,sweep_mode,sweep_id)
        run_id_container["run_id"] = wandb.run.id
        run_id_container["run_name"] = wandb.run.name
    if sweep_mode:
        wandb_config = temp_sweep_config
    else:
        wandb_config = run_config
    
    BATCH_SIZE, learning_rate, TOTAL_EPOCHS, d_model, n_head, ffn_hidden, num_layers, max_len, spin_up_len, drop_prob = wandb_parameters_return(wandb_config=wandb_config)
    print('BATCH_SIZE: ', BATCH_SIZE, ' learning_rate: ', learning_rate, 'TOTAL_EPOCHS: ', TOTAL_EPOCHS,
          ' d_model: ', d_model, ' n_head: ', n_head, ' ffn_hidden: ', ffn_hidden, ' num_layers: ', num_layers,
          ' max_len: ', max_len, ' spin_up_len: ', spin_up_len, ' drop_prob: ', drop_prob)
    try:
        channels_to_exclude = wandb_config.get("channel_to_exclude", [])
    except AttributeError:
        channels_to_exclude = []
    total_channel_names, main_stream_channel_names, side_stream_channel_names = Get_channel_names(channels_to_exclude=channels_to_exclude)
    index_of_main_stream_channels_of_initial = [init_total_channel_names.index(channel) for channel in main_stream_channel_names]
    print('X_train:', X_train[0,:])
    X_train = X_train[:, :,index_of_main_stream_channels_of_initial]
    X_test = X_test[:, :,index_of_main_stream_channels_of_initial]

    print('Transformer Architecture X_train shape: ',X_train.shape)
    print('Transformer Architecture y_train shape: ',y_train.shape)
    print('Transformer Architecture X_test shape: ',X_test.shape)
    print('Transformer Architecture y_test shape: ',y_test.shape)

    if rank != 0:
        os.environ['WANDB_MODE'] = 'disabled'
    
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Daily_Model = Transformer(input_dim=X_train.shape[2], trg_dim=Transformer_trg_dim, d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, num_layers=num_layers, max_len=max_len+spin_up_len, drop_prob=drop_prob,device=device)
        Daily_Model.to(device)
        torch.manual_seed(21)
        train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(Dataset(X_test, y_test), 100, shuffle=False)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        device = rank
        Daily_Model = Transformer(input_dim=X_train.shape[2], trg_dim=Transformer_trg_dim, d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, num_layers=num_layers, max_len=max_len+spin_up_len, drop_prob=drop_prob,device=device)
        Daily_Model.to(device)
        torch.manual_seed(21)
        Daily_Model = DDP(Daily_Model, device_ids=[device])
        train_dataset = Dataset(X_train, y_train)
        validation_dataset = Dataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False,sampler=DistributedSampler(train_dataset))
        validation_loader = DataLoader(validation_dataset, 100, shuffle=False,sampler=DistributedSampler(validation_dataset))
    
    print('*' * 25, type(train_loader), '*' * 25)
    losses = []
    valid_losses = []
    train_acc = []
    test_acc  = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = SelfDesigned_LossFunction(losstype=Regression_loss_type)
    optimizer = optimizer_lookup(model_parameters=Daily_Model.parameters(), learning_rate=learning_rate)
    scheduler = lr_strategy_lookup_table(optimizer=optimizer)

    if 'tSATPM25' in total_channel_names:
        GeoSpecies_index = total_channel_names.index('tSATPM25')
    else:
        GeoSpecies_index = 0

    for epoch in range(TOTAL_EPOCHS):
        correct = 0
        counts = 0
        temp_losses = []
        
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_loader):
            Daily_Model.train()
            images = images.to(device)
            #labels = labels.unsqueeze(-1)
            if torch.isnan(images).any():
                print(f"NaN values found in images GG at epoch {epoch}, iteration {i}.")
                nan_indices = torch.isnan(images).nonzero(as_tuple=True)
                print(f"NaN indices in images:", images[nan_indices], ' and the labels are: ', labels[nan_indices[0],:])
            mask = ~torch.isnan(labels)  # Create a mask for valid target data (not NaN)
            filled_labels = torch.nan_to_num(labels, nan=0.0)  # Fill NaN values in target data with 0.0
            filled_labels = filled_labels.to(device)  # Ensure filled_labels has the correct shape
            optimizer.zero_grad()
            outputs = Daily_Model(images,filled_labels)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, filled_labels, images[:,:,GeoSpecies_index],input_mean[GeoSpecies_index],input_std[GeoSpecies_index],mask=mask)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            temp_losses.append(loss.item())

            # Calculate R2
            y_hat = np.squeeze(outputs.cpu().detach().numpy())[np.where(mask.squeeze().cpu().detach().numpy())]
            y_true = np.squeeze(filled_labels.cpu().detach().numpy())[np.where(mask.squeeze().cpu().detach().numpy())]
            
            R2 = linear_regression(y_hat, y_true)
            R2 = np.round(R2, 4)
            #print('size of outputs:', outputs.shape, 'size of mask', mask.shape, 'size of y_hat:', y_hat.shape, 'size of y_true:', y_true.shape,'size of filled_labels:', filled_labels.shape,
            #'size of images:', images.shape)
            #print('y_hat:', y_hat[0:20])
            #print('y_true:', y_true[0:20])
            print('R2:', R2)
            correct += R2
            counts  += 1    
            if (i+1)%10 == 0 and rank == 0:
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train) // BATCH_SIZE,
                                                                loss.item()))
        losses.append(np.mean(temp_losses))
        valid_correct = 0
        valid_counts  = 0
        temp_losses = []
        scheduler.step()
        total_valid_y_hat = []
        total_valid_y_true = []
        with torch.no_grad():
            for i, (valid_images, valid_labels) in enumerate(validation_loader):
                
                Daily_Model.eval()
                valid_images = valid_images.to(device)
                valid_mask = ~torch.isnan(valid_labels)
                valid_filled_labels = torch.nan_to_num(valid_labels, nan=0.0)
                valid_filled_labels = valid_filled_labels.to(device)
                valid_output = Daily_Model(valid_images)
                valid_loss = criterion(valid_output, valid_filled_labels, valid_images[:,:,GeoSpecies_index],input_mean[GeoSpecies_index],input_std[GeoSpecies_index],mask=valid_mask)
                temp_losses.append(valid_loss.item())
                test_y_hat = valid_output.cpu().detach().numpy()
                test_y_true = valid_filled_labels.cpu().detach().numpy()
                test_y_hat = np.squeeze(test_y_hat)[np.where(valid_mask.squeeze().cpu().detach().numpy())]
                test_y_true = np.squeeze(test_y_true)[np.where(valid_mask.squeeze().cpu().detach().numpy())]

                valid_R2 = linear_regression(test_y_hat, test_y_true)
                valid_R2 = np.round(valid_R2, 4)
                print('test_y_hat:', test_y_hat[0:200])
                print('test_y_true:', test_y_true[0:200])
                valid_correct += valid_R2
                valid_counts += 1
                print('valid_R2:', valid_R2)
                total_valid_y_hat.append(test_y_hat)
                total_valid_y_true.append(test_y_true)
                if rank == 0:  # Only print from the main process
                        print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                        i + 1, len(X_train) // BATCH_SIZE,
                                                                        valid_loss.item(), valid_R2))
        valid_losses.append(np.mean(temp_losses))
        accuracy = correct / counts
        valid_accuracy = np.round(linear_regression(np.concatenate(total_valid_y_hat), np.concatenate(total_valid_y_true)), 4)
        print('Epoch : %d/%d, Train Loss: %.4f, Train R2: %.4f' % (epoch + 1, TOTAL_EPOCHS, np.mean(temp_losses), accuracy))
        if rank == 0 and ifold == 0:
            wandb.log({
                'epoch': epoch,
                'learning_rates': optimizer.param_groups[0]['lr'],
                'train_loss': losses[-1],
                'valid_loss': valid_losses[-1],
                'train_accuracy': accuracy,
                'valid_accuracy': valid_accuracy
            })

        train_acc.append(accuracy)
        test_acc.append(valid_accuracy)
        print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])
    raw_model = Daily_Model.module if world_size > 1 else Daily_Model  # Get the underlying model if using DDP

    if world_size > 1:
        dist.barrier()  # ← very important
        # synchronize all ranks before any one finishes

    if rank == 0:
        save_daily_datesbased_model(model=raw_model,evaluation_type=evaluation_type,typeName=typeName,
                                                begindates=begindates,enddates=enddates,
                                                version=version,species=species,nchannel=len(main_stream_channel_names),d_model=d_model
                                                ,n_head=n_head,ffn_hidden=ffn_hidden,num_layers=num_layers,max_len=max_len+spin_up_len,
                                                special_name=description,ifold=ifold)
    if rank == 0 and ifold == 0:
        wandb.finish()  # only finalize logging after all training done

    if world_size > 1:
        destroy_process_group()

    return

def CNN3D_train(rank,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,init_total_channel_names,X_train, y_train,X_test,y_test,input_mean, input_std,width,height,depth,
              evaluation_type,typeName,begindates,enddates,ifold=0):
    import sys
    import os
    # 在函数最开头显式设置，确保子进程也有
    # CNN3D_train函数最开头：
    os.environ['TORCH_LOGS'] = 'recompiles'  # ← 第一行，其他所有代码之前
    print(f"[Rank {rank}] ENV: {os.environ.get('TORCHINDUCTOR_CACHE_DIR', 'NOT SET')}", flush=True)
    cache_dir = '/s.siyuan/my-projects2/torch_compile_cache'
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir
    os.environ['TORCH_COMPILE_CACHE_DIR'] = cache_dir
    print(f"[Rank {rank}] Set TORCHINDUCTOR_CACHE_DIR and TORCH_COMPILE_CACHE_DIR to: {cache_dir}", flush=True)
    
    
    print('fold {} is starting...'.format(ifold))
    print('world_size: {}'.format(world_size))
    try:
        print(f"[Rank {rank}] Starting 3DCNN_train")
        # Your original 3DCNN_train logic goes here...
    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        print(f"[Rank {rank}] Exception occurred:\n{traceback.format_exc()}")
        raise e
    print(f"Rank {rank} process started.")

    run_config = wandb_run_config()
    if rank == 0 and ifold == 0:
        if sweep_mode:
            wandb_initialize(temp_sweep_config,rank,sweep_mode,sweep_id)
        else:
            wandb_initialize(run_config,rank,sweep_mode,sweep_id)
        print(f"type(run_id_container): {type(run_id_container)}")
        run_id_container["run_id"] = wandb.run.id
        run_id_container["run_name"] = wandb.run.name   
    if sweep_mode:
        wandb_config = temp_sweep_config
    else:
        wandb_config = run_config

    BATCH_SIZE, learning_rate, TOTAL_EPOCHS = wandb_parameters_return(wandb_config=wandb_config)
    
    ## This is for training
    try:
        channels_to_exclude = wandb_config.get("channel_to_exclude", [])
    except AttributeError:
        channels_to_exclude = []
    if Apply_3D_CNN_architecture and MoCE_Settings:
        total_channel_names, main_stream_channel_names, side_stream_channel_names = Get_channel_names(channels_to_exclude=channels_to_exclude,
                                                                                                      MoCE_base_model_channels=wandb_config['MoCE_base_model_channels'],
                                                                                                MoCE_side_experts_channels_list=wandb_config['MoCE_side_experts_channels_list'])
    else:
        total_channel_names, main_stream_channel_names, side_stream_channel_names = Get_channel_names(channels_to_exclude=channels_to_exclude)


    index_of_main_stream_channels_of_initial = [init_total_channel_names.index(channel) for channel in main_stream_channel_names]

    X_train = X_train[:,index_of_main_stream_channels_of_initial,:,:,:]
    X_test  = X_test[:,index_of_main_stream_channels_of_initial,:,:,:]


    print('X_train shape: ',X_train.shape)
    print('y_train shape: ',y_train.shape)
    print('X_test shape: ',X_test.shape)
    print('y_test shape: ',y_test.shape)
    scaler = GradScaler('cuda')
    
    if rank != 0:
        os.environ['WANDB_MODE'] = 'disabled'
    torch.set_float32_matmul_precision('high')
    if world_size <= 1:
        torch.manual_seed(21)
        Daily_Model = initial_3dcnn_net(main_stream_channel=main_stream_channel_names,wandb_config=wandb_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Daily_Model.to(device)
        torch._inductor.config.fx_graph_cache = True
        Daily_Model = torch.compile(Daily_Model) # Optional: Compile the model for potential speedup (PyTorch 2.0+)
        
        train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(Dataset(X_test, y_test), 2000, shuffle=True)
    elif world_size > 1:
        torch.manual_seed(21)
        ddp_setup(rank, world_size)
        Daily_Model = initial_3dcnn_net(main_stream_channel=main_stream_channel_names,wandb_config=wandb_config)
        device = rank
        Daily_Model.to(device)
        torch._inductor.config.fx_graph_cache = True
        Daily_Model = torch.compile(Daily_Model,mode='reduce-overhead') # Optional: Compile the model for potential speedup (PyTorch 2.0+)
        Daily_Model = DDP(Daily_Model, device_ids=[device])
        Daily_Model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)  # Register the allreduce hook for gradient synchronization
        train_dataset = Dataset(X_train, y_train)
        validation_dataset = Dataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False,sampler=DistributedSampler(train_dataset, drop_last=True),
                                  num_workers=0,        
                                  pin_memory=True,      # faster CPU→GPU transfer
                                  drop_last=True,  # ← 丢弃最后一个不完整batch，避免重新编译
                                  )
        validation_loader = DataLoader(validation_dataset, BATCH_SIZE, shuffle=False,sampler=DistributedSampler(validation_dataset, drop_last=True),num_workers=0,pin_memory=True,
                                       drop_last=True)  # must use the same batch size for validation to avoid re-compilation in DDP
    
    
    print('*' * 25, type(train_loader), '*' * 25)
    losses = []
    valid_losses = []
    train_acc = []
    test_acc  = []

    criterion = SelfDesigned_LossFunction(losstype=Regression_loss_type)
    optimizer = optimizer_lookup(model_parameters=Daily_Model.parameters(), learning_rate=learning_rate)
    scheduler = lr_strategy_lookup_table(optimizer=optimizer)
    
    if 'tSATPM25' in total_channel_names:
        GeoSpecies_index = total_channel_names.index('tSATPM25')
    else:
        GeoSpecies_index = 0

    # Before epoch loop:
    
    total_epoch_time = 0
    
    center_width = int((width-1)/2)
    center_height = int((height-1)/2)
    '''
    # 在epoch循环之前加这段诊断代码，只运行一次
    print("=== 数据管道诊断 ===")
    # 测试1：纯数据加载速度（不做任何GPU操作）
    t0 = time.perf_counter()
    batch_count = 0
    for images, labels in train_loader:
        batch_count += 1
        if batch_count == 50:
            break
    load_only_time = (time.perf_counter() - t0) / 50
    print(f"纯数据加载时间（无GPU）: {load_only_time*1000:.1f}ms/batch")

    # 测试2：数据加载 + GPU传输
    t0 = time.perf_counter()
    batch_count = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        torch.cuda.synchronize()
        batch_count += 1
        if batch_count == 50:
            break
    load_transfer_time = (time.perf_counter() - t0) / 50
    print(f"数据加载+GPU传输时间: {load_transfer_time*1000:.1f}ms/batch")

    # 测试3：完整forward+backward
    t0 = time.perf_counter()
    batch_count = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32, non_blocking=True)
        labels = torch.squeeze(labels)
        with autocast('cuda'):
            outputs = Daily_Model(images)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, labels,
                        images[:,GeoSpecies_index,-1,center_width,center_height],
                        input_mean[GeoSpecies_index,-1,center_width,center_height],
                        input_std[GeoSpecies_index,-1,center_width,center_height])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        batch_count += 1
        if batch_count == 50:
            break
    full_time = (time.perf_counter() - t0) / 50
    print(f"完整训练步骤时间: {full_time*1000:.1f}ms/batch")
    print(f"=== GPU计算时间占比: {(full_time - load_transfer_time)/full_time*100:.1f}% ===")
    print(f"=== 数据瓶颈占比:   {load_transfer_time/full_time*100:.1f}% ===")
        
    # 测试4: 单独计时forward，loss计算，backward
    t0 = time.perf_counter()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32, non_blocking=True)
        labels = torch.squeeze(labels)
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        # 单独计时forward
        with autocast('cuda'):
            outputs = Daily_Model(images)
            outputs = torch.squeeze(outputs)
        torch.cuda.synchronize()
        t_forward = time.perf_counter()
        
        # 单独计时loss
        with autocast('cuda'):
            loss = criterion(outputs, labels,
                            images[:,GeoSpecies_index,-1,center_width,center_height],
                            input_mean[GeoSpecies_index,-1,center_width,center_height],
                            input_std[GeoSpecies_index,-1,center_width,center_height])
        torch.cuda.synchronize()
        t_loss = time.perf_counter()
        
        # 单独计时backward
        # 用no_sync()禁用DDP梯度同步，只测纯backward计算
        with Daily_Model.no_sync():  # 如果使用DDP，避免在每个batch都进行梯度同步
            scaler.scale(loss).backward(retain_graph=True) 
        torch.cuda.synchronize()
        t_backward = time.perf_counter()
        
        # 正常 backward(含 DDP AllReduce同步)
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t_backward_sync = time.perf_counter()
        
        if rank == 0:
            print(f"Batch {i}: "
            f"forward={1000*(t_forward-t_start):.1f}ms, "
            f"loss={1000*(t_loss-t_forward):.1f}ms, "
            f"backward={1000*(t_backward-t_loss):.1f}ms, "
            f"backward_sync={1000*(t_backward_sync-t_backward):.1f}ms")
    '''
    for epoch in range(TOTAL_EPOCHS):
        total_samples_processed = 0
        temp_losses = []
        # Accumulate outputs for epoch-level R2 instead of per-batch
        running_sum_yhat  = 0.0
        running_sum_ytrue = 0.0
        running_sum_yhat2 = 0.0
        running_sum_ytrue2 = 0.0
        running_sum_cross = 0.0
        running_count     = 0
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        Daily_Model.train()
        '''
        if rank == 0 and epoch == 0:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False
            ) as prof:
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, dtype=torch.float32, non_blocking=True)
                    labels = torch.squeeze(labels)
                    with record_function("forward"):
                        with autocast('cuda'):
                            outputs = Daily_Model(images)
                            outputs = torch.squeeze(outputs)
                            loss = criterion(outputs, labels,
                                            images[:,GeoSpecies_index,-1,center_width,center_height],
                                            input_mean[GeoSpecies_index,-1,center_width,center_height],
                                            input_std[GeoSpecies_index,-1,center_width,center_height])
                    with record_function("backward"):
                        scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if i == 10:  # 只profile前10个batch
                        break
            
            # 打印最耗时的操作
            print("=== Profiler Results (Top 15) ===")
            print(prof.key_averages().table(
                sort_by="cuda_time_total", 
                row_limit=15),
                flush=True)
        '''
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} started. Processing batches...")
        epoch_start = time.perf_counter()
        for i, (images, labels) in enumerate(train_loader):
            batch_start = time.perf_counter()
            ## Example: Check for NaN values in input images
            images = images.to(device, non_blocking=True) 
            labels = labels.to(device,dtype=torch.float32, non_blocking=True)  # Ensure labels are float for regression
            labels = torch.squeeze(labels)  # Remove extra dimensions if necessary
            
            if epoch == 0 and i == 0 and rank == 0:
                t0 = time.perf_counter()
        
            optimizer.zero_grad(set_to_none=True)  # More efficient zeroing of gradients
            with autocast('cuda'):
                outputs = Daily_Model(images)
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, labels, images[:,GeoSpecies_index,-1,center_width,center_height],input_mean[GeoSpecies_index,-1,center_width,center_height],input_std[GeoSpecies_index,-1,center_width,center_height])
            scaler.scale(loss).backward()  ## backward
            scaler.step(optimizer)
            scaler.update()
            # 计时结束，只执行一次
            if epoch == 0 and i == 0 and rank == 0:
                torch.cuda.synchronize()  # 只在这一个batch用
                print(f"First batch time: {time.perf_counter()-t0:.1f}s at fold {ifold}", flush=True)
                # <2s  → cache命中
                # >60s → 正在编译
    
            #outputs = Daily_Model(images)
            #outputs = torch.squeeze(outputs)
            #loss = criterion(outputs, labels, images[:,GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)])
            #loss.backward()
            #optimizer.step()
            
            temp_losses.append(loss.item())
            
            # Inside training loop, after optimizer.step():
            batch_time = time.perf_counter() - batch_start
            samples_per_sec = len(images) / batch_time
            total_samples_processed += len(images)
            
            with torch.no_grad():
                n      = outputs.numel()
                running_sum_yhat  += outputs.sum().item()    # .item() is just a scalar — fast
                running_sum_ytrue += labels.sum().item()
                running_sum_yhat2 += (outputs**2).sum().item()
                running_sum_ytrue2+= (labels**2).sum().item()
                running_sum_cross += (outputs * labels).sum().item()
                running_count     += n

            if (i + 1) % 10 == 0 and rank == 0 :
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f, Throughput: %.1f samples/sec' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train)// world_size // BATCH_SIZE,
                                                                loss.item(), samples_per_sec))
        # After each epoch:
        epoch_time = time.perf_counter() - epoch_start
        total_epoch_time += epoch_time
        # Compute R2 from accumulated statistics — no large tensor transfer needed
        mean_yhat  = running_sum_yhat  / running_count
        mean_ytrue = running_sum_ytrue / running_count
        ss_res = (running_sum_yhat2 - 2*mean_ytrue*running_sum_yhat + running_count*mean_ytrue**2)  
        # simplification — just use pearson r formula:
        numerator   = running_sum_cross - running_count * mean_yhat * mean_ytrue
        denom_yhat  = (running_sum_yhat2  - running_count * mean_yhat**2)  ** 0.5
        denom_ytrue = (running_sum_ytrue2 - running_count * mean_ytrue**2) ** 0.5
        train_accuracy = np.round((numerator / (denom_yhat * denom_ytrue + 1e-8)) ** 2, 4)

        if rank == 0:
            print(f"Epoch {epoch} took {epoch_time:.1f}s | "
                f"Avg throughput: {total_samples_processed/epoch_time:.1f} samples/sec | "
                f"Avg Each Epoch time {total_epoch_time/(epoch+1):.1f}s")
        losses.append(np.mean(temp_losses))
        scheduler.step()
        

        temp_losses = []
        val_running_sum_yhat  = 0.0
        val_running_sum_ytrue = 0.0
        val_running_sum_yhat2 = 0.0
        val_running_sum_ytrue2= 0.0
        val_running_sum_cross = 0.0
        val_running_count     = 0
        Daily_Model.eval()
        time_valid_start = time.perf_counter()
        for i, (valid_images, valid_labels) in enumerate(validation_loader):
            valid_images = valid_images.to(device, non_blocking=True)
            valid_labels = valid_labels.to(device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad(), autocast('cuda'):
                valid_output = Daily_Model(valid_images)
                valid_output = torch.squeeze(valid_output)
                valid_loss   = criterion(valid_output, valid_labels, valid_images[:,GeoSpecies_index,-1,center_width,center_height],input_mean[GeoSpecies_index,-1,center_width,center_height],input_std[GeoSpecies_index,-1,center_width,center_height])
            temp_losses.append(valid_loss.item())
            with torch.no_grad():
                n = valid_output.numel()
                val_running_sum_yhat  += valid_output.sum().item()
                val_running_sum_ytrue += valid_labels.sum().item()
                val_running_sum_yhat2 += (valid_output**2).sum().item()
                val_running_sum_ytrue2+= (valid_labels**2).sum().item()
                val_running_sum_cross += (valid_output * valid_labels).sum().item()
                val_running_count     += n
        valid_losses.append(np.mean(temp_losses))
        # Compute val R2 from scalars
        mean_yhat  = val_running_sum_yhat  / val_running_count
        mean_ytrue = val_running_sum_ytrue / val_running_count
        numerator   = val_running_sum_cross - val_running_count * mean_yhat * mean_ytrue
        denom_yhat  = (val_running_sum_yhat2  - val_running_count * mean_yhat**2)  ** 0.5
        denom_ytrue = (val_running_sum_ytrue2 - val_running_count * mean_ytrue**2) ** 0.5
        test_accuracy = np.round((numerator / (denom_yhat * denom_ytrue + 1e-8)) ** 2, 4)
        time_valid_end = time.perf_counter()
        if rank == 0:
            print(f'Epoch {epoch+1}/{TOTAL_EPOCHS} | '
                  f'Training Time: {epoch_time:.1f}s | '
                  f'Validation Time: {time_valid_end - time_valid_start:.1f}s | '
                  f'Train Loss: {losses[-1]:.4f} | Train R2: {train_accuracy:.4f} | '
                  f'Val Loss: {valid_losses[-1]:.4f} | Val R2: {test_accuracy:.4f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
                  )
            
            if ifold == 0:  # Log only from the main process
                wandb.log({
                    'epoch': epoch,
                    'learning_rates': optimizer.param_groups[0]['lr'],
                    'train_loss': losses[-1],
                    'valid_loss': valid_losses[-1],
                    'train_accuracy': train_accuracy,
                    'valid_accuracy': test_accuracy,
                    'throughput_samples_per_sec': total_samples_processed/epoch_time,
                    'epoch_time_sec': epoch_time,
                    'validation_time_sec': time_valid_end - time_valid_start,
                })

        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])
    raw_model = Daily_Model.module if world_size > 1 else Daily_Model  # Get the underlying model if using DDP

    if rank == 0:
        print(f"Training finished, starting barrier...", flush=True)
    if world_size > 1:
        time_barrier_start = time.perf_counter()
        dist.barrier()  # ← very important
        time_barrier_end = time.perf_counter()
        if rank == 0:
            print(f"Barrier passed, time waited: {time_barrier_end - time_barrier_start:.1f}s, saving model...", flush=True)
        # synchronize all ranks before any one finishes

    if rank == 0:
        print(f"Barrier passed, saving model...", flush=True)
        time_save_start = time.perf_counter()
        save_daily_datesbased_model(model=raw_model,evaluation_type=evaluation_type,typeName=typeName,
                                                begindates=begindates,enddates=enddates,
                                                version=version,species=species,nchannel=len(main_stream_channel_names),width=width,height=height,depth=depth,
                                                special_name=description,ifold=ifold)
        time_save_end = time.perf_counter()
        print(f"Model saved, time taken: {time_save_end - time_save_start:.1f}s, finishing wandb...", flush=True)
        
    if rank == 0 and ifold == 0:
        time_wandb_start = time.perf_counter()
        wandb.finish()  # only finalize logging after all training done
        time_wandb_end = time.perf_counter()
        print(f"wandb finished, time taken: {time_wandb_end - time_wandb_start:.1f}s", flush=True)
    if world_size > 1:
        print(f"[Rank {rank}] destroying process group...", flush=True)
        time_destroy_start = time.perf_counter()
        destroy_process_group()
        time_destroy_end = time.perf_counter()
        print(f"[Rank {rank}] process group destroyed, time taken: {time_destroy_end - time_destroy_start:.1f}s", flush=True)

    

def CNN_train(rank,world_size,temp_sweep_config,sweep_mode,sweep_id,run_id_container,init_total_channel_names,X_train, y_train,X_test,y_test,input_mean, input_std,width,height,
              evaluation_type,typeName,begindates,enddates,ifold=0):
    print('fold {} is starting...'.format(ifold))
    print('world_size: {}'.format(world_size))
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
        if sweep_mode:
            wandb_initialize(temp_sweep_config,rank,sweep_mode,sweep_id)
        else:
            wandb_initialize(run_config,rank,sweep_mode,sweep_id)
        run_id_container["run_id"] = wandb.run.id
        run_id_container["run_name"] = wandb.run.name
    if sweep_mode:
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

    if world_size <= 1:
        Daily_Model = initial_cnn_network(width=width, main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_channel_names),wandb_config=wandb_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Daily_Model.to(device)
        torch.manual_seed(21)
        train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(Dataset(X_test, y_test), 2000, shuffle=True)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        Daily_Model = initial_cnn_network(width=width, main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_channel_names),wandb_config=wandb_config)
        device = rank
        Daily_Model.to(device)
        torch.manual_seed(21 + rank)
        Daily_Model = DDP(Daily_Model, device_ids=[device],output_device=device,find_unused_parameters=False,static_graph=True,gradient_as_bucket_view=True,)
        train_dataset = Dataset(X_train, y_train)
        validation_dataset = Dataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False,sampler=DistributedSampler(train_dataset, drop_last=True))
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

    if 'tSATPM25' in total_channel_names:
        GeoSpecies_index = total_channel_names.index('tSATPM25')
    else:
        GeoSpecies_index = 0


    for epoch in range(TOTAL_EPOCHS):
        correct = 0
        counts = 0
        temp_losses = []
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
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

        if rank == 0 and ifold == 0:  # Log only from the main process
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

    # 1. Synchronize all ranks before any one finishes
    if world_size > 1:
        dist.barrier()  # ← very important
    raw_model = Daily_Model.module if world_size > 1 else Daily_Model  # Get the underlying model if using DDP
    if rank == 0:
        save_daily_datesbased_model(model=raw_model,evaluation_type=evaluation_type,typeName=typeName,
                                                begindates=begindates,enddates=enddates,
                                                version=version,species=species,nchannel=len(main_stream_channel_names),width=width,height=height,
                                                special_name=description,ifold=ifold)

    if rank == 0 and ifold == 0:
        wandb.finish()  # only finalize logging after all training done
    if world_size > 1:
        destroy_process_group()

    

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
    all_outputs = []
    predictinput = DataLoader(Dataset_Val(inputarray), batch_size= batchsize,num_workers=0, # 和训练时一样，避免序列化开销
        pin_memory=True,
)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device  # 直接从模型获取device，不强制转移
    model.to(device)
    
    with torch.no_grad():
        for i, image in enumerate(predictinput):
            image = image.to(device, non_blocking=True)
            output = model(image).cpu().detach().numpy()
            with torch.amp.autocast('cuda'):        # AMP加速推理
                output = model(image)
            all_outputs.append(output.detach().cpu())  # 保持tensor，不转numpy
    
    return torch.cat(all_outputs, dim=0).numpy().flatten()

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

def transformer_predict(inputarray, model, batchsize,initial_channel_names,mainstream_channel_names,sidestream_channel_names):
    """
    This function is used to predict the PM2.5 concentration using a Transformer model.
    
    Args:
        inputarray (numpy.ndarray): The input data for prediction.
        model (torch.nn.Module): The trained Transformer model.
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

def cnn_transformer_predict(CNN_inputarray, Transformer_inputarray, model, batchsize):
    model.eval()
    final_output = []
    final_output = np.array(final_output)
    predictinput = DataLoader(Dataset(CNN_inputarray, Transformer_inputarray), batch_size= batchsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for i, (cnn_image, transformer_image) in enumerate(predictinput):
            cnn_image = cnn_image.to(device)
            transformer_image = transformer_image.to(device)
            output = model(cnn_image, transformer_image, teacher_forcing=False).cpu().detach().numpy()
            final_output = np.append(final_output,output)
    
    return final_output