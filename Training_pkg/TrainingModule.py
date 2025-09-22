import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from Training_pkg.utils import *
from Training_pkg.Statistic_func import linear_regression
from Training_pkg.TensorData_func import Dataset,Dataset_Val,CNN_Transformer_Dataset,CNN_Transformer_Dataset_Val
from Training_pkg.Loss_func import SelfDesigned_LossFunction
from Training_pkg.iostream import save_daily_datesbased_model
import torch.nn.functional as F
from Model_Structure_pkg.Transformer_Model.model.transformer import Transformer
from Model_Structure_pkg.CNN_Transformer_Model.model.CNN_transformer import CNN_Transformer
from Model_Structure_pkg.CNN_Module import initial_cnn_network
from Model_Structure_pkg.ResCNN3D_Module import initial_3dcnn_net
from Model_Structure_pkg.utils import *
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from wandb_config import *


def ddp_setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

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
        validation_loader = DataLoader(CNN_Transformer_Dataset_Val(X_test_CNN,X_test_Transformer), 100, shuffle=False)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        device = rank
        Daily_Model = CNN_Transformer(CNN_input_dim=X_train_CNN.shape[2], transformer_input_dim=X_train_Transformer.shape[2], trg_dim=CNN_Transformer_trg_dim, d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, num_layers=num_layers, max_len=max_len+spin_up_len, drop_prob=drop_prob,device=device)
        Daily_Model.to(device)
        torch.manual_seed(21)
        Daily_Model = DDP(Daily_Model, device_ids=[device])
        train_dataset = CNN_Transformer_Dataset(X_train_CNN,X_train_Transformer, y_train)
        validation_dataset = CNN_Transformer_Dataset_Val(X_test_CNN,X_test_Transformer)
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
            outputs = Daily_Model(CNN_images,Transformer_images,filled_labels)
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
                valid_output = Daily_Model(CNN_valid_images, Transformer_valid_images)
                valid_loss = criterion(valid_output, valid_filled_labels, Transformer_valid_images[:,:,GeoSpecies_index],input_mean_Transformer[GeoSpecies_index],input_std_Transformer[GeoSpecies_index],mask=valid_mask)
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
    
    if world_size <= 1:
        Daily_Model = initial_3dcnn_net(main_stream_nchannel=len(main_stream_channel_names),wandb_config=wandb_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Daily_Model.to(device)
        torch.manual_seed(21)
        train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(Dataset(X_test, y_test), 2000, shuffle=True)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        Daily_Model = initial_3dcnn_net(main_stream_nchannel=len(main_stream_channel_names),wandb_config=wandb_config)
        device = rank
        Daily_Model.to(device)
        torch.manual_seed(21)
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
        total_valid_y_hat = []
        total_valid_y_true = []
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
            total_valid_y_hat.append(test_y_hat)
            total_valid_y_true.append(test_y_true)
            if rank == 0:  # Only print from the main process
                print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train) // BATCH_SIZE,
                                                                valid_loss.item(), Valid_R2))
        valid_losses.append(np.mean(temp_losses))
        accuracy = correct / counts
        test_accuracy = np.round(linear_regression(np.concatenate(total_valid_y_hat), np.concatenate(total_valid_y_true)), 4)
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
    raw_model = Daily_Model.module if world_size > 1 else Daily_Model  # Get the underlying model if using DDP

    
    if world_size > 1:
        dist.barrier()  # ← very important
        # synchronize all ranks before any one finishes

    if rank == 0:
        save_daily_datesbased_model(model=raw_model,evaluation_type=evaluation_type,typeName=typeName,
                                                begindates=begindates,enddates=enddates,
                                                version=version,species=species,nchannel=len(main_stream_channel_names),width=width,height=height,depth=depth,
                                                special_name=description,ifold=ifold)
    if rank == 0 and ifold == 0:
        wandb.finish()  # only finalize logging after all training done

    if world_size > 1:
        destroy_process_group()

    

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
    predictinput = DataLoader(TensorDataset(CNN_inputarray, Transformer_inputarray), batch_size= batchsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for i, (cnn_image, transformer_image) in enumerate(predictinput):
            cnn_image = cnn_image.to(device)
            transformer_image = transformer_image.to(device)
            output = model(cnn_image, transformer_image).cpu().detach().numpy()
            final_output = np.append(final_output,output)
    
    return final_output