import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.utils.data import DataLoader
from Training_pkg.utils import *
from Training_pkg.Statistic_func import linear_regression
from Training_pkg.TensorData_func import Dataset,Dataset_Val
from Training_pkg.Loss_func import SelfDesigned_LossFunction
import torch.nn.functional as F
from Model_Structure_pkg.utils import *

def CNN3D_train(model, X_train, y_train, X_test, y_test, input_mean, input_std, width, height, BATCH_SIZE, learning_rate, TOTAL_EPOCHS, channel_names):
    print('X_train shape: ',X_train.shape)
    print('y_train shape: ',y_train.shape)
    print('X_test shape: ',X_test.shape)
    print('y_test shape: ',y_test.shape)

    train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(Dataset(X_test, y_test), 200000, shuffle=True)
    print('*' * 25, type(train_loader), '*' * 25)
    losses = []
    valid_losses = []
    train_acc = []
    test_acc  = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = SelfDesigned_LossFunction(losstype=Regression_loss_type)
    optimizer = optimizer_lookup(model_parameters=model.parameters(), learning_rate=learning_rate)
    scheduler = lr_strategy_lookup_table(optimizer=optimizer)

    GeoSpecies_index = channel_names.index('tSATPM25')

    for epoch in range(TOTAL_EPOCHS):
        correct = 0
        counts = 0
        temp_losses = []
        for i, (images, labels) in enumerate(train_loader):
            
            model.train()
            images = images.to(device)
            labels = torch.squeeze(labels.type(torch.FloatTensor))
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
            if (i + 1) % 10 == 0:
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train) // BATCH_SIZE,
                                                                loss.item()))
        losses.append(np.mean(temp_losses))
        valid_correct = 0
        valid_counts  = 0
        temp_losses = []
        scheduler.step()

        for i, (valid_images, valid_labels) in enumerate(validation_loader):
            model.eval()
            valid_images = valid_images.to(device)
            valid_labels = valid_labels.to(device)
            valid_output = model(valid_images)
            valid_output = torch.squeeze(valid_output)
            valid_loss   = criterion(valid_output, valid_labels, valid_images[:,GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)],input_mean[GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)],input_std[GeoSpecies_index,-1,int((width-1)/2),int((height-1)/2)])
            temp_losses.append(valid_loss.item())
            test_y_hat   = valid_output.cpu().detach().numpy()
            test_y_true  = valid_labels.cpu().detach().numpy()
            Valid_R2 = linear_regression(test_y_hat, test_y_true)
            Valid_R2 = np.round(Valid_R2, 4)
            valid_correct += Valid_R2
            valid_counts  += 1    
            print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train) // BATCH_SIZE,
                                                                valid_loss.item(), Valid_R2))
        valid_losses.append(np.mean(temp_losses))
        accuracy = correct / counts
        test_accuracy = valid_correct / valid_counts
        print('Epoch: ',epoch, ', Training Loss: ', loss.item(),', Training accuracy:',accuracy, ', \nTesting Loss:', valid_loss.item(),', Testing accuracy:', test_accuracy)
        if wandb.run is not None:   
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
    

    return losses, train_acc, valid_losses, test_acc

def CNN_train(model,X_train, y_train,X_test,y_test,input_mean, input_std,width,height, BATCH_SIZE, learning_rate, TOTAL_EPOCHS,channel_names):
    print('X_train shape: ',X_train.shape)
    print('y_train shape: ',y_train.shape)
    print('X_test shape: ',X_test.shape)
    print('y_test shape: ',y_test.shape)

    train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(Dataset(X_test, y_test), 2000, shuffle=True)
    print('*' * 25, type(train_loader), '*' * 25)
    losses = []
    valid_losses = []
    train_acc = []
    test_acc  = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = SelfDesigned_LossFunction(losstype=Regression_loss_type)
    #optimizer = torch.optim.Adam(params=model.parameters(),betas=(), lr=learning_rate)
    optimizer = optimizer_lookup(model_parameters=model.parameters(),learning_rate=learning_rate)
    scheduler = lr_strategy_lookup_table(optimizer=optimizer)

    GeoSpecies_index = channel_names.index('tSATPM25')

    for epoch in range(TOTAL_EPOCHS):
        correct = 0
        counts = 0
        temp_losses = []
        for i, (images, labels) in enumerate(train_loader):
            
            model.train()
            images = images.to(device)
            labels = torch.squeeze(labels.type(torch.FloatTensor))
            labels = labels.to(device)
            optimizer.zero_grad()  # Set grads to zero
            outputs = model(images) #dimension: Nx1
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
            if (i + 1) % 100 == 0:
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
            model.eval()
            valid_images = valid_images.to(device)
            valid_labels = valid_labels.to(device)
            valid_output = model(valid_images)
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
            print('Epoch : %d/%d, Iter : %d/%d,  Validate Loss: %.4f, Validate R2: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                i + 1, len(X_train) // BATCH_SIZE,
                                                                valid_loss.item(), Valid_R2)) 
        valid_losses.append(np.mean(temp_losses))
        accuracy = correct / counts
        test_accuracy = valid_correct / valid_counts
        print('Epoch: ',epoch, ', Training Loss: ', loss.item(),', Training accuracy:',accuracy, ', \nTesting Loss:', valid_loss.item(),', Testing accuracy:', test_accuracy)

        if wandb.run is not None:
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
    return losses, train_acc, valid_losses, test_acc

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