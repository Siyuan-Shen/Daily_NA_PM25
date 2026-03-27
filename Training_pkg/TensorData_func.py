import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CNN_Transformer_Dataset(torch.utils.data.Dataset):  # 'Characterizes a dataset for PyTorch'
    '''
    This class is for training datasets. It is used for the global datasets, which is continuous data.
    '''
    def __init__(self, CNN_traindata, Transformer_traindata, truedata):  # 'Initialization' Data Loading
        '''

        :param traindata:
            Training data.
        :param truedata:
            Ture data to learn.
        :param beginyear:
            The begin year.
        :param endyear:
            The end year.
        :param nsite:
            The number of sites. For example, for overall observation it is 10870.
        '''
        super(CNN_Transformer_Dataset, self).__init__()


        self.CNN_traindatasets = torch.Tensor(CNN_traindata) #torch.squeeze(torch.Tensor(traindata))
        self.Transformer_traindatasets = torch.Tensor(Transformer_traindata) #torch.squeeze(torch.Tensor(traindata))
        self.truedatasets = torch.Tensor(truedata) #torch.squeeze(torch.Tensor(trued
        print(self.truedatasets.shape)
        print(self.Transformer_traindatasets.shape)
        print(self.CNN_traindatasets.shape)
        self.CNN_shape = self.CNN_traindatasets.shape
        self.Transformer_shape = self.Transformer_traindatasets.shape
    def __getitem__(self, index):  # 'Generates one sample of data'
        # Select sample
        CNN_traindata = self.CNN_traindatasets[index, :, :]
        Transformer_traindata = self.Transformer_traindatasets[index, :, :]
        truedata = self.truedatasets[index]
        return CNN_traindata, Transformer_traindata, truedata
        # Load data and get label
    def __len__(self):  # 'Denotes the total number of samples'
        return self.CNN_traindatasets.shape[0]  # Return the total number of dataset

class CNN_Transformer_Dataset_Val(torch.utils.data.Dataset):  # 'Characterizes a dataset for PyTorch'
    '''
    This class is for validation datasets/ estimation datasets
    '''
    def __init__(self, CNN_traindata, Transformer_traindata):  # 'Initialization' Data Loading
            super(CNN_Transformer_Dataset_Val, self).__init__()
            self.CNN_traindatasets = torch.Tensor(CNN_traindata) #torch.squeeze(torch.Tensor(traindata))
            self.Transformer_traindatasets = torch.Tensor(Transformer_traindata) #torch.squeeze(torch.Tensor(traindata))
            print(self.Transformer_traindatasets.shape)
            print(self.CNN_traindatasets.shape)
            self.CNN_shape = self.CNN_traindatasets.shape
            self.Transformer_shape = self.Transformer_traindatasets.shape
    def __getitem__(self, index):  # 'Generates one sample of data'
            # Select sample
            CNN_traindata = self.CNN_traindatasets[index, :, :]
            Transformer_traindata = self.Transformer_traindatasets[index, :, :]
            return CNN_traindata, Transformer_traindata
            # Load data 
    def __len__(self):  # 'Denotes the total number of samples'
            return self.CNN_traindatasets.shape[0]  # Return the total number of datasets
    

class Dataset(torch.utils.data.Dataset):  # 'Characterizes a dataset for PyTorch'
    '''
    This class is for training datasets. It is used for the global datasets, which is continuous data.
    '''
    def __init__(self, traindata, truedata):  # 'Initialization' Data Loading
        '''

        :param traindata:
            Training data.
        :param truedata:
            Ture data to learn.
        :param beginyear:
            The begin year.
        :param endyear:
            The end year.
        :param nsite:
            The number of sites. For example, for overall observation it is 10870.
        '''
        super(Dataset, self).__init__()
        # 强制复制到RAM，避免memmap的磁盘I/O
        # 如果已经是shared tensor，直接用，不再复制
        if isinstance(traindata, torch.Tensor):
            self.traindatasets = traindata  # 零拷贝
            self.truedatasets  = truedata
        else:
            # 原来的numpy路径
            self.traindatasets = torch.from_numpy(
                traindata if traindata.dtype == np.float32
                else traindata.astype(np.float32))
            self.truedatasets = torch.from_numpy(
                truedata if truedata.dtype == np.float32
                else truedata.astype(np.float32))
            
        print(self.truedatasets.shape)
        print(self.traindatasets.shape)
        self.shape = self.traindatasets.shape
    def __getitem__(self, index):  # 'Generates one sample of data'
        # Select sample
        traindata = self.traindatasets[index, :, :]
        truedata = self.truedatasets[index]
        return traindata, truedata
        # Load data and get label
    def __len__(self):  # 'Denotes the total number of samples'
        return self.traindatasets.shape[0]  # Return the total number of dataset
    

class Dataset_Val(torch.utils.data.Dataset):  # 'Characterizes a dataset for PyTorch'
    '''
    This class is for validation datasets/ estimation datasets
    '''
    def __init__(self, traindata):  # 'Initialization' Data Loading
            super(Dataset_Val, self).__init__()
            self.traindatasets = torch.Tensor(traindata) #torch.squeeze(torch.Tensor(traindata))
            print(self.traindatasets.shape)
            self.shape = self.traindatasets.shape
    def __getitem__(self, index):  # 'Generates one sample of data'
            # Select sample
            traindata = self.traindatasets[index, :, :]
            return traindata
            # Load data 
    def __len__(self):  # 'Denotes the total number of samples'
            return self.traindatasets.shape[0]  # Return the total number of datasets
    
