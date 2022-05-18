import numpy as np
import pandas as pd
import torch
from config import *
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import random


class interactDataSet(IterableDataset):
    def __init__(self, train, dataSetName):
        super().__init__()
        if dataSetName == 'ml-1m':
            userProfileDict = pd.read_csv(
                "data/ml-1m/userProfile.csv").drop(columns=['user_id']).values
            if train:
                dataPath = 'data/ml-1m/interactTrain.npy'
            else:
                dataPath = 'data/ml-1m/interactTest.npy'
        else:
            userProfileDict = pd.read_csv(
                "data/insurance/userProfile.csv").drop(columns=['user_id']).values
            if train:
                dataPath = 'data/insurance/interactTrain.npy'
            else:
                dataPath = 'data/insurance/interactTest.npy'

        self.dataSet = np.load(dataPath, allow_pickle=True)
        userProfile = np.take(userProfileDict, self.dataSet[:, 0], axis=0)
        self.inputData = np.concatenate((self.dataSet, userProfile), axis=1)
        self.dataSize = len(self.inputData)
        self.inputData = torch.LongTensor(self.inputData)

    def __len__(self):
        return self.dataSize

    def __iter__(self):
        for i in range(self.dataSize):
            yield self.inputData[i][0], self.inputData[i][3:], self.inputData[i][2], self.inputData[i][1]

# userID,userProfile,label,other(itemID)


class sequenceDataSet(IterableDataset):
    def __init__(self, train):
        super().__init__()
        if train:
            dataPath = 'data/ml-1m/seqTrain.npy'
        else:
            dataPath = 'data/ml-1m/seqTest.npy'
        userProfileDict = pd.read_csv(
            "data/ml-1m/userProfile.csv").drop(columns=['user_id']).values

        self.dataSet = np.load(dataPath, allow_pickle=True)
        userProfile = np.take(userProfileDict, self.dataSet[:, 0], axis=0)
        self.inputData = np.concatenate((self.dataSet, userProfile), axis=1)
        self.dataSize = self.inputData.shape[0]
        self.inputData = torch.LongTensor(self.inputData)

    def __len__(self):
        return self.dataSize

    def __iter__(self):
        for i in range(self.dataSize):
            yield self.inputData[i][0], self.inputData[i][303:],  self.inputData[i][302], self.inputData[i][1:302]

# userID,userProfile,label,other(userBehavior + targetItem)


def getDataLoader(batchSize, train, use_cuda, algo, dataSetName):
    if algo.startswith("din"):
        dataSet = sequenceDataSet(train=train)
    else:
        dataSet = interactDataSet(train=train, dataSetName=dataSetName)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(2021)

    dataLoader = DataLoader(dataSet, batch_size=batchSize,
                            prefetch_factor=2, shuffle=False, pin_memory=use_cuda,
                            num_workers=0, worker_init_fn=seed_worker, generator=g)
    return dataLoader
