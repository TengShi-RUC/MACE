import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from config import *
from data.dataloader import getDataLoader
from models import DIN, DMF, DeepModel


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


setup_seed(2021)

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--useSensitiveFeature", action="store_true")
parser.add_argument("--algos", type=str, default="deep_mixup(GF)")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--groupFeature", type=str, default="gender")
parser.add_argument("--dataset", type=str, default="insurance")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--fairWeight", type=float, default=0.02)
parser.add_argument("--discriminatorWeight", type=float, default=10)
parser.add_argument("--analysis", action="store_true")
parser.add_argument("--train_h1", type=float, default=1.0)
parser.add_argument("--train_h2", type=float, default=0.0)
parser.add_argument("--test_h1", type=float, default=1.0)
parser.add_argument("--test_h2", type=float, default=0.0)
args = parser.parse_args()
args.train = False
args.analysis = False
args.useSensitiveFeature = True
bestEpoch = 10

config = DeepModel_Config(args, GF_mixup=True, mixup_method="mixup")
model = DeepModel.DeepModel(config)
modelInfo = "epoch:{}".format(bestEpoch)
model._load_model(config.modelPath, modelInfo)
print("load model from {}{}.pth".format(config.modelPath, modelInfo))
model.eval()
use_cuda = False if config.device == "cpu" else True
test_dataloader = getDataLoader(
    config.test_batch_size,
    train=False,
    use_cuda=use_cuda,
    algo=args.algos,
    dataSetName=config.dataset,
)
CGF_DP_OUT_MCGF, CGF_EO_OUT_MCGF, CGF_DP_AUC_MCGF = model.getDPandEO(
    test_dataloader)
final_auc = np.mean(model.evaluate(test_dataloader))
print("AUC:{}".format(final_auc))


userProfileDict = pd.read_csv(
    "data/insurance/userProfile.csv").drop(columns=['user_id']).values
dataSet = np.load('data/insurance/interactTest.npy', allow_pickle=True)
userProfile = np.take(userProfileDict, dataSet[:, 0], axis=0)
inputData = np.concatenate((dataSet, userProfile), axis=1)
label_list = [0, 1, 2]
testSplitLabel = np.random.choice(label_list, size=(61338,))


class interactDataSet(IterableDataset):
    def __init__(self, label):
        super().__init__()
        self.inputData = inputData[testSplitLabel == label]
        self.dataSize = len(self.inputData)
        self.inputData = torch.LongTensor(self.inputData)
        print("=====load dataset label:{} length:{}=====".format(
            label, self.dataSize))

    def __len__(self):
        return self.dataSize

    def __iter__(self):
        for i in range(self.dataSize):
            yield self.inputData[i][0], self.inputData[i][3:], self.inputData[i][2], self.inputData[i][1]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(2021)


for label in label_list:
    print("=====label:{}=====".format(label))
    dataSet = interactDataSet(label=label)
    dataLoader = DataLoader(dataSet, batch_size=config.test_batch_size,
                            prefetch_factor=2, shuffle=False, pin_memory=use_cuda,
                            num_workers=0, worker_init_fn=seed_worker, generator=g)
    auc_score_list = model.evaluate(dataLoader)
    print("MeanAuc:{}".format(np.mean(auc_score_list)))
    model.getDPandEO(dataLoader)
