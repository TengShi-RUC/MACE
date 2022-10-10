import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import *
from data.dataloader import getDataLoader
from models import DMF
from models.modules.mi_estimators import *


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
parser.add_argument("--algos", type=str, default="dmf")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--groupFeature", type=str, default="gender")
parser.add_argument("--dataset", type=str, default="insurance")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--fairWeight", type=float, default=0.86)
parser.add_argument("--discriminatorWeight", type=float, default=10)
parser.add_argument("--analysis", action="store_true")
parser.add_argument("--train_h1", type=float, default=1.0)
parser.add_argument("--train_h2", type=float, default=0.0)
parser.add_argument("--test_h1", type=float, default=1.0)
parser.add_argument("--test_h2", type=float, default=0.0)
args = parser.parse_args()
args.train = False
args.analysis = False
args.useSensitiveFeature = False
bestEpoch = 13


config = DMF_Config(args)
model = DMF.DeepMatrixFactorization(config)
modelInfo = "epoch:{}".format(bestEpoch)
model._load_model(config.modelPath, modelInfo)
print("load model from {}{}.pth".format(config.modelPath, modelInfo))
model.eval()

use_cuda = False if config.device == "cpu" else True
print("load trainSet")
train_dataloader = getDataLoader(config.train_batch_size, train=True,
                                 use_cuda=use_cuda, algo=args.algos, dataSetName=config.dataset)


mi_estimator = CLUBSample(64, config.sensitiveFeatureNum,
                          hidden_size=32).to(config.device)
optimizer = torch.optim.Adam(
    mi_estimator.parameters(), lr=1e-4)
mi_est_values = []
for epoch in range(100):
    tqdm_ = tqdm(iterable=train_dataloader, mininterval=1, ncols=120)
    last_MI_est = -1
    for step, batch_data in enumerate(tqdm_):
        userID, userProfile, label, otherInfo = batch_data
        userID = userID.to(config.device)
        userProfile = userProfile.to(config.device)
        otherInfo = otherInfo.to(config.device)
        data = (userID, userProfile, otherInfo)

        # userEmbedding = model.getuserEmbedding(data)
        userEmbedding = model.userMLP(model.user_embedding(userID))

        mi_estimator.eval()
        cur_mi_est = mi_estimator(
            userEmbedding, userProfile).detach().cpu().item()
        if step % 100 == 0:
            tqdm_.set_description("epoch {:d},step {:d},MI_est {:.4f}".format(
                epoch + 1, step, cur_mi_est))
        # mi_est_values.append(cur_mi_est)
        last_MI_est = cur_mi_est

        mi_estimator.train()
        mi_loss = mi_estimator.learning_loss(userEmbedding, userProfile)

        optimizer.zero_grad()
        mi_loss.backward()
        optimizer.step()

    mi_est_values.append(last_MI_est)


plt.plot(np.arange(1, len(mi_est_values) + 1), mi_est_values)
plt.savefig('MI_estimation.png', dpi=400)
