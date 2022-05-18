import argparse
import random

import numpy as np
import torch
from tqdm import tqdm

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
parser.add_argument("--algos", type=str, default="dmf_mixup(CGF)_IV_MI")
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--groupFeature", type=str, default="occupation")
parser.add_argument("--dataset", type=str, default="ml-1m")
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
args.analysis = True
args.useSensitiveFeature = True
bestEpoch = 19

config = DMF_Config(args, CGF_mixup=True,
                    mixup_method="mixup", use_iv=True, use_MI=True)
model = DMF.DeepMatrixFactorization(config)

use_cuda = False if config.device == "cpu" else True

print("load testSet")
dataloader = getDataLoader(
    config.test_batch_size,
    train=False,
    use_cuda=use_cuda,
    algo=args.algos,
    dataSetName=config.dataset,
)

modelInfo = "epoch:{}".format(bestEpoch)
model._load_model(config.modelPath, modelInfo)
print("load model from {}{}.pth".format(config.modelPath, modelInfo))
model.eval()
CGF_DP_OUT_MCGF, CGF_EO_OUT_MCGF, CGF_DP_AUC_MCGF = model.getDPandEO(
    dataloader)
final_auc = np.mean(model.evaluate(dataloader))
print("AUC:{}".format(final_auc))
