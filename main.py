import argparse
import random

import numpy as np
import torch

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
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--groupFeature", type=str, default="age")
parser.add_argument("--dataset", type=str, default="rentTheRunWay")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--d_num", type=int, default=6)
parser.add_argument("--fairWeight", type=float, default=1)
parser.add_argument("--discriminatorWeight", type=float, default=10)
parser.add_argument("--initWeight", type=float, default=0.6)

parser.add_argument("--analysis", action="store_true")
parser.add_argument("--train_h1", type=float, default=1)
parser.add_argument("--train_h2", type=float, default=0)
parser.add_argument("--test_h1", type=float, default=1)
parser.add_argument("--test_h2", type=float, default=0)
args = parser.parse_args()

# args.train = True
# args.useSensitiveFeature = True

print()
print(args)
if args.algos == "din":
    config = DIN_Config(args)
elif args.algos == "din_AdvLearning":
    config = DIN_Config(args, AdvLearning=True)
elif args.algos == "din_GapReg(GF)":
    config = DIN_Config(args, GF_mixup=True, mixup_method="GapReg")
elif args.algos == "din_mixup(GF)":
    config = DIN_Config(args, GF_mixup=True, mixup_method="mixup")
elif args.algos == "din_GapReg(CGF)_IV_MI":
    config = DIN_Config(
        args, CGF_mixup=True, mixup_method="GapReg", use_iv=True, use_MI=True
    )
elif args.algos == "din_mixup(CGF)_randomIV":
    config = DIN_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=True, random_iv=True
    )
elif args.algos == "din_mixup(GF)_IV_MI":
    config = DIN_Config(
        args, GF_mixup=True, mixup_method="mixup", use_iv=True, use_MI=True
    )
elif args.algos == "din_mixup(CGF)_IV_MI_directGexo":
    config = DIN_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=True, use_MI=True, direct_use_gexo=True
    )
elif args.algos == "din_mixup(CGF)_IV_MI":
    config = DIN_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=True, use_MI=True
    )
elif args.algos == "dmf":
    config = DMF_Config(args)
elif args.algos == "dmf_AdvLearning":
    config = DMF_Config(args, AdvLearning=True)
elif args.algos == "dmf_GapReg(GF)":
    config = DMF_Config(args, GF_mixup=True, mixup_method="GapReg")
elif args.algos == "dmf_mixup(GF)":
    config = DMF_Config(args, GF_mixup=True, mixup_method="mixup")
elif args.algos == "dmf_GapReg(CGF)_IV_MI":
    config = DMF_Config(
        args, CGF_mixup=True, mixup_method="GapReg", use_iv=True, use_MI=True
    )
elif args.algos == "dmf_mixup(CGF)_MI":  # Directly using exogenous part
    config = DMF_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=False, use_MI=True
    )
elif args.algos == "dmf_mixup(CGF)_randomIV":
    config = DMF_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=True, random_iv=True
    )
elif args.algos == "dmf_mixup(GF)_IV_MI":
    config = DMF_Config(
        args, GF_mixup=True, mixup_method="mixup", use_iv=True, use_MI=True
    )
elif args.algos == "dmf_mixup(CGF)_IV_MI_directGexo":
    config = DMF_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=True, use_MI=True, direct_use_gexo=True
    )
elif args.algos == "dmf_mixup(CGF)_IV_MI":
    config = DMF_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=True, use_MI=True
    )
elif args.algos == "deep":
    config = DeepModel_Config(args)
elif args.algos == "deep_AdvLearning":
    config = DeepModel_Config(args, AdvLearning=True)
elif args.algos == "deep_GapReg(GF)":
    config = DeepModel_Config(args, GF_mixup=True, mixup_method="GapReg")
elif args.algos == "deep_mixup(GF)":
    config = DeepModel_Config(args, GF_mixup=True, mixup_method="mixup")
elif args.algos == "deep_GapReg(CGF)_IV_MI":
    config = DeepModel_Config(
        args, CGF_mixup=True, mixup_method="GapReg", use_iv=True, use_MI=True
    )
elif args.algos == "deep_mixup(CGF)_MI":  # Directly using exogenous part
    config = DeepModel_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=False, use_MI=True
    )
elif args.algos == "deep_mixup(CGF)_randomIV":
    config = DeepModel_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=True, random_iv=True
    )
elif args.algos == "deep_mixup(GF)_IV_MI":
    config = DeepModel_Config(
        args, GF_mixup=True, mixup_method="mixup", use_iv=True, use_MI=True
    )
elif args.algos == "deep_mixup(CGF)_IV_MI":
    config = DeepModel_Config(
        args, CGF_mixup=True, mixup_method="mixup", use_iv=True, use_MI=True
    )
else:
    raise ValueError("algo error!")

if args.algos.startswith("din"):
    model = DIN.DeepInterestNetwork(config)
elif args.algos.startswith("dmf"):
    model = DMF.DeepMatrixFactorization(config)
elif args.algos.startswith("deep"):
    model = DeepModel.DeepModel(config)

use_cuda = False if config.device == "cpu" else True

print("load testSet")
test_dataloader = getDataLoader(
    config.test_batch_size,
    train=False,
    use_cuda=use_cuda,
    algo=args.algos,
    dataSetName=config.dataset,
)

if config.train:
    print("load trainSet")
    train_dataloader = getDataLoader(
        config.train_batch_size,
        train=True,
        use_cuda=use_cuda,
        algo=args.algos,
        dataSetName=config.dataset,
    )
    model.fit(config, train_dataloader, test_dataloader)
else:
    model.test_model(test_dataloader, config)
# else:
#     model._load_model(config.modelPath, config.modelInfo)
#     model.eval()
#     if config.fair:
#         model.getTestMIvalue(test_dataloader)
#     model.evaluate(test_dataloader)
#     model.getDPandEO(test_dataloader)
