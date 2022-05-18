import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


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


def process_ml1m():
    def changeRateToLabel(x):
        if x == 4 or x == 5:
            return 1
        else:
            return 0
    user_df = pd.read_csv(
        "data/ml-1m/users.csv").sort_values(by=['user_id']).drop(columns=['zip']).set_index('user_id')
    user_df = user_df.astype({'gender': 'category',
                              'age': 'category',
                              'occupation': 'category'})
    user_df['gender'] = user_df['gender'].cat.codes.values
    user_df['age'] = user_df['age'].cat.codes.values
    user_df['occupation'] = user_df['occupation'].cat.codes.values
    print("gender_class:{}\n age_class:{}\n occupation_class:{}".format(
        sorted(user_df['gender'].unique()),
        sorted(user_df['age'].unique()),
        sorted(user_df['occupation'].unique())))
    user_df.to_csv("data/ml-1m/userProfile.csv")

    data = pd.read_csv(
        "data/ml-1m/ratings.csv").sort_values(by=['user_id', 'timestamp'])
    data['label'] = data['rating'].apply(
        changeRateToLabel).drop(columns=['rating'])
    max_seq_len = 300

    dataSize = len(data)
    numUser = data['user_id'].unique().max()
    numItem = data['movie_id'].unique().max()
    print("userNum:{} itemNum:{}".format(numUser, numItem))
    seqTrainSet = []
    seqTestSet = []
    userHistory = []
    interactTrainSet = []
    interactTestSet = []
    interactMatrix = np.zeros((numUser, numItem))
    for i in tqdm(range(dataSize)):
        movie_id = data['movie_id'].iloc[i] - 1
        user_id = data['user_id'].iloc[i] - 1
        label = data['label'].iloc[i]
        interactMatrix[user_id][movie_id] = label
        if (i == dataSize - 1) or (data['user_id'].iloc[i + 1] != data['user_id'].iloc[i]):
            mode = 'test'
            interactTestSet.append((user_id, movie_id, label))
        else:
            mode = 'train'
            interactTrainSet.append((user_id, movie_id, label))

        if len(userHistory) == 0:
            userHistory.append(movie_id)
            continue

        user_behaviors = userHistory.copy(
        ) + [0] * (max_seq_len - len(userHistory))
        if len(userHistory) == max_seq_len:
            userHistory.pop(0)
        userHistory.append(movie_id)

        curData = [user_id]
        curData.extend(user_behaviors)
        curData.extend([movie_id, label])
        if mode == 'train':
            seqTrainSet.append(curData)
        else:
            seqTestSet.append(curData)
            userHistory = []

    seqTrainMatrix = np.array(seqTrainSet)
    seqTestMatrix = np.array(seqTestSet)
    print("seqTrainSize: ", seqTrainMatrix.shape[0])
    print("seqTestSize: ", seqTestMatrix.shape[0])
    np.save("data/ml-1m/seqTrain.npy", seqTrainMatrix)
    np.save("data/ml-1m/seqTest.npy", seqTestMatrix)

    interactTrainMatrix = np.array(interactTrainSet)
    interactTestMatrix = np.array(interactTestSet)
    print("interactTrainSize: ", interactTrainMatrix.shape[0])
    print("interactTestSize: ", interactTestMatrix.shape[0])
    np.save("data/ml-1m/interactTrain.npy", interactTrainMatrix)
    np.save("data/ml-1m/interactTest.npy", interactTestMatrix)
    np.save("data/ml-1m/interactMatrix.npy", interactMatrix)


def process_insurance():
    insurance = pd.read_csv("data/insurance/Train.csv").drop(
        columns=['join_date', 'birth_year', 'branch_code', 'occupation_code'])
    insurance = insurance.astype(
        {'ID': 'category', 'sex': 'category', 'marital_status': 'category', 'occupation_category_code': 'category'})
    insurance = insurance.rename(columns={"ID": "user_id",
                                          "sex": "gender",
                                          "occupation_category_code": "occupation"})
    insurance.to_csv("data/insurance/labels.csv", index=False)

    numUser = len(insurance)
    numItem = len(insurance.columns) - 4
    print("userNum:{} itemNum:{}".format(numUser, numItem))

    insurance['user_id'] = insurance['user_id'].cat.codes.values
    insurance['gender'] = insurance['gender'].cat.codes.values
    insurance['marital_status'] = insurance['marital_status'].cat.codes.values
    insurance['occupation'] = insurance['occupation'].cat.codes.values
    print("gender_class:{}\n marital_status_class:{}\n occupation_class:{}".format(
        sorted(insurance['gender'].unique()),
        sorted(insurance['marital_status'].unique()),
        sorted(insurance['occupation'].unique())))
    insurance = insurance.sort_values(by=['user_id'])
    userProfile = insurance[['user_id', 'gender', 'marital_status',
                             'occupation']].set_index('user_id')
    userProfile.to_csv("data/insurance/userProfile.csv")
    labelData = insurance.iloc[:, 4:]
    interactMatrix = labelData.values

    dataSize = numUser * numItem
    maskArray = np.random.choice(
        [False, True], size=(numUser, numItem), p=[0.1, 0.9])
    interactTrainSet = []
    interactTestSet = []
    for i in range(numUser):
        for j in range(numItem):
            if maskArray[i][j]:
                interactTrainSet.append((i, j, interactMatrix[i][j]))
            else:
                interactTestSet.append((i, j, interactMatrix[i][j]))
    interactTrainMatrix = np.array(interactTrainSet)
    interactTestMatrix = np.array(interactTestSet)
    print("interactTrainSize: ", interactTrainMatrix.shape[0])
    print("interactTestSize: ", interactTestMatrix.shape[0])
    np.save("data/insurance/interactTrain.npy", interactTrainMatrix)
    np.save("data/insurance/interactTest.npy", interactTestMatrix)
    np.save("data/insurance/interactMatrix.npy", interactMatrix)


print("ml-1m")
process_ml1m()
print("insurance")
process_insurance()
