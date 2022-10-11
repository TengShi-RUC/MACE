import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from models.Discriminators import *
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class BasicModel(nn.Module):
    def __init__(self, config: BaseConfig):
        """
        basic model:
            create model and optimizer
            initialize model and hyper-parameter
        """
        super().__init__()
        self.device = config.device
        self.name = config.description
        self.regularization_weight = []
        self.useSensitiveFeature = config.useSensitiveFeature
        self.sensitiveFeatureNum = config.sensitiveFeatureNum
        self.use_iv = config.use_iv
        self.ridge_lambd = config.ridge_lambd
        self.use_MI = config.use_MI
        self.mixup_group_size = config.mixup_group_size
        self.userEmbeddingSize = config.userEmbeddingSize
        self.random_iv = config.random_iv

        self.AdvLearning = config.AdvLearning
        if self.AdvLearning:
            self.init_sensitive_filter()
            self.discriminator_dict = {}
            self.discriminator_optimizer = {}
            for feature, num_class in config.sensitiveFeatureClass.items():
                disc = Discriminator(self.userEmbeddingSize,
                                     num_class, self.device)
                self.discriminator_dict[feature] = disc
                self.discriminator_optimizer[feature] = torch.optim.Adam(
                    disc.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.l2_penalty,
                )

            # Discriminator

        self.dataset = config.dataset
        if self.dataset == "ml-1m":
            self.user_df = (
                pd.read_csv("data/ml-1m/userProfile.csv")
                .sort_values(by=["user_id"])
                .set_index("user_id")
            )
        elif self.dataset == 'insurance':
            self.user_df = (
                pd.read_csv("data/insurance/userProfile.csv")
                .sort_values(by=["user_id"])
                .set_index("user_id")
            )
        elif self.dataset == 'rentTheRunWay':
            self.user_df = (
                pd.read_csv("data/rentTheRunWay/userProfile.csv")
                .sort_values(by=["user_id"])
                .set_index("user_id")
            )
        else:
            raise ValueError("Dataset error")
        self.user_df = self.user_df.astype("category")
        self.userProfileDict = self.user_df.values

        self.groupFeature = config.groupFeature
        index2feature = self.user_df.columns.to_list()
        self.feature2index = {
            index2feature[i]: i for i in range(len(index2feature))}
        self.featureIndex = self.feature2index[self.groupFeature]
        print("groupFeature:{} index:{}".format(
            self.groupFeature, self.featureIndex))

        self.d_num = config.d_num

        self.allDistribution, self.featureValueList = self.getAllDistribution(
            d_num=self.d_num)
        self.testDistribution, self.infoDict = self.getTestDistribution()

        numDistribution = self.allDistribution.size()[1]
        #self.distribution_weight = [0.08, 0.08, 0.08, 0.08, 0.08, 0.6]
        # self.distribution_weight = np.ones(
        #     (numDistribution,)) / (numDistribution * 1.0)
        if numDistribution == 1:
            self.distribution_weight = [1]
        else:
            self.distribution_weight = np.zeros((numDistribution,))
            init_weight = config.initWeight
            self.distribution_weight[0] = init_weight
            self.distribution_weight[1:] = (
                1 - init_weight) / (numDistribution - 1)
        print("distribution_weight:{}".format(self.distribution_weight))

        self.analysis = config.analysis
        self.direct_use_gexo = config.direct_use_gexo
        if config.analysis:
            self.train_h1 = config.train_h1
            self.train_h2 = config.train_h2
            self.test_h1 = config.test_h1
            self.test_h2 = config.test_h2
            print("train_h1:{} train_h2:{} test_h1:{} test_h2:{}".format(self.train_h1,
                                                                         self.train_h2, self.test_h1, self.test_h2))
        self.running_mode = 'test'

    def _save_model(self, info, model_save_path):
        """
            save model
        """
        final_path = "%s%s.pth" % (model_save_path, info)
        print("==========save model to: {}==========".format(final_path))
        torch.save(self.state_dict(), final_path)

    def _load_model(self, load_path, info):
        """
            loading model
        """
        final_path = "%s%s.pth" % (load_path, info)
        print("==========load model from: {}==========".format(final_path))
        self.load_state_dict(
            torch.load(final_path,
                       map_location=self.device)
        )

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        """
            calculate regularization loss(l1 or l2)
        """
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 *
                                                    torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def getAllDistribution(self, d_num):
        userProfile = self.user_df
        userNum = len(userProfile)
        featureValueCounts = userProfile[self.groupFeature].value_counts()
        featureValueDict = (featureValueCounts /
                            np.sum(featureValueCounts)).to_dict()
        featureValueDict = {int(k): v for k, v in featureValueDict.items()}
        initDistribution = np.array(
            [v[1]
                for v in sorted(featureValueDict.items(), key=lambda x: x[0])]
        )
        # for x in initDistribution:
        #     print(x)
        numFeature = len(initDistribution)
        featureValueList = list(range(numFeature))
        uniformDistribution = np.array(
            [1 / numFeature for _ in range(numFeature)])

        t_value_list = np.linspace(0, 1, num=d_num)
        allDistribution = torch.empty(
            (userNum, len(t_value_list)), dtype=torch.int64)
        for i in range(len(t_value_list)):
            t = t_value_list[i]
            curDistribution = (1 - t) * initDistribution + \
                t * uniformDistribution
            curUserFeature = np.random.choice(
                featureValueList, (userNum,), p=curDistribution
            )
            allDistribution[:, i] = torch.LongTensor(curUserFeature.copy())
        return allDistribution, featureValueList

    def getTestDistribution(self):
        userProfile = self.user_df
        userNum = len(userProfile)

        featureList = list(
            range(
                len(userProfile[self.groupFeature].cat.categories.tolist()))
        )
        numTestDistribution = 4
        testDistribution = np.empty(
            (userNum, numTestDistribution), dtype=np.int)
        testDistribution[:, 0] = np.random.choice(featureList, (userNum,))
        testDistribution[:, 3] = userProfile[
            self.groupFeature
        ].cat.codes.values.reshape((-1,))
        infoDict = {0: "uniform", 1: "extreme1", 2: "extreme2", 3: "init"}

        if self.dataset == "ml-1m":
            if self.groupFeature == "gender":
                testDistribution[:, 1] = np.random.choice(
                    featureList, (userNum,), p=[0.1, 0.9]
                )
                testDistribution[:, 2] = np.random.choice(
                    featureList, (userNum,), p=[0.9, 0.1]
                )
            elif self.groupFeature == "occupation":
                female_mask = self.userProfileDict[:, 0] == 0
                num_Female = female_mask.sum()
                female_sample = np.random.choice(
                    [15, 11, 19, 18, 8], (num_Female,))
                testDistribution[:, 1][female_mask] = female_sample

                male_mask = self.userProfileDict[:, 0] == 1
                num_Male = male_mask.sum()
                male_sample = np.random.choice([3, 18, 19, 8, 9], (num_Male,))
                testDistribution[:, 1][male_mask] = male_sample

                probability = [0.03, 0.03, 0.03, 0.03, 0.03,
                               0.03, 0.03, 0.03, 0.03, 0.03,
                               0.03, 0.03, 0.03, 0.03, 0.03,
                               0.03, 0.03, 0.03, 0.03, 0.03, 0.4]
                testDistribution[:, 2] = np.random.choice(
                    featureList, (userNum,), p=probability)

                # probability = [0, 0, 0, 0, 0,
                #                0, 0, 0, 0, 0,
                #                0, 0, 0, 0, 0,
                #                0, 1, 0, 0, 0, 0]
            # elif self.groupFeature == 'age':
            #     probability = [0.1, 0.1, 0.1,
            #                    0.1, 0.1, 0.1, 0.4]

        elif self.dataset == "insurance":
            if self.groupFeature == "gender":
                testDistribution[:, 1] = np.random.choice(
                    featureList, (userNum,), p=[0.1, 0.9]
                )
                testDistribution[:, 2] = np.random.choice(
                    featureList, (userNum,), p=[0.9, 0.1]
                )
        elif self.dataset == 'rentTheRunWay':
            testDistribution[:, 1] = np.random.choice(featureList, (userNum,), p=[
                                                      0.1, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
            testDistribution[:, 2] = np.random.choice(featureList, (userNum,), p=[
                                                      0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1, 0.1])
        return torch.LongTensor(testDistribution), infoDict

    def init_sensitive_filter(self, filter_mode="combine"):
        def get_sensitive_filter(embed_dim):
            sequential = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(embed_dim),
            )
            return sequential

        num_features = self.sensitiveFeatureNum
        filter_num = num_features if filter_mode == "combine" else 2 ** num_features
        self.filter_dict = nn.ModuleDict(
            {
                str(i + 1): get_sensitive_filter(self.userEmbeddingSize)
                for i in range(filter_num)
            }
        )

    def applyFilter(self, userEmbedding):
        result = None
        for idx, filter in self.filter_dict.items():
            result = (
                filter(userEmbedding)
                if result is None
                else result + filter(userEmbedding)
            )
        result /= self.sensitiveFeatureNum
        return result

    def getuserEmbedding(self, x):
        pass

    def iv(self, userEmbedding):
        gexo = self.Gexo(userEmbedding)  # (batch_size, d_exo)
        if self.direct_use_gexo:
            return gexo
        gendo = self.Gendo(userEmbedding)  # (batch_size, d_endo)
        # solution = torch.lstsq(
        #     gendo, gexo).solution[:gexo.size(1)]  # (d_exo, d_endo) 
        # solution = gexo.pinverse(rcond=0).matmul(gendo) 
        # solution = gexo.pinverse().matmul(gendo)  # (d_exo, d_endo)
        gexo_T = gexo.T
        XTX = gexo_T.matmul(gexo)
        m = XTX.size()[0]
        I = torch.eye(m).to(self.device)
        solution = (
            torch.inverse(
                XTX + I * self.ridge_lambd).matmul(gexo_T).matmul(gendo)
        )

        g1 = torch.matmul(gexo, solution)  # (batch_size, d_endo)
        g2 = gendo - g1  # (batch_size, d_endo)


        if self.analysis:
            if self.running_mode == 'train':
                h1 = self.train_h1
                h2 = self.train_h2
            else:
                h1 = self.test_h1
                h2 = self.test_h2
        else:
            h_mlp_input = torch.cat([gendo, gexo], dim=-1)
            h1 = self.prob_sigmoid(self.h1_mlp(h_mlp_input))
            h2 = self.prob_sigmoid(self.h2_mlp(h_mlp_input))

        hx = h1 * g1 + h2 * g2  # (batch_size, d_endo)
        return hx

    def dp_loss_allDistribution(self, batch_data, method):
        userID, userProfile, label, otherInfo = batch_data
        numDistribution = self.allDistribution.size()[1]
        dp_loss = 0
        first = True
        for t in range(numDistribution):
            curUserFeature = self.allDistribution[:, t][userID]
            userProfile[:, self.featureIndex] = curUserFeature.view((-1,))
            cur_loss = (
                self.dp_loss_initDistribution(
                    (userID, userProfile, label, otherInfo), method
                )
                * self.distribution_weight[t]
            )
            if first:
                dp_loss = cur_loss
                first = False
            else:
                dp_loss += cur_loss
        return dp_loss

    def eo_loss_allDistribution(self):
        pass

    def dp_loss_initDistribution(self, batch_data, method):
        userID, userProfile, label, otherInfo = batch_data
        batch_group_input = []
        for i, feature in enumerate(self.featureValueList):
            mask = userProfile[:, self.featureIndex] == feature
            if mask.sum() == 0:
                continue
            index_list = torch.where(mask)[0]
            sample_index = index_list[
                torch.randint(0, len(index_list), (self.mixup_group_size,))
            ]
            # if mask.sum() <= 1:
            #     continue
            curUserProfile = userProfile[sample_index]
            curOtherInfo = otherInfo[sample_index]
            curUserID = userID[sample_index]
            batch_group_input.append((curUserID, curUserProfile, curOtherInfo))
        numGroups = len(batch_group_input)

        if method == "mixup": 
            final_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            first = True

            for i in range(numGroups):
                for j in range(i + 1, numGroups):
                    loss_reg = self.mixup_loss(
                        batch_group_input[i], batch_group_input[j]
                    )
                    if first:
                        final_loss = loss_reg
                        first = False
                    else:
                        if loss_reg > final_loss:
                            final_loss = loss_reg
            return final_loss

        elif method == "GapReg":
            out_list = []
            for curUserID, curUserProfile, curOtherInfo in batch_group_input:
                modelOut = self((curUserID, curUserProfile, curOtherInfo))
                out_list.append(modelOut.mean())
            maxTensor = out_list[0]
            minTensor = out_list[0]
            for i in range(1, len(out_list)):
                if out_list[i] > maxTensor:
                    maxTensor = out_list[i]
                if out_list[i] < minTensor:
                    minTensor = out_list[i]
            dp_loss = torch.square(maxTensor - minTensor)
            return dp_loss
        else:
            raise ValueError("mixup method error!")

    def set_discriminator_grad(self, grad):
        for _, discriminator in self.discriminator_dict.items():
            discriminator.requires_grad_(grad)

    def _run_train(
        self, trainLoader, testLoader, optimizer, loss_func, config: BaseConfig
    ):
        total_loss = 0
        total_steps = 0

        best_epoch = -1
        best_auc = -1
        best_CGF = float("inf")
        min_loss = float("inf")

        epoch_list = []
        # MI_est = []
        first_MI_list = []
        last_MI_list = []
        OutPredict_loss_list = []
        auc_list_0 = []
        auc_list_1 = []
        auc_list_2 = []
        mean_auc_list = []
        CGF_DP_OUT_MCGF_list = []
        CGF_EO_OUT_MCGF_list = []
        CGF_DP_AUC_MCGF_list = []

        for epoch in range(config.epochs):
            self.train()
            self.running_mode = 'train'
            epoch_loss = 0
            predictLoss = 0
            fairLoss = 0
            tqdm_ = tqdm(iterable=trainLoader, mininterval=1, ncols=120)
            print()
            print("=" * 20 + "epoch " + str(epoch + 1) + "=" * 20)
            print("*" * 5 + "minimizeOutPredict" + "*" * 5)
            for step, x in enumerate(tqdm_):
                if self.AdvLearning:
                    self.set_discriminator_grad(False)
                userID, userProfile, label, otherInfo = x
                userID = userID.to(self.device)
                userProfile = userProfile.to(self.device)
                label = label.to(self.device)
                otherInfo = otherInfo.to(self.device)

                data = (userID, userProfile, otherInfo)
                pred = self(data)

                loss = loss_func(pred, label.float())
                reg_loss = self.get_regularization_loss()
                all_loss = loss + reg_loss

                predictLoss += all_loss.item()

                if self.AdvLearning:
                    userEmbedding = self.getuserEmbedding(data)
                    for feature, discriminator in self.discriminator_dict.items():
                        cur_label = userProfile[:, self.feature2index[feature]]
                        cur_loss = discriminator(userEmbedding, cur_label)
                        all_loss -= config.discriminator_weight * cur_loss

                if config.GF_mixup:
                    dp_loss = (
                        self.dp_loss_initDistribution(
                            (userID, userProfile, label,
                             otherInfo), config.mixup_method
                        )
                        * config.fair_loss_weight
                    )
                elif config.CGF_mixup:
                    dp_loss = (
                        self.dp_loss_allDistribution(
                            (userID, userProfile, label,
                             otherInfo), config.mixup_method
                        )
                        * config.fair_loss_weight
                    )
                if config.GF_mixup or config.CGF_mixup:
                    all_loss += dp_loss
                    fairLoss += dp_loss.item()

                epoch_loss += all_loss.item()
                total_loss += all_loss.item()

                total_steps += 1

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                if self.AdvLearning:
                    self.requires_grad_(False)
                    self.set_discriminator_grad(True)
                    userEmbedding = self.getuserEmbedding(data)
                    for _ in range(config.Adv_discriminator_iter):
                        for feature, discriminator in self.discriminator_dict.items():
                            cur_label = userProfile[:,
                                                    self.feature2index[feature]]
                            cur_loss = discriminator(userEmbedding, cur_label)
                            self.discriminator_optimizer[feature].zero_grad()
                            cur_loss.backward()
                            self.discriminator_optimizer[feature].step()
                    self.requires_grad_(True)

                if step % config.interval == 0 and step > 0:
                    if config.GF_mixup or config.CGF_mixup:
                        tqdm_.set_description(
                            "epoch {:d},step {:d},predictLoss {:.4f},fairLoss {:.4f}".format(
                                epoch + 1, step, predictLoss / step, fairLoss / step
                            )
                        )
                    else:
                        tqdm_.set_description(
                            "epoch {:d},step {:d},predictLoss {:.4f}".format(
                                epoch + 1, step, predictLoss / step
                            )
                        )

            epoch_loss /= step
            print("OutPredict_loss:{}".format(epoch_loss))
            OutPredict_loss_list.append(epoch_loss)

            curModelInfo = "epoch:{}".format(epoch + 1)
            if config.CGF_mixup:
                curModelInfo += " dNum_{} init_{}".format(
                    self.d_num, config.initWeight)

            if config.use_MI:
                first_MI, last_MI = self.minimizeMI(trainLoader, config)
                first_MI_list.append(first_MI)
                last_MI_list.append(last_MI)
                # MI_est.append(last_MI)
                # epoch_loss += tempMI * config.MI_loss_weight
            # else:
            #     if epoch_loss < min_loss:
            #         min_loss = epoch_loss
            #         best_epoch = epoch + 1

            # print('totalLoss:{}'.format(epoch_loss))

            # if epoch_loss < min_loss:
            #     min_loss = epoch_loss
            #     best_epoch = epoch + 1
            #     self._save_model(config.modelInfo, config.modelPath)

            self._save_model(curModelInfo, config.modelPath)

            self.eval()
            self.running_mode = 'test'
            auc_score_list = self.evaluate(testLoader)
            meanAuc = np.mean(auc_score_list)
            print("Epoch:{} AUC:{}".format(epoch + 1, meanAuc))
            epoch_list.append(epoch + 1)
            auc_list_0.append(auc_score_list[0])
            auc_list_1.append(auc_score_list[1])
            auc_list_2.append(auc_score_list[2])
            mean_auc_list.append(meanAuc)
            CGF_DP_OUT_MCGF, CGF_EO_OUT_MCGF, CGF_DP_AUC_MCGF = self.getDPandEO(
                testLoader
            )
            CGF_DP_OUT_MCGF_list.append(CGF_DP_OUT_MCGF)
            CGF_EO_OUT_MCGF_list.append(CGF_EO_OUT_MCGF)
            CGF_DP_AUC_MCGF_list.append(CGF_DP_AUC_MCGF)

            if meanAuc > best_auc:
                best_auc = meanAuc
                best_epoch = epoch + 1
            # if epoch + 1 - best_epoch > config.earlyStop:
            #     print("Normal Early stop at epoch{}!".format(epoch + 1))
            #     break
        if config.use_MI:
            result_df = pd.DataFrame(
                {
                    "epoch": epoch_list,
                    "OutPredict_loss": OutPredict_loss_list,
                    "first_MI_est": first_MI_list,
                    "last_MI_est": last_MI_list,
                    "{}AUC".format(self.infoDict[0]): auc_list_0,
                    "{}AUC".format(self.infoDict[1]): auc_list_1,
                    "{}AUC".format(self.infoDict[2]): auc_list_2,
                    "meanAUC": mean_auc_list,
                    "CGF_DP_OUT_MCGF": CGF_DP_OUT_MCGF_list,
                    "CGF_EO_OUT_MCGF": CGF_EO_OUT_MCGF_list,
                    "CGF_DP_AUC_MCGF": CGF_DP_AUC_MCGF_list,
                }
            )
            result_df.to_csv(config.csvPath, index=False)

            x_label_len = len(last_MI_list)
            plt.plot(np.arange(1, x_label_len + 1),
                     last_MI_list,
                     label=config.MI_estimator_name + " optimize")
            plt.plot(np.arange(1, x_label_len + 1),
                     [first_MI_list[0]] * x_label_len,
                     label="unOptimize")
            plt.legend()
            plt.xticks(np.arange(1, x_label_len + 1))
            plt.savefig(config.MI_figure_save_path, dpi=400)

        # if not config.use_MI:
        if config.CGF_mixup:
            self._load_model(config.modelPath, "epoch:{} dNum_{} init_{}".format(
                best_epoch, config.d_num, config.initWeight))
        else:
            self._load_model(config.modelPath, "epoch:{}".format(best_epoch))
        self.eval()
        self.running_mode = 'test'
        final_auc = np.mean(self.evaluate(testLoader))

        print("Final AUC: {}, At Epoch {}".format(final_auc, best_epoch))
        self.getDPandEO(testLoader)

    def test_model(self, testLoader, config):
        best_epoch = -1
        best_CGF = float("inf")
        for epoch in range(1, 21):
            if config.CGF_mixup:
                self._load_model(config.modelPath, "epoch:{} dNum_{} init_{}".format(
                    epoch, config.d_num, config.initWeight))
            else:
                self._load_model(config.modelPath, "epoch:{}".format(epoch))
            self.eval()
            self.running_mode = 'test'
            final_auc = np.mean(self.evaluate(testLoader))

            print("AUC: {}, At Epoch {}".format(final_auc, epoch))
            CGF_DP_OUT_MCGF, CGF_EO_OUT_MCGF, CGF_DP_AUC_MCGF = self.getDPandEO(
                testLoader
            )
            if CGF_DP_OUT_MCGF < best_CGF:
                best_CGF = CGF_DP_OUT_MCGF
                best_epoch = epoch

        if config.CGF_mixup:
            self._load_model(config.modelPath, "epoch:{} dNum_{} init_{}".format(
                best_epoch, config.d_num, config.initWeight))
        else:
            self._load_model(config.modelPath, "epoch:{}".format(best_epoch))
        self.eval()
        self.running_mode = 'test'
        final_auc = np.mean(self.evaluate(testLoader))

        print("Final AUC: {}, At Epoch {}".format(final_auc, best_epoch))
        self.getDPandEO(testLoader)

    def fit(self, config, trainLoader, testLoader):
        self.train()

        loss_func = nn.BCELoss()
        optimizer = optim.Adam(
            self.parameters(), lr=config.learning_rate, weight_decay=config.l2_penalty
        )

        self._run_train(
            trainLoader=trainLoader,
            testLoader=testLoader,
            optimizer=optimizer,
            loss_func=loss_func,
            config=config,
        )

    def _run_eval(self, dataloader):
        tqdm_ = tqdm(iterable=dataloader, mininterval=1, ncols=120)
        numDistribution = self.testDistribution.size()[1]

        auc_score_list = []

        for i in range(numDistribution):
            curUserFeature = self.testDistribution[:, i]
            label_list = []
            pred_score = []
            for batch in tqdm_:
                userID, userProfile, label, otherInfo = batch
                tempUserFeature = curUserFeature[userID]
                userProfile[:, self.featureIndex] = tempUserFeature.view((-1,))

                userID = userID.to(self.device)
                userProfile = userProfile.to(self.device)
                otherInfo = otherInfo.to(self.device)

                data = (userID, userProfile, otherInfo)
                pred = self(data).detach().cpu().numpy().tolist()

                label_list.extend(label.numpy().tolist())
                pred_score.extend(pred)

            try:
                auc_score = roc_auc_score(
                    np.array(label_list), np.array(pred_score))
                print("{} AUC:{}".format(self.infoDict[i], auc_score))
                auc_score_list.append(auc_score)
            except:
                pass
        return auc_score_list

    def evaluate(self, dataloader):
        self.eval()
        self.running_mode = 'test'
        concatStr = "*" * 5
        print(concatStr + "evaluating" + concatStr)

        return self._run_eval(dataloader)

    def getDPandEO(self, dataloader):
        numDistribution = self.testDistribution.size()[1]
        allDistributionOUT_DP = []
        allDistributionOUT_EO = []
        allDistributionAUC_DP = []

        def checkList(curList):
            if (np.sum(curList) == len(curList)) or (np.sum(curList) == 0):
                return True
            else:
                return False

        for t in range(numDistribution):
            curUserFeature = self.testDistribution[:, t]
            # tqdm_ = tqdm(iterable=dataloader, mininterval=1, ncols=120)
            out_dp_dict = {}
            out_eo_dict = {0: {}, 1: {}}

            auc_dp_dict = {"true": {}, "pred": {}}
            auc_dp_groupScore = {}
            for feature in self.featureValueList:
                out_dp_dict[feature] = []
                out_eo_dict[0][feature] = []
                out_eo_dict[1][feature] = []
                auc_dp_dict["true"][feature] = []
                auc_dp_dict["pred"][feature] = []

            for batch in dataloader:
                userID, userProfile, label, otherInfo = batch
                tempUserFeature = curUserFeature[userID]
                userProfile[:, self.featureIndex] = tempUserFeature.view((-1,))

                userID = userID.to(self.device)
                userProfile = userProfile.to(self.device)
                otherInfo = otherInfo.to(self.device)

                for i, feature in enumerate(self.featureValueList):
                    mask = userProfile[:, self.featureIndex] == feature
                    if mask.sum() <= 1:
                        continue
                    curUserProfile = userProfile[mask]
                    curUserID = userID[mask]
                    curOtherInfo = otherInfo[mask]
                    curLabel = label[mask]
                    modelOut = (
                        self((curUserID, curUserProfile,
                             curOtherInfo)).detach().cpu()
                    )

                    auc_dp_dict["true"][feature].extend(
                        curLabel.numpy().tolist())
                    auc_dp_dict["pred"][feature].extend(
                        modelOut.numpy().tolist())

                    out_dp_dict[feature].extend(modelOut.numpy().tolist())
                    out_eo_dict[0][feature].extend(
                        modelOut[curLabel == 0].numpy().tolist()
                    )
                    out_eo_dict[1][feature].extend(
                        modelOut[curLabel == 1].numpy().tolist()
                    )
            out_dp_socre_list = [
                np.mean(v) for k, v in out_dp_dict.items() if len(v) > 0
            ]
            out_dp_score = np.max(out_dp_socre_list) - \
                np.min(out_dp_socre_list)

            out_eo_score = 0
            for curLabel in out_eo_dict.keys():
                curDP_score_list = [
                    np.mean(v) for k, v in out_eo_dict[curLabel].items() if len(v) > 0
                ]
                curDP_score = np.max(curDP_score_list) - \
                    np.min(curDP_score_list)
                out_eo_score += curDP_score

            for feature in self.featureValueList:
                if checkList(auc_dp_dict["true"][feature]):
                    continue
                auc_dp_groupScore[feature] = roc_auc_score(
                    np.array(auc_dp_dict["true"][feature]),
                    np.array(auc_dp_dict["pred"][feature]),
                )

            # print("{} AUC:".format(self.infoDict[t]))
            # for _, auc in sorted(auc_dp_groupScore.items(), key=lambda x: x[0]):
            #     print(auc)

            auc_dp_score_list = [v for k, v in auc_dp_groupScore.items()]
            auc_dp_score = np.max(auc_dp_score_list) - \
                np.min(auc_dp_score_list)

            allDistributionOUT_DP.append(out_dp_score)
            allDistributionOUT_EO.append(out_eo_score)
            allDistributionAUC_DP.append(auc_dp_score)

        CGF_DP_OUT_MCGF = np.mean(allDistributionOUT_DP) ** 2 + np.var(
            allDistributionOUT_DP
        )
        CGF_EO_OUT_MCGF = np.mean(allDistributionOUT_EO) ** 2 + np.var(
            allDistributionOUT_EO
        )
        CGF_DP_AUC_MCGF = np.mean(allDistributionAUC_DP) ** 2 + np.var(
            allDistributionAUC_DP
        )
        print("CGF(DP)_OUT MCGF:{}".format(CGF_DP_OUT_MCGF))
        print("CGF(EO)_OUT MCGF:{}".format(CGF_EO_OUT_MCGF))
        print("CGF(DP)_AUC_MCGF:{}".format(CGF_DP_AUC_MCGF))

        return CGF_DP_OUT_MCGF, CGF_EO_OUT_MCGF, CGF_DP_AUC_MCGF

    def minimizeMI(self, dataLoader, config):
        self.requires_grad_(False)
        self.Gexo.requires_grad_(True)
        self.mi_estimator.requires_grad_(True)
        # min_mi_est = float('inf')
        # min_state_dict = None

        print("=" * 5 + "minimizeMI" + "=" * 5)
        # for epoch in range(self.MI_train_epochs):
        tqdm_ = tqdm(iterable=dataLoader, mininterval=1, ncols=120)
        mi_est_values = []
        first_MI = -1
        last_MI = -1
        for step, batch_data in enumerate(tqdm_):
            userID, userProfile, label, otherInfo = batch_data
            userID = userID.to(self.device)
            userProfile = userProfile.to(self.device)
            otherInfo = otherInfo.to(self.device)
            data = (userID, userProfile, otherInfo)

            userEmbedding = self.Gexo(self.getuserEmbedding(data))

            self.Gexo.train()
            self.mi_estimator.eval()
            gexo_loss = self.mi_estimator(userEmbedding, userProfile)
            self.Gexo_optimizer.zero_grad()
            gexo_loss.backward()
            self.Gexo_optimizer.step()

            self.mi_estimator.train()
            for j in range(config.MI_eachBatchIter):
                userEmbedding = self.Gexo(self.getuserEmbedding(data))
                mi_loss = self.mi_estimator.learning_loss(
                    userEmbedding, userProfile)
                self.mi_optimizer.zero_grad()
                mi_loss.backward()
                self.mi_optimizer.step()

            curMi_est = self.mi_estimator(
                userEmbedding, userProfile).detach().cpu().item()
            if step == 0:
                first_MI = curMi_est
            last_MI = curMi_est
            mi_est_values.append(curMi_est)

            if step % config.interval == 0 and step > 0:
                tqdm_.set_description(
                    "step {:d},MI_est: {:.4f}".format(
                        step, curMi_est)
                )
        mi_data = pd.DataFrame({"MI": mi_est_values})
        mi_data.to_csv(config.base_model_dir + "MI_est.csv", index=False)

        self.requires_grad_(True)
        self.mi_estimator.requires_grad_(False)

        print("mi_est:{}".format(last_MI))
        return first_MI, last_MI

    def getTestMIvalue(self, dataLoader):
        tqdm_ = tqdm(iterable=dataLoader, mininterval=1, ncols=120)
        mi_est_values = []
        for step, batch_data in enumerate(tqdm_):
            userID, userProfile, label, otherInfo = batch_data
            userID = userID.to(self.device)
            userProfile = userProfile.to(self.device)
            otherInfo = otherInfo.to(self.device)
            data = (userID, userProfile, otherInfo)

            userEmbedding = self.Gexo(self.getuserEmbedding(data))

            curMi_est = self.mi_estimator(
                userEmbedding, userProfile).detach().cpu()
            mi_est_values.append(np.abs(curMi_est.item()))

        print("mi_est:{}".format(np.mean(mi_est_values)))
