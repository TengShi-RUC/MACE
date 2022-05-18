# import numpy as np
# import torch
# import torch.nn as nn
# from tqdm import tqdm

# from models import DIN

# from .modules.attention_pooling_layer import AttentionSequencePoolingLayer
# from .modules.basic_model import BasicModel
# from .modules.fc import FullyConnectedLayer
# from .modules.mi_estimators import *
# from .modules.mlp import MLP


# class DeepInterestNetwork_Fair(BasicModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.name = config.description
#         self.device = config.device

#         self.itemEmbedding = nn.Embedding(
#             config.itemNum + 1, config.itemEmbeddingSize, padding_idx=0)
#         # +1用于pad

#         self.attention = AttentionSequencePoolingLayer(
#             embedding_dim=config.itemEmbeddingSize)

#         self.Gexo = MLP(config.Gexo)
#         self.Gendo = MLP(config.Gendo)
#         self.h1_mlp = MLP(config.h1)
#         self.h2_mlp = MLP(config.h2)
#         self.prob_sigmoid = nn.Sigmoid()

#         inputSize = config.itemEmbeddingSize + config.d_endo
#         self.fc_layer = FullyConnectedLayer(input_size=inputSize,
#                                             hidden_unit=[128, 64, 1],
#                                             bias=[True, True, False],
#                                             batch_norm=False,
#                                             sigmoid=True,
#                                             activation='dice',
#                                             dice_dim=2)

#         self.mi_estimator = eval(config.MI_estimator_name)(
#             config.d_exo, config.sensitiveFeatureNum, config.MI_estimator_hiddenSize)
#         self.Gexo_optimizer = torch.optim.Adam(
#             self.Gexo.parameters(), lr=config.learning_rate)
#         self.mi_optimizer = torch.optim.Adam(
#             self.mi_estimator.parameters(), lr=config.learning_rate)
#         # self.MI_train_epochs = config.MI_train_epochs
#         # self.MI_eachBatchIter = config.MI_eachBatchIter
#         self.mi_estimator.requires_grad_(False)

#         for m in self.children():
#             m.to(self.device)

#     # def iv(self, userEmbedding):
#     #     gexo = self.Gexo(userEmbedding)  # (batch_size, d_exo)
#     #     gendo = self.Gendo(userEmbedding)  # (batch_size, d_endo)
#     #     # solution = torch.lstsq(
#     #     #     gendo, gexo).solution[:gexo.size(1)]  # (d_exo, d_endo) 无法求导
#     #     solution = gexo.pinverse(rcond=0).matmul(gendo)  # (d_exo, d_endo)

#     #     g1 = torch.matmul(gexo, solution)  # (batch_size, d_endo)
#     #     g2 = gendo - g1  # (batch_size, d_endo)

#     #     h_mlp_input = torch.cat([gendo, gexo], dim=-1)
#     #     h1 = self.prob_sigmoid(self.h1_mlp(h_mlp_input))
#     #     h2 = self.prob_sigmoid(self.h2_mlp(h_mlp_input))

#     #     hx = h1 * g1 + h2 * g2  # (batch_size, d_endo)
#     #     return hx

#     def getuserEmbedding(self, x):
#         userID, userProfile, otherInfo = x
#         userProfile = userProfile.to(self.device)
#         userBehavior = otherInfo[:, 0:-1]
#         targetItem = otherInfo[:, -1]
#         userBehavior = userBehavior.to(self.device)
#         targetItem = targetItem.to(self.device)

#         targetItemEmbed = self.itemEmbedding(targetItem.unsqueeze(1))

#         behaviorEmbedding = self.itemEmbedding(userBehavior)
#         behaviorMask = torch.where(
#             userBehavior == 0, 1, 0).bool()

#         userEmbedding = self.attention(
#             targetItemEmbed, behaviorEmbedding, behaviorMask).squeeze(1)
#         # (batch_size, itemEmbeddingSize)
#         # 应该拼上userProfile(如果加上sensitiveFeature)
#         if self.useSensitiveFeature:
#             # (batch_size, itemEmbeddingSize+3)
#             return torch.cat([userEmbedding, userProfile], dim=-1)
#         else:
#             return userEmbedding  # (batch_size, itemEmbeddingSize)

#     def forward(self, x):
#         userID, userProfile, otherInfo = x
#         userProfile = userProfile.to(self.device)
#         userBehavior = otherInfo[:, 0:-1]
#         targetItem = otherInfo[:, -1]
#         # userProfile (batchSize, sensitiveFeatureNum)
#         # userBehavior (batchSize, seqLen)
#         # targetItem (batchSize, )
#         targetItem = targetItem.to(self.device)

#         # (batchSize, 1, itemEmbeddingSize)
#         targetItemEmbed = self.itemEmbedding(targetItem.unsqueeze(1))

#         userEmbedding = self.getuserEmbedding(
#             x)  # (batch_size, itemEmbeddingSize)
#         userEmbedding = self.iv(userEmbedding)  # (batch_size, d_endo)

#         concatFeature = torch.cat(
#             [userEmbedding, targetItemEmbed.squeeze(1)], dim=-1)

#         output = self.fc_layer(concatFeature).squeeze()
#         # output (batchSize, )
#         return output

#     # def minimizeMI(self, dataLoader, config):
#     #     self.requires_grad_(False)
#     #     self.Gexo.requires_grad_(True)
#     #     self.mi_estimator.requires_grad_(True)
#     #     # min_mi_est = float('inf')
#     #     # min_state_dict = None

#     #     print("=" * 5 + "minimizeMI" + "=" * 5)
#     #     # for epoch in range(self.MI_train_epochs):
#     #     tqdm_ = tqdm(iterable=dataLoader, mininterval=1, ncols=120)
#     #     mi_est_values = []
#     #     for step, batch_data in enumerate(tqdm_):
#     #         userID, userProfile, userBehavior, targetItem, label = batch_data
#     #         userProfile = userProfile.to(self.device)
#     #         userBehavior = userBehavior.to(self.device)
#     #         targetItem = targetItem.to(self.device)
#     #         data = (userProfile, userBehavior, targetItem)

#     #         userEmbedding = self.Gexo(self.getuserEmbedding(data))

#     #         self.Gexo.train()
#     #         self.mi_estimator.eval()
#     #         gexo_loss = self.mi_estimator(userEmbedding, userProfile)
#     #         self.Gexo_optimizer.zero_grad()
#     #         gexo_loss.backward()
#     #         self.Gexo_optimizer.step()

#     #         self.mi_estimator.train()
#     #         for j in range(config.MI_eachBatchIter):
#     #             userEmbedding = self.Gexo(self.getuserEmbedding(data))
#     #             mi_loss = self.mi_estimator.learning_loss(
#     #                 userEmbedding, userProfile)
#     #             self.mi_optimizer.zero_grad()
#     #             mi_loss.backward()
#     #             self.mi_optimizer.step()

#     #         curMi_est = self.mi_estimator(
#     #             userEmbedding, userProfile).detach().cpu()
#     #         mi_est_values.append(np.abs(curMi_est.item()))

#     #         if step % config.interval == 0 and step > 0:
#     #             tqdm_.set_description(
#     #                 "step {:d},MI_est: {:.4f}".format(step, np.mean(mi_est_values)))

#     #     # curEpochMeanMi = np.mean(mi_est_values)
#     #     # if curEpochMeanMi < min_mi_est:
#     #     #     min_mi_est = curEpochMeanMi
#     #     #     min_state_dict = self.Gexo.state_dict()

#     #     # self.Gexo.load_state_dict(min_state_dict)
#     #     self.requires_grad_(True)
#     #     self.mi_estimator.requires_grad_(False)

#     #     curEpochMI_est = np.mean(mi_est_values)
#     #     print('mi_est:{}'.format(curEpochMI_est))
#     #     return curEpochMI_est

#     # def getTestMIvalue(self, dataLoader):
#     #     tqdm_ = tqdm(iterable=dataLoader, mininterval=1, ncols=120)
#     #     mi_est_values = []
#     #     for step, batch_data in enumerate(tqdm_):
#     #         userID, userProfile, userBehavior, targetItem, label = batch_data
#     #         userProfile = userProfile.to(self.device)
#     #         userBehavior = userBehavior.to(self.device)
#     #         targetItem = targetItem.to(self.device)
#     #         data = (userProfile, userBehavior, targetItem)

#     #         userEmbedding = self.Gexo(self.getuserEmbedding(data))

#     #         curMi_est = self.mi_estimator(
#     #             userEmbedding, userProfile).detach().cpu()
#     #         mi_est_values.append(np.abs(curMi_est.item()))

#     #     print('mi_est:{}'.format(np.mean(mi_est_values)))
