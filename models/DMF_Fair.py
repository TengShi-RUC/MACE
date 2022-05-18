# import numpy as np
# import torch
# import torch.nn as nn
# from tqdm import tqdm

# from models import DMF
# from .modules.mlp import MLP
# from .modules.mi_estimators import *
# from .modules.basic_model import BasicModel


# class DeepMatrixFactorization_Fair(BasicModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.embedding = torch.FloatTensor(
#             np.load(config.embeddingPath, allow_pickle=True))

#         if config.useSensitiveFeature:
#             self.user_embedding = nn.Embedding.from_pretrained(
#                 torch.cat((self.embedding, torch.FloatTensor(self.userProfileDict)), dim=-1))
#         else:
#             self.user_embedding = nn.Embedding.from_pretrained(self.embedding)
#         self.item_embedding = nn.Embedding.from_pretrained(self.embedding.T)

#         self.userMLP = MLP(config.userMLP)
#         self.itemMLP = MLP(config.itemMLP)

#         self.cos = nn.CosineSimilarity()
#         self.Sigmoid = nn.Sigmoid()  # sigmoid要不要加？

#         self.Gexo = MLP(config.Gexo)
#         self.Gendo = MLP(config.Gendo)
#         self.h1_mlp = MLP(config.h1)
#         self.h2_mlp = MLP(config.h2)
#         self.prob_sigmoid = nn.Sigmoid()

#         self.mi_estimator = eval(config.MI_estimator_name)(
#             config.d_exo, config.sensitiveFeatureNum, config.MI_estimator_hiddenSize)
#         self.Gexo_optimizer = torch.optim.Adam(
#             self.Gexo.parameters(), lr=config.learning_rate)
#         self.mi_optimizer = torch.optim.Adam(
#             self.mi_estimator.parameters(), lr=config.learning_rate)
#         self.mi_estimator.requires_grad_(False)

#         for m in self.children():
#             m.to(self.device)

#     # def iv(self, userEmbedding):
#     #     gexo = self.Gexo(userEmbedding)  # (batch_size, d_exo)
#     #     gendo = self.Gendo(userEmbedding)  # (batch_size, d_endo)
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
#         user = userID.to(self.device)
#         return self.user_embedding(user)

#     def forward(self, x):
#         userID, userProfile, otherInfo = x
#         user = userID.to(self.device)
#         item = otherInfo.to(self.device)
#         user_input = self.user_embedding(user)
#         item_input = self.item_embedding(item)

#         user_input = self.iv(user_input)

#         user_out = self.userMLP(user_input)
#         item_out = self.itemMLP(item_input)

#         # norm_user_output = torch.sqrt(torch.sum(user_out**2, dim=1)) + 1e-1
#         # norm_item_output = torch.sqrt(torch.sum(item_out**2, dim=1)) + 1e-1
#         # y_ = torch.sum(user_out * item_out, dim=1) / \
#         #     (norm_user_output*norm_item_output)
#         y_ = self.Sigmoid(self.cos(user_out, item_out))
#         return y_

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
#     #         userID, itemID, label, userProfile = batch_data
#     #         userID = userID.to(self.device)
#     #         userProfile = userProfile.to(self.device)

#     #         userEmbedding = self.Gexo(self.getuserEmbedding(userID))

#     #         self.Gexo.train()
#     #         self.mi_estimator.eval()
#     #         gexo_loss = self.mi_estimator(userEmbedding, userProfile)
#     #         self.Gexo_optimizer.zero_grad()
#     #         gexo_loss.backward()
#     #         self.Gexo_optimizer.step()

#     #         self.mi_estimator.train()
#     #         for j in range(config.MI_eachBatchIter):
#     #             userEmbedding = self.Gexo(self.getuserEmbedding(userID))
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

#     #     self.requires_grad_(True)
#     #     self.mi_estimator.requires_grad_(False)

#     #     curEpochMI_est = np.mean(mi_est_values)
#     #     print('mi_est:{}'.format(curEpochMI_est))
#     #     return curEpochMI_est

#     # def getTestMIvalue(self, dataLoader):
#     #     tqdm_ = tqdm(iterable=dataLoader, mininterval=1, ncols=120)
#     #     mi_est_values = []
#     #     for step, batch_data in enumerate(tqdm_):
#     #         userID, itemID, label, userProfile = batch_data
#     #         userID = userID.to(self.device)
#     #         userProfile = userProfile.to(self.device)
#     #         userEmbedding = self.Gexo(self.getuserEmbedding(userID))

#     #         curMi_est = self.mi_estimator(
#     #             userEmbedding, userProfile).detach().cpu()
#     #         mi_est_values.append(np.abs(curMi_est.item()))

#     #     print('mi_est:{}'.format(np.mean(mi_est_values)))
