import numpy as np
import torch
import torch.nn as nn
from numpy.random import beta

from .modules.basic_model import BasicModel
from .modules.mi_estimators import *
from .modules.mlp import MLP
from .modules.mixup import *


class DeepMatrixFactorization(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = torch.FloatTensor(
            np.load(config.embeddingPath, allow_pickle=True)
        ).to(self.device)

        # self.user_embedding = nn.Embedding.from_pretrained(embedding)
        # self.item_embedding = nn.Embedding.from_pretrained(embedding.T)

        self.userMLP = MLP(config.userMLP)
        self.itemMLP = MLP(config.itemMLP)

        self.cos = nn.CosineSimilarity()
        self.Sigmoid = nn.Sigmoid() 

        if self.use_iv:
            self.Gexo = MLP(config.Gexo)
            if not self.direct_use_gexo:
                self.Gendo = MLP(config.Gendo)
                self.h1_mlp = MLP(config.h1)
                self.h2_mlp = MLP(config.h2)
                self.prob_sigmoid = nn.Sigmoid()
            if self.random_iv:
                self.Gexo.requires_grad_(False)

        if self.use_MI:
            self.mi_estimator = eval(config.MI_estimator_name)(
                config.d_exo, config.sensitiveFeatureNum, config.MI_estimator_hiddenSize
            )
            self.Gexo_optimizer = torch.optim.Adam(
                self.Gexo.parameters(),
                lr=config.learning_rate,
                weight_decay=config.l2_penalty,
            )
            self.mi_optimizer = torch.optim.Adam(
                self.mi_estimator.parameters(),
                lr=config.learning_rate,
                weight_decay=config.l2_penalty,
            )
            self.mi_estimator.requires_grad_(False)

        for m in self.children():
            # if isinstance(m, nn.Embedding):
            #     continue
            m.to(self.device)

    def user_embedding(self, x):
        return self.embedding[x]

    def item_embedding(self, x):
        return self.embedding[:, x].T

    def getuserEmbedding(self, x):
        userID, userProfile, otherInfo = x
        # user = userID.to(self.device)
        # userProfile = userProfile.to(self.device)
        userEmbedding = self.user_embedding(userID)
        userEmbedding = self.userMLP(userEmbedding)
        if self.useSensitiveFeature:
            userEmbedding = torch.cat([userEmbedding, userProfile], dim=-1)
        # else:
        #     return userEmbedding
        if self.AdvLearning:
            userEmbedding = self.applyFilter(userEmbedding)
        return userEmbedding

    def forward(self, x):
        userID, userProfile, otherInfo = x
        # item = otherInfo.to(self.device)
        item_input = self.item_embedding(otherInfo)

        user_out = self.getuserEmbedding(x)
        if self.use_iv:
            user_out = self.iv(user_out)
        item_out = self.itemMLP(item_input)

        # norm_user_output = torch.sqrt(torch.sum(user_out**2, dim=1)) + 1e-1
        # norm_item_output = torch.sqrt(torch.sum(item_out**2, dim=1)) + 1e-1
        # y_ = torch.sum(user_out * item_out, dim=1) / \
        #     (norm_user_output*norm_item_output)
        y_ = self.Sigmoid(self.cos(user_out, item_out))
        return y_

    def mixup_loss(self, batch_input_0, batch_input_1):
        userID_0, userProfile_0, otherInfo_0 = batch_input_0
        userID_1, userProfile_1, otherInfo_1 = batch_input_1
        user_input_0 = self.user_embedding(userID_0)
        user_input_1 = self.user_embedding(userID_1)
        item_input_0 = self.item_embedding(otherInfo_0)
        item_input_1 = self.item_embedding(otherInfo_1)

        alpha = 1
        gamma = beta(alpha, alpha)
        user_input_mix = getMix(user_input_0, user_input_1, gamma)
        item_input_mix = getMix(item_input_0, item_input_1, gamma)
        userProfile_mix = getMix(userProfile_0, userProfile_1, gamma)

        user_out = self.userMLP(user_input_mix)
        item_out = self.itemMLP(item_input_mix)
        if self.useSensitiveFeature:
            user_out = torch.cat([user_out, userProfile_mix], dim=-1)
        if self.use_iv:
            user_out = self.iv(user_out)
        output = self.Sigmoid(self.cos(user_out, item_out))

        gradx = torch.autograd.grad(
            output.sum(),
            [user_input_mix, userProfile_mix, item_input_mix],
            create_graph=True,
            allow_unused=True,
        )

        user_input_loss = getLossReg(gradx[0], user_input_1 - user_input_0)
        userProfile_loss = getLossReg(gradx[1], userProfile_1 - userProfile_0)
        item_input_loss = getLossReg(gradx[2], item_input_1 - item_input_0)
        return user_input_loss + item_input_loss + userProfile_loss
