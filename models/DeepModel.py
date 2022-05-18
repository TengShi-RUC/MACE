import torch
import torch.nn as nn
from numpy.random import beta

from .modules.basic_model import BasicModel
from .modules.mi_estimators import *
from .modules.mlp import MLP
from .modules.mixup import *


class DeepModel(BasicModel):
    def __init__(self, config):
        super().__init__(config)

        self.userEmbedding = nn.Embedding(
            config.userNum, config.user_vector_size)
        self.itemEmbedding = nn.Embedding(
            config.itemNum, config.item_vector_size)

        self.mainMLP = MLP(config.mainMLP)
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
            m.to(self.device)

    def getuserEmbedding(self, x):
        userID, userProfile, otherInfo = x
        userEmbedding = self.userEmbedding(userID)
        if self.useSensitiveFeature:
            userEmbedding = torch.cat([userEmbedding, userProfile], dim=-1)
        if self.AdvLearning:
            userEmbedding = self.applyFilter(userEmbedding)
        return userEmbedding

    def forward(self, x):
        userID, userProfile, itemID = x

        userEmbedding = self.getuserEmbedding(x)
        if self.use_iv:
            userEmbedding = self.iv(userEmbedding)
        itemEmbedding = self.itemEmbedding(itemID)

        mlp_input = torch.cat([userEmbedding, itemEmbedding], dim=-1)

        output = self.Sigmoid(self.mainMLP(mlp_input)).squeeze()

        return output

    def mixup_loss(self, batch_input_0, batch_input_1):
        userID_0, userProfile_0, itemID_0 = batch_input_0
        userID_1, userProfile_1, itemID_1 = batch_input_1
        user_input_0 = self.userEmbedding(userID_0)
        user_input_1 = self.userEmbedding(userID_1)
        item_input_0 = self.itemEmbedding(itemID_0)
        item_input_1 = self.itemEmbedding(itemID_1)

        alpha = 1
        gamma = beta(alpha, alpha)
        user_input_mix = getMix(user_input_0, user_input_1, gamma)
        item_input_mix = getMix(item_input_0, item_input_1, gamma)
        userProfile_mix = getMix(userProfile_0, userProfile_1, gamma)

        if self.useSensitiveFeature:
            user_out = torch.cat([user_input_mix, userProfile_mix], dim=-1)
        if self.use_iv:
            user_out = self.iv(user_out)
        mlp_input = torch.cat([user_out, item_input_mix], dim=-1)
        output = self.Sigmoid(self.mainMLP(mlp_input))

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
