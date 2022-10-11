import torch
import torch.nn as nn
from numpy.random import beta

from models.modules.mixup import getLossReg, getMix

from .modules.attention_pooling_layer import AttentionSequencePoolingLayer
from .modules.basic_model import BasicModel
from .modules.fc import FullyConnectedLayer
from .modules.mi_estimators import *
from .modules.mlp import MLP


class DeepInterestNetwork(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = config.description
        self.device = config.device

        self.itemEmbedding = nn.Embedding(
            config.itemNum + 1, config.itemEmbeddingSize, padding_idx=0
        )

        self.attention = AttentionSequencePoolingLayer(
            embedding_dim=config.itemEmbeddingSize
        )

        if self.use_iv:
            self.Gexo = MLP(config.Gexo)
            if not self.direct_use_gexo:
                self.Gendo = MLP(config.Gendo)
                self.h1_mlp = MLP(config.h1)
                self.h2_mlp = MLP(config.h2)
                self.prob_sigmoid = nn.Sigmoid()
            if self.random_iv:
                self.Gexo.requires_grad_(False)

        if self.use_iv:
            if self.direct_use_gexo:
                inputSize = config.itemEmbeddingSize + config.d_exo
            else:
                inputSize = config.itemEmbeddingSize + config.d_endo
        else:
            if self.useSensitiveFeature:
                inputSize = config.sensitiveFeatureNum + 2 * config.itemEmbeddingSize
            else:
                inputSize = 2 * config.itemEmbeddingSize
        self.fc_layer = FullyConnectedLayer(
            input_size=inputSize,
            hidden_unit=[128, 64, 1],
            bias=[True, True, False],
            batch_norm=False,
            sigmoid=True,
            activation="dice",
            dice_dim=2,
        )
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
            # self.MI_train_epochs = config.MI_train_epochs
            # self.MI_eachBatchIter = config.MI_eachBatchIter
            self.mi_estimator.requires_grad_(False)

        for m in self.children():
            m.to(self.device)

    def getuserEmbedding(self, x):
        userID, userProfile, otherInfo = x
        # userProfile = userProfile.to(self.device)
        userBehavior = otherInfo[:, 0:-1]
        targetItem = otherInfo[:, -1]
        # userBehavior = userBehavior.to(self.device)
        # targetItem = targetItem.to(self.device)

        targetItemEmbed = self.itemEmbedding(targetItem.unsqueeze(1))

        behaviorEmbedding = self.itemEmbedding(userBehavior)
        behaviorMask = torch.where(userBehavior == 0, 1, 0).bool()

        userEmbedding = self.attention(
            targetItemEmbed, behaviorEmbedding, behaviorMask
        ).squeeze(1)

        if self.useSensitiveFeature:
            # (batch_size, itemEmbeddingSize+3)
            userEmbedding = torch.cat([userEmbedding, userProfile], dim=-1)
        # else:
        #     userEmbedding = userEmbedding  # (batch_size, itemEmbeddingSize)
        if self.AdvLearning:
            userEmbedding = self.applyFilter(userEmbedding)

        return userEmbedding

    def forward(self, x):
        userID, userProfile, otherInfo = x
        # userProfile = userProfile.to(self.device)
        userBehavior = otherInfo[:, 0:-1]
        targetItem = otherInfo[:, -1]
        # userProfile (batchSize, sensitiveFeatureNum)
        # userBehavior (batchSize, seqLen)
        # targetItem (batchSize, )
        # targetItem = targetItem.to(self.device)

        # (batchSize, 1, itemEmbeddingSize)
        targetItemEmbed = self.itemEmbedding(targetItem.unsqueeze(1))

        userEmbedding = self.getuserEmbedding(
            x)  # (batch_size, itemEmbeddingSize)

        if self.use_iv:
            userEmbedding = self.iv(userEmbedding)

        concatFeature = torch.cat(
            [userEmbedding, targetItemEmbed.squeeze(1)], dim=-1)

        output = self.fc_layer(concatFeature).squeeze()
        # output (batchSize, )
        return output

    def mixup_loss(self, batch_input_0, batch_input_1):
        userID_0, userProfile_0, otherInfo_0 = batch_input_0
        targetItem_0 = otherInfo_0[:, -1]
        targetItemEmbed_0 = self.itemEmbedding(targetItem_0)
        userEmbedding_0 = self.getuserEmbedding(batch_input_0)

        userID_1, userProfile_1, otherInfo_1 = batch_input_1
        targetItem_1 = otherInfo_1[:, -1]
        targetItemEmbed_1 = self.itemEmbedding(targetItem_1)
        userEmbedding_1 = self.getuserEmbedding(batch_input_1)

        alpha = 1
        gamma = beta(alpha, alpha)
        targetItemEmbed_mix = getMix(
            targetItemEmbed_0, targetItemEmbed_1, gamma)
        userEmbedding_mix = getMix(userEmbedding_0, userEmbedding_1, gamma)

        if self.use_iv:
            user_input = self.iv(userEmbedding_mix)
        else:
            user_input = userEmbedding_mix
        concatFeature = torch.cat([user_input, targetItemEmbed_mix], dim=-1)
        output = self.fc_layer(concatFeature).squeeze()

        gradx = torch.autograd.grad(
            output.sum(),
            [userEmbedding_mix, targetItemEmbed_mix],
            create_graph=True,
            allow_unused=True,
        )

        user_loss = getLossReg(gradx[0], userEmbedding_1 - userEmbedding_0)
        item_loss = getLossReg(gradx[1], targetItemEmbed_1 - targetItemEmbed_0)
        return user_loss + item_loss
