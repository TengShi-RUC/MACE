import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, embed_dim, num_class, device, dropout=0.3, neg_slope=0.2):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.neg_slope = neg_slope
        self.out_dim = num_class
        self.criterion = nn.NLLLoss()

        self.network = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 2),
                      int(self.embed_dim * 4), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 4),
                      int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 2),
                      int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

        for m in self.children():
            m.to(self.device)

    def forward(self, embeddings, labels):
        scores = self.network(embeddings)
        output = F.log_softmax(scores, dim=1)
        loss = self.criterion(output.squeeze(), labels)
        return loss
