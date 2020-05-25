# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.embedding(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)



class classifier1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.act = nn.LogSoftmax()

    def forward(self, text):
        embedded = self.embedding(text)
        packed_output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        dense_outputs=self.fc(hidden)
        outputs=self.act(dense_outputs)
        return outputs
