import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext import data, vocab

import numpy as np

import random
import math
import time
from helper import *


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding.weight.requires_grad=False
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.embedding.weight.requires_grad=False
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, vocab_dim, emb_dim, hid_dim, n_layers, enc_dropout, dec_dropout, use_cuda):
        super().__init__()
        self.encoder = Encoder(vocab_dim, emb_dim, hid_dim, n_layers, enc_dropout)
        self.decoder = Decoder(vocab_dim, emb_dim, hid_dim, n_layers, dec_dropout)
        # self.device = device
        self.use_cuda = use_cuda
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        self.init_weights()

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        if self.use_cuda:
            outputs = outputs.cuda()
        hidden, cell = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

    def init_weights(self):
        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file = '../data/friends_test.tsv'
    g_sequence_len = 20
    embed_file = '../glove/glove.6B.200d.txt'
    corpus, TEXT, LABEL = load_conv_data(file, g_sequence_len, embed_file)
    data_iterator = data.Iterator(corpus, batch_size=10)

    VOCAB_DIM = len(TEXT.vocab)
    EMB_DIM = 200
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    model = Seq2Seq(VOCAB_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, DEC_DROPOUT, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters())

    TRG_PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    def train(model, iterator, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
        for batch in data_iterator:
            src = batch.text1
            trg = batch.text2
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    N_EPOCHS = 10
    CLIP = 1
    for epoch in range(N_EPOCHS):
        train_loss = train(model, data_iterator, optimizer, criterion, CLIP)
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

if __name__ == '__main__':
    main()
