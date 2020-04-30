# -*- coding:utf-8 -*-

import sys
from helper import *
import os
import random
import math
import argparse
from tqdm import tqdm
import warnings
import numpy as np
import torchtext
from torchtext import data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# import transformer
import torchtext
from torchtext import data
from generator import Generator
from discriminator import Discriminator
# from target_lstm import TargetLSTM
from rollout import Rollout
import pickle
# from nltk.translate.bleu_score import corpus_bleu
import dill
# from data_iter import GenDataIter, DisDataIter
warnings.filterwarnings("ignore")

# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--test', action='store_true')
opt = parser.parse_args()

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 100
TOTAL_BATCH = 100
GENERATED_NUM = 1000
ROOT_PATH =  '../models/imdb/'
VOCAB_SIZE = 15000
PRE_EPOCH_NUM = 5
CHECKPOINT_PATH = ROOT_PATH + 'checkpoints/'
# DATA_FILE = '../data/imdb_sentences.txt'
DATA_FILE = '../data/data.tsv'
# EMBED_FILE = "/home/scratch/dex/glove/glove.6B.200d.txt"
EMBED_FILE = "../glove/glove.6B.200d.txt"
try:
    os.makedirs(CHECKPOINT_PATH)
except OSError:
    print('Directory already exists!')

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 200
g_hidden_dim = 200
g_sequence_len = 17
# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]

d_dropout = 0.75
d_num_class = None
# ================== Parameter Definition =================

def train_generator(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    # for each in data_iter:#
    for each in tqdm(data_iter, mininterval=2, desc=' - Generator Training', leave=False):
        data, target = each.text[:,:-1], each.text[:,1:]
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        if len(pred.shape) > 2:
            pred = torch.reshape(pred, (pred.shape[0] * pred.shape[1], -1))
        loss = criterion(pred, target)
        total_loss += loss.data.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return math.exp(total_loss / total_words)

def train_discriminator(model, generators, data_iter, criterion, optimizer):
    total_loss = 0.
    total_sents = 0.
    total_correct = 0.
    # for real_data in data_iter:
    for real_data in tqdm(data_iter, mininterval=2, desc=' - Discriminator Training', leave=False):
        fake_data = []
        fake_label = []
        for i in range(len(generators)):
            fake_data.append(generators[i].sample(BATCH_SIZE, g_sequence_len).detach().cpu())
            fake_label.append(torch.zeros(BATCH_SIZE, dtype=torch.int64) + i + 1)
        fake_data = torch.cat(fake_data)
        fake_label = torch.cat(fake_label)
        data = torch.cat([real_data.text, fake_data])
        target = torch.cat([torch.zeros(real_data.text.shape[0], dtype=torch.int64), fake_label])
        shuffle_index = torch.randperm(real_data.text.shape[0] + BATCH_SIZE*len(generators))
        data = data[shuffle_index]
        target = target[shuffle_index]
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        total_correct += torch.sum((target == torch.max(pred, axis=1)[1]))
        if len(pred.shape) > 2:
            pred = torch.reshape(pred, (pred.shape[0] * pred.shape[1], -1))
        loss = criterion(pred, target)
        total_loss += loss.data.item()
        total_sents += data.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_sents = torch.tensor(total_sents)
    if opt.cuda:
        total_sents = total_sents.cuda()
    return total_loss / total_sents, total_correct / total_sents

def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data.item()
        total_words += data.size(0) * data.size(1)
    data_iter.reset()
    return math.exp(total_loss / total_words)


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss


# def main():
random.seed(SEED)
np.random.seed(SEED)
# global VOCAB_SIZE

corpus, TEXT, LABEL, label_names, label_datasets = load_data_2(DATA_FILE, g_sequence_len, EMBED_FILE)
real_data_iterator = data.Iterator(corpus, batch_size=BATCH_SIZE)
label_data_iterators = [data.Iterator(d, batch_size=BATCH_SIZE) for d in label_datasets]
VOCAB_SIZE = len(TEXT.vocab)
print('VOCAB SIZE:', VOCAB_SIZE)

with open(CHECKPOINT_PATH + "TEXT.Field","wb")as f:
     dill.dump(TEXT,f)
     # TEXT=dill.load(f)


# Define Networks
# nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2 # the number of heads in the multiheadattention models
# dropout = 0.2 # the dropout value
# generator = TransformerModel(VOCAB_SIZE, g_emb_dim, nhead, g_hidden_dim, nlayers, dropout)

generators = []
for label in label_names:
    temp = Generator(label, VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    temp.emb.weight.data = TEXT.vocab.vectors
    generators.append(temp)
if opt.cuda:
    for i in range(len(label_names)):
        generators[i] = generators[i].cuda()

# Pretrain Generators using MLE
print('Pretrain Generators ...')
gen_criterions = [nn.NLLLoss(size_average=False) for _ in label_names]
gen_optimizers = [optim.Adam(generators[i].parameters()) for i in range(len(label_names))]
if opt.cuda:
    for i in range(len(label_names)):
        gen_criterions[i] = gen_criterions[i].cuda()
for epoch in range(PRE_EPOCH_NUM):
    for i in range(len(label_names)):
        loss = train_generator(generators[i], label_data_iterators[i], gen_criterions[i], gen_optimizers[i])
        bleu_s = bleu_4(TEXT, corpus, generators[i], g_sequence_len, count=100)
        print('Epoch [{}], Generator: {}, loss: {}, bleu_4: {}'.format(epoch, generators[i].name, loss, bleu_s))

d_num_class = len(label_names) + 1
discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
if opt.cuda:
    discriminator = discriminator.cuda()

# Pretrain Discriminator
dis_criterion = nn.NLLLoss(size_average=False)
dis_optimizer = optim.Adam(discriminator.parameters())
if opt.cuda:
    dis_criterion = dis_criterion.cuda()
print('Pretrain Discriminator ...')
for epoch in range(PRE_EPOCH_NUM):
    loss, acc = train_discriminator(discriminator, generators, real_data_iterator, dis_criterion, dis_optimizer)
    print('Epoch [{}], loss: {}, accuracy: {}'.format(epoch, loss, acc))

# # Adversarial Training
rollouts = [Rollout(generator, 0.8) for generator in generators]
print('#####################################################')
print('Start Adversarial Training...')
gen_gan_losses = [GANLoss() for _ in generators]
gen_gan_optm = [optim.Adam(generator.parameters()) for generator in generators]
if opt.cuda:
    gen_gan_loss = [gen_gan_loss.cuda() for gen_gan_loss in gen_gan_losses]
# gen_criterion = nn.NLLLoss(size_average=False)
# if opt.cuda:
    # gen_criterion = gen_criterion.cuda()
# dis_criterion = nn.NLLLoss(size_average=False)
# dis_optimizer = optim.Adam(discriminator.parameters())
# if opt.cuda:
    # dis_criterion = dis_criterion.cuda()
for total_batch in range(TOTAL_BATCH):
    for _ in range(3):
        for i in range(len(generators)):
            samples = generators[i].sample(BATCH_SIZE, g_sequence_len)
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollouts[i].get_reward(samples, 16, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            if opt.cuda:
                rewards = rewards.cuda()
            rewards = torch.exp(rewards).contiguous().view((-1,))
            prob = generators[i].forward(inputs)
            loss = gen_gan_losses[i](prob, targets, rewards)
            gen_gan_optm[i].zero_grad()
            loss.backward()
            gen_gan_optm[i].step()

    if total_batch % 10 == 0 or total_batch == TOTAL_BATCH - 1:
        for generator in generators:
            print('Saving generator {} with bleu_4: {}'.format(generator.name, bleu_4(TEXT, corpus, generator, g_sequence_len, count=100)))
            torch.save(generator.state_dict(), CHECKPOINT_PATH + 'generator_seqgan_{}.model'.format(generator.name))
        torch.save(discriminator.state_dict(), CHECKPOINT_PATH + 'discriminator_seqgan.model')
    for rollout in rollouts:
        rollout.update_params()

    for _ in range(1):
        loss, acc = train_discriminator(discriminator, generators, real_data_iterator, dis_criterion, dis_optimizer)
        print('Epoch [%d], loss: %f, accuracy: %f' % (total_batch, loss, acc))

# if __name__ == '__main__':
    # main()
