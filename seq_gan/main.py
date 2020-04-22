# -*- coding:utf-8 -*-

import sys
sys.path.insert(0, '../core')

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
from nltk.translate.bleu_score import corpus_bleu
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
PRE_EPOCH_NUM = 2
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
d_num_class = 3
# ================== Parameter Definition =================

def train_generator(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for each in data_iter:#
    #for each in tqdm(data_iter, mininterval=2, desc=' - Generator Training', leave=False):
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

def train_discriminator(model, generator, data_iter, criterion, optimizer):
    total_loss = 0.
    total_sents = 0.
    total_correct = 0.
    for real_data in data_iter:
    #for real_data in tqdm(data_iter, mininterval=2, desc=' - Discriminator Training', leave=False):
        fake_data = generator.sample(BATCH_SIZE, g_sequence_len).detach().cpu()
        data = torch.cat([real_data.text, fake_data])
        target = torch.cat([torch.ones(real_data.text.shape[0], dtype=torch.int64), torch.zeros(BATCH_SIZE, dtype=torch.int64)])
        shuffle_index = torch.randperm(real_data.text.shape[0] + BATCH_SIZE)
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

real_data_iterator, TEXT, corpus = load_data_2(DATA_FILE, g_sequence_len, BATCH_SIZE, EMBED_FILE)
VOCAB_SIZE = len(TEXT.vocab)
print('VOCAB SIZE:', VOCAB_SIZE)

with open(CHECKPOINT_PATH + "TEXT.Field","wb")as f:
     dill.dump(TEXT,f)
     # TEXT=dill.load(f)

aksp

# Define Networks
# nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2 # the number of heads in the multiheadattention models
# dropout = 0.2 # the dropout value
# generator = TransformerModel(VOCAB_SIZE, g_emb_dim, nhead, g_hidden_dim, nlayers, dropout)

# positive
generator_1 = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
generator_1.emb.weight.data = TEXT.vocab.vectors
# negative
generator_2 = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
generator_2.emb.weight.data = TEXT.vocab.vectors

discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
if opt.cuda:
    generator_1 = generator_1.cuda()
    generator_2 = generator_2.cuda()
    discriminator = discriminator.cuda()

# Pretrain Generator using MLE
gen_criterion = nn.NLLLoss(size_average=False)
gen_optimizer = optim.Adam(generator.parameters())
if opt.cuda:
    gen_criterion = gen_criterion.cuda()
print('Pretrain with MLE ...')
for epoch in range(PRE_EPOCH_NUM):
    loss = train_generator(generator, real_data_iterator, gen_criterion, gen_optimizer)
    bleu_s = bleu_4(TEXT, corpus, generator, g_sequence_len, count=100)
    print('Epoch [%d], loss: %f, bleu_4: %f'% (epoch, loss, bleu_s))

# Pretrain Discriminator
dis_criterion = nn.NLLLoss(size_average=False)
dis_optimizer = optim.Adam(discriminator.parameters())
if opt.cuda:
    dis_criterion = dis_criterion.cuda()
print('Pretrain Discriminator ...')
for epoch in range(PRE_EPOCH_NUM):
    loss, acc = train_discriminator(discriminator, generator, real_data_iterator, dis_criterion, dis_optimizer)
    print('Epoch [%d], loss: %f, accuracy: %f' % (epoch, loss, acc))


# # Adversarial Training
rollout = Rollout(generator, 0.8)
print('#####################################################')
print('Start Adversarial Training...\n')
gen_gan_loss = GANLoss()
gen_gan_optm = optim.Adam(generator.parameters())
if opt.cuda:
    gen_gan_loss = gen_gan_loss.cuda()
gen_criterion = nn.NLLLoss(size_average=False)
if opt.cuda:
    gen_criterion = gen_criterion.cuda()
dis_criterion = nn.NLLLoss(size_average=False)
dis_optimizer = optim.Adam(discriminator.parameters())
if opt.cuda:
    dis_criterion = dis_criterion.cuda()
for total_batch in range(TOTAL_BATCH):
    for _ in range(3):
        samples = generator.sample(BATCH_SIZE, g_sequence_len)
        zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
        if samples.is_cuda:
            zeros = zeros.cuda()
        inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
        targets = Variable(samples.data).contiguous().view((-1,))
        # calculate the reward
        rewards = rollout.get_reward(samples, 16, discriminator)
        rewards = Variable(torch.Tensor(rewards))
        if opt.cuda:
            rewards = rewards.cuda()
        rewards = torch.exp(rewards).contiguous().view((-1,))
        prob = generator.forward(inputs)
        # print('SHAPE: ', prob.shape, targets.shape, rewards.shape)
        loss = gen_gan_loss(prob, targets, rewards)
        gen_gan_optm.zero_grad()
        loss.backward()
        gen_gan_optm.step()
        # print('GEN PRED DIM: ', prob.shape)

    if total_batch % 10 == 0 or total_batch == TOTAL_BATCH - 1:
        print('Saving with Bleu_4: %f' % (bleu_4(TEXT, corpus, generator, g_sequence_len, count=100)))
        torch.save(generator.state_dict(), CHECKPOINT_PATH + 'generator_seqgan.model')
        torch.save(discriminator.state_dict(), CHECKPOINT_PATH + 'discriminator_seqgan.model')
    rollout.update_params()

    for _ in range(1):
        loss, acc = train_discriminator(discriminator, generator, real_data_iterator, dis_criterion, dis_optimizer)
        print('Epoch [%d], loss: %f, accuracy: %f' % (total_batch, loss, acc))

# if __name__ == '__main__':
    # main()
