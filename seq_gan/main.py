import sys
from helper import *
import os
import random
import math
import argparse
from tqdm import tqdm
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
from seq2seq_generator import Seq2Seq
# from target_lstm import TargetLSTM
from rollout import Rollout
import pickle
# from nltk.translate.bleu_score import corpus_bleu
import dill
import warnings
warnings.filterwarnings("ignore")

# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--server', action='store_true')
opt = parser.parse_args()
# Basic Training Paramters
ROOT_PATH =  '../models/'
CHECKPOINT_PATH = ROOT_PATH + 'imdb/'
SEED = 88
BATCH_SIZE = 100
TOTAL_BATCH = 100
PRE_EPOCH_NUM = 5
EMBED_FILE = "../glove/glove.6B.200d.txt"
# DATA_FILE = '../data/data.tsv'
DATA_FILE = '../data/friends_train.tsv'
# DATA_FILE = '../data/imdb_sentences.txt'
if opt.server:
    TOTAL_BATCH = 1000
    PRE_EPOCH_NUM = 150
    EMBED_FILE = "/home/scratch/dex/glove/glove.6B.200d.txt"
try:
    os.makedirs(CHECKPOINT_PATH)
except OSError:
    print('Directory already exists!')

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
n_layers = 2
g_emb_dim = 200
g_hidden_dim = 200
g_sequence_len = 20
g_dropout = 0.75
clip = 1
# Discriminator Parameters
d_emb_dim = 200
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]

d_dropout = 0.75
d_num_class = None
# ================== Parameter Definition =================

def train_generator(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    if not opt.server:
        data_iter = tqdm(data_iter, mininterval=2, desc=' - Generator Training', leave=False)
    for each in data_iter:
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

def train_seq2seq_generator(model, data_iter, criterion, optimizer, clip):
    epoch_loss = 0
    for batch in data_iter:
        src = batch.text1 #[batch_size, seq_len]
        trg = batch.text2 #[batch_size, seq_len]
        if opt.cuda:
            src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_iter)


def train_discriminator(model, generators, data_iter, criterion, optimizer):
    total_loss = 0.
    total_sents = 0.
    total_correct = 0.
    if not opt.server:
        data_iter = tqdm(data_iter, mininterval=2, desc=' - Discriminator Training', leave=False)
    for real_data in data_iter:
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

corpus, TEXT, LABEL, label_names, label_datasets = load_conv_data(DATA_FILE, g_sequence_len, EMBED_FILE)
real_data_iterator = data.Iterator(corpus, batch_size=BATCH_SIZE)
label_data_iterators = [data.Iterator(d, batch_size=BATCH_SIZE) for d in label_datasets]
VOCAB_SIZE = len(TEXT.vocab)
print('VOCAB SIZE:', VOCAB_SIZE)
print(corpus.fields['label'].vocab.freqs)
# exit(0)

with open(CHECKPOINT_PATH + "TEXT.Field","wb")as f:
     dill.dump(TEXT,f)
     # TEXT=dill.load(f)
generators = []
for label in label_names:
    temp = Seq2Seq(label, VOCAB_SIZE, g_emb_dim, g_hidden_dim, n_layers, g_dropout, g_dropout, opt.cuda)
    temp.encoder.embedding.weight.data = TEXT.vocab.vectors
    temp.decoder.embedding.weight.data = TEXT.vocab.vectors
    # temp = Generator(label, VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    # temp.emb.weight.data = TEXT.vocab.vectors
    generators.append(temp)
if opt.cuda:
    for i in range(len(label_names)):
        generators[i] = generators[i].cuda()

# exit(0)
# Pretrain Generators using MLE
# print('Pretrain Generators ...')
# # gen_criterions = [nn.NLLLoss(size_average=False) for _ in label_names]
# gen_criterions = [nn.CrossEntropyLoss(ignore_index=TEXT.vocab.stoi[TEXT.pad_token]) for _ in label_names]
# gen_optimizers = [optim.Adam(generators[i].parameters()) for i in range(len(label_names))]
# if opt.cuda:
#     for i in range(len(label_names)):
#         gen_criterions[i] = gen_criterions[i].cuda()
# for epoch in range(PRE_EPOCH_NUM):
#     for i in range(len(label_names)):
#         loss = train_seq2seq_generator(generators[i], label_data_iterators[i], gen_criterions[i], gen_optimizers[i], clip)
#         # loss = train_generator(generators[i], label_data_iterators[i], gen_criterions[i], gen_optimizers[i])
#         bleu_s = 0#bleu_4(TEXT, corpus, generators[i], g_sequence_len, count=100)
#         print('Epoch [{}], Generator: {}, loss: {}, Perplexity: {}'.format(epoch, generators[i].name, loss, math.exp(loss)))
#     print('-'*25)

exit(0)
d_num_class = len(label_names) + 1
discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
discriminator.embedding.weight.data = TEXT.vocab.vectors
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
            torch.save(generator.state_dict(), CHECKPOINT_PATH + 'generator_seqgan_{}.gen'.format(generator.name))
        torch.save(discriminator.state_dict(), CHECKPOINT_PATH + 'discriminator_seqgan.dis')
    for rollout in rollouts:
        rollout.update_params()

    for _ in range(1):
        loss, acc = train_discriminator(discriminator, generators, real_data_iterator, dis_criterion, dis_optimizer)
        print('Epoch [%d], loss: %f, accuracy: %f' % (total_batch, loss, acc))

# if __name__ == '__main__':
    # main()
