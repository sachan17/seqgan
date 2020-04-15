import sys
sys.path.insert(0, '../core')

from helper import *
#
import os
import random
import math
import argparse
import tqdm
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
warnings.filterwarnings("ignore")


SEED = 88
BATCH_SIZE = 100
TOTAL_BATCH = 10000
GENERATED_NUM = 1000
ROOT_PATH =  '../models/imdb/'
POSITIVE_FILE = ROOT_PATH + 'real.data'
TEST_FILE     = ROOT_PATH + 'test.data'
NEGATIVE_FILE = ROOT_PATH + 'gene.data'
DEBUG_FILE = ROOT_PATH + 'debug.data'
EVAL_FILE = ROOT_PATH + 'eval.data'
VOCAB_SIZE = 15000
PRE_EPOCH_NUM = 150
CHECKPOINT_PATH = ROOT_PATH + 'checkpoints/'
DATA_FILE = '../data/imdb_sentences.txt'
try:
    os.makedirs(CHECKPOINT_PATH)
except OSError:
    print('Directory already exists!')

g_emb_dim = 200
g_hidden_dim = 200
g_sequence_len = 17

idx_to_word, word_to_idx, VOCAB_SIZE = load_vocab(CHECKPOINT_PATH + 'metadata.data')
test_iter = GenDataIter(TEST_FILE, BATCH_SIZE)
generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, True)
generator = generator.cuda()
generator.load_state_dict(torch.load(CHECKPOINT_PATH+'generator_seqgan.model'))
sentences = generator.sample(10, g_sequence_len).cpu().data.numpy().tolist()
for each_sen in list(sentences):
    print('Output:', ' '.join(generate_sentence_from_id(idx_to_word, each_sen)))
