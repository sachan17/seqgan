import os
import random
import warnings
import helper
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from generator import Generator
from discriminator import Discriminator
import dill
warnings.filterwarnings("ignore")

SEED = 88
BATCH_SIZE = 100
ROOT_PATH =  '../models/'
PRE_EPOCH_NUM = 150
CHECKPOINT_PATH = ROOT_PATH + 'imdb_pos_neg_1/'
g_emb_dim = 200
g_hidden_dim = 200
g_sequence_len = 17

models = []
for each in os.listdir(CHECKPOINT_PATH):
    if '.gen' in each:
        models.append(each)

with open(CHECKPOINT_PATH + "TEXT.Field","rb")as f:
     TEXT=dill.load(f)
VOCAB_SIZE = len(TEXT.vocab)
for model in models:
    print('Generated text by:', model)
    generator = Generator(model, VOCAB_SIZE, g_emb_dim, g_hidden_dim, False)
    generator.load_state_dict(torch.load(CHECKPOINT_PATH + model, map_location=torch.device('cpu')))
    sentences = generator.sample(10, g_sequence_len).cpu().data.numpy().tolist()
    for each_sen in list(sentences):
        print('Output:', ' '.join([TEXT.vocab.itos[i] for i in each_sen if i not in [0, 1, 2, 3]]))
