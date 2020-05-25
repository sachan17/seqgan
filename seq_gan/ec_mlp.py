from helper import *
import warnings
import argparse
import os
import dill
from discriminator import *
import torch
from torch import nn, optim
warnings.filterwarnings("ignore")

# ================== Parameter Definition =================
parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--server', action='store_true')
opt = parser.parse_args()
# Basic Training Paramters
ROOT_PATH =  '../models/'
CHECKPOINT_PATH = ROOT_PATH + 'chat_ec/'
SEED = 88
BATCH_SIZE = 100
EPOCH_NUM = 150
EMBED_FILE = "../glove/glove.6B.200d.txt"
DATA_FILE = '../data/friends_train.tsv'
DEV_FILE = '../data/friends_dev.tsv'
TEST_FILE = '../data/friends_test.tsv'
if opt.server:
    EPOCH_NUM = 150
    EMBED_FILE = "/home/scratch/dex/glove/glove.6B.200d.txt"
try:
    os.makedirs(CHECKPOINT_PATH)
except OSError:
    print('Directory already exists!')

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

g_sequence_len = 20
d_emb_dim = 200
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]

d_dropout = 0.75
d_num_class = None
# ================== Parameter Definition =================

def train_discriminator(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_sents = 0.
    total_correct = 0.
    if not opt.server:
        data_iter = tqdm(data_iter, mininterval=2, desc=' - Discriminator Training', leave=False)
    for batch in data_iter:
        data = batch.text2
        target = batch.label
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

def eval_discriminator(model, data_iter, criterion):
    total_loss = 0.
    total_sents = 0.
    total_correct = 0.
    if not opt.server:
        data_iter = tqdm(data_iter, mininterval=2, desc=' - Discriminator Evaluation', leave=False)
    for batch in data_iter:
        data = batch.text2
        target = batch.label
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
    total_sents = torch.tensor(total_sents)
    if opt.cuda:
        total_sents = total_sents.cuda()
    return total_loss / total_sents, total_correct / total_sents

corpus, TEXT, LABEL = load_conv_data(DATA_FILE, g_sequence_len, EMBED_FILE, min_freq=1)
data_iterator = data.Iterator(corpus, batch_size=BATCH_SIZE)
VOCAB_SIZE = len(TEXT.vocab)
print('VOCAB SIZE:', VOCAB_SIZE)

dev_data = data.Iterator(data.TabularDataset(DEV_FILE, format='tsv', fields=[('text1', TEXT), ('text2', TEXT), ('label', LABEL)]), batch_size=BATCH_SIZE)
test_data = data.Iterator(data.TabularDataset(TEST_FILE, format='tsv', fields=[('text1', TEXT), ('text2', TEXT), ('label', LABEL)]), batch_size=BATCH_SIZE)

with open(CHECKPOINT_PATH + "TEXT.Field", "wb")as f:
     dill.dump(TEXT,f)


d_num_class = len(LABEL.vocab.itos)
discriminator = Discriminator(d_num_class, VOCAB_SIZE,  d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
# discriminator = classifier1(VOCAB_SIZE,  d_emb_dim, 100, d_num_class, 2, True, 0.3)
discriminator.embedding.weight.data = TEXT.vocab.vectors
if opt.cuda:
    discriminator = discriminator.cuda()

dis_criterion = nn.NLLLoss(size_average=False)
dis_optimizer = optim.Adam(discriminator.parameters())
if opt.cuda:
    dis_criterion = dis_criterion.cuda()
for epoch in range(EPOCH_NUM):
    discriminator.train()
    loss, acc = train_discriminator(discriminator, data_iterator, dis_criterion, dis_optimizer)
    discriminator.eval()
    v_loss, v_acc = eval_discriminator(discriminator, dev_data, dis_criterion)
    print('Epoch [{}], t_loss: {}, t_accuracy: {} v_loss: {}, v_accuracy: {}'.format(epoch, loss, acc, v_loss, v_acc))
t_loss, t_acc = eval_discriminator(discriminator, test_data, dis_criterion)
print('Final test_loss: {}, test_accuracy: {}'.format(t_loss, t_acc))
torch.save(discriminator.state_dict(), CHECKPOINT_PATH + 'ec_cnn.model')
