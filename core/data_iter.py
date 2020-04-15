import os
import random
import math

import tqdm

import copy
import numpy as np
import torch
class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.LongTensor(np.array(d, dtype='int64'))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = self.read_file(real_data_file)
        fake_data_lis = self.read_file(fake_data_file)
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        # self.pairs = zip(self.data, self.labels)
        self.pairs_ind = np.arange(0, len(self.data))
        # TODO: Check if data is shuffled
        self.data_num  = len(self.data)
        self.indices   = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs_ind)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs = [self.pairs_ind[i] for i in index]
        # pairs = [self.pairs[i] for i in index]
        data = [self.data[p] for p in pairs]
        label = [self.labels[p] for p in pairs]
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis


def main():
    ROOT_PATH =  '../models/imdb/'
    POSITIVE_FILE = ROOT_PATH + 'real.data'
    TEST_FILE     = ROOT_PATH + 'test.data'
    NEGATIVE_FILE = ROOT_PATH + 'gene.data'
    DEBUG_FILE = ROOT_PATH + 'debug.data'
    EVAL_FILE = ROOT_PATH + 'eval.data'
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)


if __name__ == '__main__':
    main()
