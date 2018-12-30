import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')
import config
from anikattu.logger import CMDFilter
import logging
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from functools import partial


import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from anikattu.utilz import Var, LongVar, init_hidden

class Base(nn.Module):
    def __init__(self, config, name):
        super(Base, self).__init__()
        self._name = name
        self.log = logging.getLogger(self._name)
        size_log_name = '{}.{}'.format(self._name, 'size')
        self.log.info('constructing logger: {}'.format(size_log_name))
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.info('size_log')
        self.log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.size_log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.print_instance = 0
        
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
            if self.print_instance or print_instance:
                self.size_log.debug(tensor)

            
        return tensor

    def name(self, n):
        return '{}.{}'.format(self._name, n)

class Model(Base):
    def __init__(self, Config, name, input_vocab_size, output_vocab_size):
        super(Model, self).__init__(config, name)
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = config.HPCONFIG.hidden_dim
        self.embed_dim = config.HPCONFIG.embed_dim

        self.embed = nn.Embedding(self.input_vocab_size, self.embed_dim)
        self.encode = nn.LSTM(self.embed_dim, self.hidden_dim, bidirectional=True, num_layers=config.HPCONFIG.num_layers)
        self.classify = nn.Linear(2*self.hidden_dim, self.output_vocab_size)
        
        if config.CONFIG.cuda:
            self.cuda()
        
    def init_hidden(self, batch_size):
        ret = torch.zeros(2, batch_size, self.hidden_dim)
        if config.HPCONFIG().cuda: ret = ret.cuda()
        return Variable(ret)
    
    def forward(self, input_):
        ids, (seq,), _ = input_
        if seq.dim() == 1: seq = seq.unsqueeze(0)
            
        batch_size, seq_size = seq.size()
        seq_emb = F.tanh(self.embed(seq))
        seq_emb = seq_emb.transpose(1, 0)
        pad_mask = (seq > 0).float()
        
        states, cell_state = self.encode(seq_emb)

        logits = self.classify(states[-1])
        
        return F.log_softmax(logits, dim=-1)
