import os
import re
import sys
import glob
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

from functools import partial
from collections import namedtuple, defaultdict, Counter


from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.datafeed import DataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq

from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize


VOCAB =  ['PAD', 'UNK', 'GO', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""
Sample   =  namedtuple('Sample', ['id', 'sequence', 'label'])

def load_news_data(config,
               filename=('../dataset/news/text.subword_nmt.txt', '../dataset/news/label.txt'),
               max_sample_size=None):
    
    samples = []
    skipped = 0

    input_vocab = Counter()
    output_vocab = Counter()
    
    try:
        log.info('processing file: {}'.format(filename))
        text_file, label_file = [open(f).readlines() for f in filename]
        for i, (s, l) in tqdm(enumerate(zip(text_file, label_file)),
                            desc='processing {}'.format(filename)):

            s, l = s.strip(), l.strip()

            if l in config.HPCONFIG.labels.keys():
                samples.append(
                    Sample(i,
                           s.strip().split(),
                           l.strip().lower())
                )
            
            
            if  max_sample_size and len(samples) > max_sample_size:
                break

    except:
        skipped += 1
        log.exception('{}'.format(line))

    print('skipped {} samples'.format(skipped))
    
    samples = sorted(samples, key=lambda x: len(x.sequence), reverse=True)
    if max_sample_size:
        samples = samples[:max_sample_size]

    log.info('building input_vocabulary...')
    for sample in samples:
        input_vocab.update(sample.sequence)            
        output_vocab.update([sample.label])

    pivot = int(len(samples) * config.CONFIG.split_ratio)
    train_samples, test_samples = samples[:pivot], samples[pivot:]
    return Dataset(filename,
                   (train_samples, test_samples),
                   Vocab(input_vocab, special_tokens=VOCAB),
                   Vocab(output_vocab))

def load_filmreviews_data(config,
                          filename=('../dataset/filmreviews/reviews.subword_nmt.csv',
                                    '../dataset/filmreviews/ratings.csv'),
                          max_sample_size=None):
    
    samples = []
    skipped = 0

    input_vocab = Counter()
    output_vocab = Counter()
    
    try:
        log.info('processing file: {}'.format(filename))
        text_file, label_file = [open(f).readlines() for f in filename]
        for i, (s, l) in tqdm(enumerate(zip(text_file, label_file)),
                            desc='processing {}'.format(filename)):

            s, l = s.strip(), l.strip()
            label = float(l.strip().lower())
            if label >= 2.75:
                label = 'positive'
            else:
                label = 'negative'
            samples.append(
                Sample(i,
                       s.strip().split(),
                       label
                )
            )
            
            
            if  max_sample_size and len(samples) > max_sample_size:
                break

    except:
        skipped += 1
        log.exception('{}'.format(line))

    print('skipped {} samples'.format(skipped))
    

    if max_sample_size:
        samples = samples[:max_sample_size]

    log.info('building input_vocabulary...')
    for sample in samples:
        input_vocab.update(sample.sequence)            
        output_vocab.update([sample.label])

    pivot = int(len(samples) * config.CONFIG.split_ratio)
    train_samples, test_samples = samples[:pivot], samples[pivot:]
    train_samples = sorted(train_samples, key=lambda x: len(x.sequence), reverse=True)
    test_samples  = sorted(test_samples, key=lambda x: len(x.sequence), reverse=True)
    return Dataset(filename,
                   (train_samples, test_samples),
                   Vocab(input_vocab, special_tokens=VOCAB),
                   Vocab(output_vocab))

def load_data(config, filename=None):
    if config.HPCONFIG.dataset == 'filmreviews':
        return load_filmreviews_data(config, filename=config.HPCONFIG.dataset_path)

    if config.HPCONFIG.dataset == 'news':
        return load_news_data(config, filename=config.HPCONFIG.dataset_path)

        
# ## Loss and accuracy function
def loss(output, batch, loss_function, *args, **kwargs):
    indices, (sequence,), (label) = batch
    return loss_function(output, label)

def accuracy(output, batch, *args, **kwargs):
    indices, (sequence,), (label) = batch
    return (output.max(dim=1)[1] == label).sum().float()/float(label.size(0))


def waccuracy(output, batch, config, *args, **kwargs):
    indices, (sequence, ), (label) = batch

    index = label
    src = Var(config, torch.ones(label.size()))
    
    acc_nomin = Var(config, torch.zeros(output.size(1)))
    acc_denom = Var(config, torch.ones(output.size(1)))

    acc_denom.scatter_add_(0, index, (label == label).float() )
    acc_nomin.scatter_add_(0, index, (label == output.max(1)[1]).float())

    accuracy = acc_nomin / acc_denom

    #pdb.set_trace()
    return accuracy.mean()

def repr_function(output, batch, VOCAB, LABELS, dataset):
    indices, (sequence,), (label) = batch
    results = []
    output = output.max(1)[1]
    output = output.cpu().numpy()
    for idx, c, a, o in zip(indices, sequence, label, output):

        c = ' '.join([VOCAB[i] for i in c]).replace('\n', ' ')
        a = ' '.join([LABELS[a]])
        o = ' '.join([LABELS[o]])
        
        results.append([str(idx), c, a, o, str(a == o) ])
        
    return results


def batchop(datapoints, VOCAB, LABELS, config,  for_prediction=False, *args, **kwargs):
    indices = [d.id for d in datapoints]
    sequence = []
    label = []

    for d in datapoints:
        sequence.append([VOCAB[w] for w in d.sequence])

        if not for_prediction:
            label.append(LABELS[d.label])

    sequence = LongVar(config, pad_seq(sequence))
    if not for_prediction:
        label   = LongVar(config, label)

    batch = indices, (sequence, ), (label)
    return batch

def portion(dataset, percent):
    return dataset[ : int(len(dataset) * percent) ]

def train(config, argv, name, ROOT_DIR,  model, dataset):
    _batchop = partial(batchop, VOCAB=dataset.input_vocab, LABELS=dataset.output_vocab)
    predictor_feed = DataFeed(name, dataset.testset, batchop=_batchop, batch_size=1)
    train_feed     = DataFeed(name, portion(dataset.trainset, config.HPCONFIG.trainset_size),
                              batchop=_batchop, batch_size=config.CONFIG.batch_size)
    
    predictor = Predictor(name,
                          model=model,
                          directory=ROOT_DIR,
                          feed=predictor_feed,
                          repr_function=partial(repr_function
                                                , VOCAB=dataset.input_vocab
                                                , LABELS=dataset.output_vocab
                                                , dataset=dataset.testset_dict))

    loss_ = partial(loss, loss_function=nn.NLLLoss())
"""
    
def predict(config, argv, model, input_string, dataset):
    tokens = input_string.strip().split()
    input_ = batchop(
        datapoints = [Sample('0', tokens, '')],
        VOCAB      = dataset.input_vocab,
        LABELS     = dataset.output_vocab,
        
        for_prediction = True
    )
            
    output = model(input_)
    return  dataset.output_vocab[output.max(1)[1]]
    
"""
