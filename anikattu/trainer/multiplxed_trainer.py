import os
import logging
import copy
from config import CONFIG
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from ..debug import memory_consumed
from ..utilz import ListTable, Averager, tqdm
from ..utilz import are_weights_same
from ..datafeed import MultiplexedDataFeed

from .trainer import FLAGS, EpochAverager

import torch

from torch import optim, nn
from collections import namedtuple

from nltk.corpus import stopwords


"""
An object that takes in MultiplexedDatafeed(N datafeeds) and a Trainer and N Tester objects.
Helps coordinates with MultiplexedDatafeed to sample data based on some criteria,

Default criteria: 
  based on accuracies of Tester objects. Lower accuracy DataFeeds are sampled frequently
"""
class MultiplexedTrainer(object):
    def __init__(self, name,
                 config,
                 model,
                 feed,
                 optimizer,
                 loss_function,
                 directory,
                 testers = [],
                 epochs=1000,
                 checkpoint=1,
                 do_every_checkpoint=None,
                 sampling_distribution=None,
    ):

        self.name  = name
        self.config = config
        self.ROOT_DIR = directory

        self.log = logging.getLogger('{}.{}.{}'.format(__name__, self.__class__.__name__, self.name))
        
        self.model = model
        assert isinstance(feed, MultiplexedDataFeed)
        self.feed = feed
        
        self.epochs     = epochs
        self.checkpoint = min(checkpoint, epochs)

        self.testers = testers
        self.do_every_checkpoint = do_every_checkpoint if not do_every_checkpoint == None else self._do_every_checkpoint
        self.sampling_distribution = sampling_distribution if not sampling_distribution == None else self._sampling_distribution

        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.optimizer = optimizer if optimizer else optim.SGD(self.model.parameters(),
                                                               lr=0.01, momentum=0.1)

        self.__build_stats()

        if self.config.CONFIG.cuda:
            self.model.cuda()
        
    def __build_stats(self):
        # necessary metrics
        self.train_loss = EpochAverager(self.config, filename = '{}/results/metrics/{}.{}'.format(self.ROOT_DIR, self.name,  'train_loss'))
        self.metrics = [self.train_loss]

    def train(self):
        for epoch in range(self.epochs):
            self.log.critical('memory consumed : {}'.format(memory_consumed()))            

            if epoch % max(1, (self.checkpoint - 1)) == 0:
                if self.do_every_checkpoint(epoch) == FLAGS.STOP_TRAINING:
                    self.log.info('loss trend suggests to stop training')
                    return

            self.model.train()
            for j in tqdm(range(self.feed.num_batch), desc='Trainer.{}'.format(self.name)):
                self.log.debug('{}th batch'.format(j))
                self.optimizer.zero_grad()
                input_ = self.feed.next_batch(self.sampling_distribution(epoch))
                output = self.model(input_)
                loss = self.loss_function(output, input_)
                self.train_loss.cache(loss.data.item())
                loss.backward()
                self.optimizer.step()


            self.log.info('-- {} -- loss: {}\n'.format(epoch, self.train_loss.epoch_cache))                
            self.train_loss.clear_cache()
            
            for m in self.metrics:
                m.write_to_file()

        return True

    def _sampling_distribution(self, epoch):
        return {k:v.accuracy[-1] for k, v in self.testers.items()}
    
    def _do_every_checkpoint(self, epoch, early_stopping=True):
        for t in tester.values():
            t.do_every_checkpoint(epoch)
