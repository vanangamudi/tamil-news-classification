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

import torch

from torch import optim, nn
from collections import namedtuple

from nltk.corpus import stopwords

class FLAGS:
    CONTINUE_TRAINING = 0
    STOP_TRAINING = 1
    

class EpochAverager(Averager):
    def __init__(self, config, filename=None, *args, **kwargs):
        super(EpochAverager, self).__init__(config, filename, *args, **kwargs)
        self.config = config
        self.epoch_cache = Averager(config, filename, *args, *kwargs)

    def cache(self, a):
        self.epoch_cache.append(a)

    def clear_cache(self):
        super(EpochAverager, self).append(self.epoch_cache.avg)
        self.epoch_cache.empty();
                
                 
class Trainer(object):
    def __init__(self, name,
                 config,
                 model,
                 feed,
                 optimizer,
                 loss_function,
                 directory,
                 epochs=1000,
                 checkpoint=1,
                 do_every_checkpoint=None
    ):

        self.name  = name
        self.config = config
        self.ROOT_DIR = directory

        self.log = logging.getLogger('{}.{}.{}'.format(__name__, self.__class__.__name__, self.name))
        
        self.model = model
        self.feed = feed
        
        self.epochs     = epochs
        self.checkpoint = min(checkpoint, epochs)

        self.do_every_checkpoint = do_every_checkpoint if not do_every_checkpoint == None else lambda x: FLAGS.CONTINUE_TRAINING

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
                input_ = self.feed.next_batch()
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


class Tester(object):
    def __init__(self, name,
                 config,
                 model,
                 feed,
                 loss_function,
                 accuracy_function,
                 directory,
                 f1score_function=None,
                 best_model=None,
                 predictor=None,
                 save_model_weights=True,
    ):

        self.name  = name
        self.config = config
        self.ROOT_DIR = directory

        self.log = logging.getLogger('{}.{}.{}'.format(__name__, self.__class__.__name__, self.name))

        self.model = model

        self.feed = feed

        self.predictor = predictor

        self.accuracy_function = accuracy_function if accuracy_function else self._default_accuracy_function
        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.f1score_function = f1score_function
        

        self.__build_stats()

        self.save_model_weights = save_model_weights
        self.best_model = (0.000001, self.model.cpu().state_dict())
        try:
            f = '{}/{}_best_model_accuracy.txt'.format(self.ROOT_DIR, self.name)
            if os.path.isfile(f):
                self.best_model = (float(open(f).read().strip()), self.model.cpu().state_dict())
                self.log.info('loaded last best accuracy: {}'.format(self.best_model[0]))
        except:
            log.exception('no last best model')

                        
        self.best_model_criteria = self.accuracy
        self.save_best_model()

        if self.config.CONFIG.cuda:
            self.model.cuda()
        
    def __build_stats(self):
        
        # necessary metrics
        self.mfile_prefix = '{}/results/metrics/{}'.format(self.ROOT_DIR, self.name)
        self.test_loss  = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,   'test_loss'))
        self.accuracy   = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'accuracy'))

        # optional metrics
        self.tp = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,   'tp'))
        self.fp = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'fp'))
        self.fn = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'fn'))
        self.tn = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'tn'))
      
        self.precision = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'precision'))
        self.recall = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'recall'))
        self.f1score   = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'f1score'))

        self.metrics = [self.test_loss, self.accuracy, self.precision, self.recall, self.f1score]
        
    def save_best_model(self):
        with open('{}/{}_best_model_accuracy.txt'.format(self.ROOT_DIR, self.name), 'w') as f:
            f.write(str(self.best_model[0]))

        if self.save_model_weights:
            self.log.info('saving the last best model with accuracy {}...'.format(self.best_model[0]))
            torch.save(self.best_model[1], '{}/weights/{:0.4f}.{}'.format(self.ROOT_DIR, self.best_model[0], 'pth'))
            torch.save(self.best_model[1], '{}/weights/{}.{}'.format(self.ROOT_DIR, self.name, 'pth'))

    def do_every_checkpoint(self, epoch, early_stopping=True):

        self.model.eval()
        for j in tqdm(range(self.feed.num_batch), desc='Tester.{}.checkpoint :{}'.format(self.name, epoch)):
            input_ = self.feed.next_batch()
            output = self.model(input_)
            
            loss = self.loss_function(output, input_)
            self.test_loss.cache(loss.item())
            accuracy = self.accuracy_function(output, input_)
            self.accuracy.cache(accuracy.item())

            if self.f1score_function:
                (tp, fn, fp, tn), precision, recall, f1score = self.f1score_function(output, input_, j)

                self.tp.cache(tp)
                self.fn.cache(fn)
                self.fp.cache(fp)
                self.tn.cache(tn)
                self.precision.cache(precision)
                self.recall.cache(recall)
                self.f1score.cache(f1score)

        self.log.info('= {} =loss:{}'.format(epoch, self.test_loss.epoch_cache))
        self.log.info('- {} -accuracy:{}'.format(epoch, self.accuracy.epoch_cache))
        if self.f1score_function:
            self.log.info('- {} -tp:{}'.format(epoch, sum(self.tp.epoch_cache)))
            self.log.info('- {} -fn:{}'.format(epoch, sum(self.fn.epoch_cache)))
            self.log.info('- {} -fp:{}'.format(epoch, sum(self.fp.epoch_cache)))
            self.log.info('- {} -tn:{}'.format(epoch, sum(self.tn.epoch_cache)))
                        
            self.log.info('- {} -precision:{}'.format(epoch, self.precision.epoch_cache))
            self.log.info('- {} -recall:{}'.format(epoch, self.recall.epoch_cache))
            self.log.info('- {} -f1score:{}\n'.format(epoch, self.f1score.epoch_cache))

        if self.best_model[0] < self.accuracy.epoch_cache.avg:
            self.log.info('beat best model...')
            last_acc = self.best_model[0]
            self.best_model = (self.accuracy.epoch_cache.avg, self.model.cpu().state_dict())
            self.save_best_model()
            
            if self.config.CONFIG.cuda:
                self.model.cuda()

            if self.predictor and self.best_model[0] > 0.75:
                log.info('accuracy is greater than 0.75...')
                if ((self.best_model[0] >= self.config.CONFIG.ACCURACY_THRESHOLD  and  (5 * (self.best_model[0] - last_acc) > self.config.CONFIG.ACCURACY_IMPROVEMENT_THRESHOLD))
                    or (self.best_model[0] - last_acc) > self.config.CONFIG.ACCURACY_IMPROVEMENT_THRESHOLD):
                    
                    self.predictor.run_prediction(self.accuracy.epoch_cache.avg)
                

        self.test_loss.clear_cache()
        self.accuracy.clear_cache()
        self.tp.clear_cache()
        self.fn.clear_cache()
        self.fp.clear_cache()
        self.tn.clear_cache()
        self.f1score.clear_cache()
        self.precision.clear_cache()
        self.recall.clear_cache()
        
        for m in self.metrics:
            m.write_to_file()
            
        if early_stopping:
            return self.loss_trend()

    def loss_trend(self, total_count=10):
        if len(self.test_loss) > 4:
            losses = self.test_loss[-4:]
            count = 0
            for l, r in zip(losses, losses[1:]):
                if l < r:
                    count += 1
                    
            if count > total_count:
                return FLAGS.STOP_TRAINING

        return FLAGS.CONTINUE_TRAINING


    def _default_accuracy_function(self):
        return -1
    
    
class Predictor(object):
    def __init__(self, name, model,
                 feed,
                 repr_function,
                 directory,
                 *args, **kwargs):
        self.name = name
        self.model = model
        self.ROOT_DIR = directory

        self.log = logging.getLogger('{}.{}.{}'.format(__name__, self.__class__.__name__, self.name))
        
        self.repr_function = repr_function
        self.feed = feed
        
    def predict(self,  batch_index=0):
        self.log.debug('batch_index: {}'.format(batch_index))
        input_ = self.feed.nth_batch(batch_index)
        self.model.eval()
        output = self.model(input_)
        results = ListTable()
        results.extend( self.repr_function(output, input_) )
        output_ = output
        return output_, results

    def run_prediction(self, accuracy):        
        dump = open('{}/results/{}_{:0.4f}.csv'.format(self.ROOT_DIR, self.name, accuracy), 'w')
        self.log.info('on {}th eon'.format(accuracy))
        results = ListTable()
        for ri in tqdm(range(self.feed.num_batch), desc='running prediction at accuracy: {:0.4f}'.format(accuracy)):
            output, _results = self.predict(ri)
            results.extend(_results)
        dump.write(repr(results))
        dump.close()
