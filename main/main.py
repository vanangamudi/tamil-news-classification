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
logging.basicConfig(format=config.FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
import sys


from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.utilz import initialize_task

from model.model import Model
from utilz import load_data, train, predict

import importlib


SELF_NAME = os.path.basename(__file__).replace('.py', '')

import sys
import pickle
import argparse
from matplotlib import pyplot as plt
plt.style.use('ggplot')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MACNet variant 2')
    parser.add_argument('-p','--hpconfig',
                        help='path to the hyperparameters config file',
                        default='hpconfig.py', dest='hpconfig')
    parser.add_argument('--log-filters',
                        help='log filters',
                        dest='log_filter')

    subparsers = parser.add_subparsers(help='commands')
    train_parser = subparsers.add_parser('train', help='starts training')
    train_parser.add_argument('--train', default='train', dest='task')
    train_parser.add_argument('--mux', action='store_true', default=False, dest='mux')
    
    predict_parser = subparsers.add_parser('predict',
                                help='''starts a cli interface for running predictions 
                                in inputs with best model from last training run''')
    predict_parser.add_argument('--predict', default='predict', dest='task')
    predict_parser.add_argument('--show-plot', action='store_true', dest='show_plot')
    predict_parser.add_argument('--save-plot', action='store_true',  dest='save_plot')
    args = parser.parse_args()
    print(args)
    if args.log_filter:
        log.addFilter(CMDFilter(args.log_filter))

    ROOT_DIR = initialize_task(args.hpconfig)

    sys.path.append('.')
    print(sys.path)
    HPCONFIG = importlib.__import__(args.hpconfig.replace('.py', ''))
    config.HPCONFIG = HPCONFIG.CONFIG
    print('====================================')
    print(ROOT_DIR)
    print('====================================')
        
    if config.CONFIG.flush:
        log.info('flushing...')
        dataset = load_data(config)
        pickle.dump(dataset, open('{}__cache.pkl'.format(SELF_NAME), 'wb'))
    else:
        dataset = pickle.load(open('{}__cache.pkl'.format(SELF_NAME), 'rb'))
        
    log.info('dataset size: {}'.format(len(dataset.trainset)))
    log.info('dataset[:10]: {}'.format(pformat(dataset.trainset[-1])))

    log.info('vocab: {}'.format(pformat(dataset.output_vocab.freq_dict)))
    
    try:
        model =  Model(config, 'macnet', len(dataset.input_vocab),  len(dataset.output_vocab))
        model_snapshot = '{}/weights/{}.{}'.format(ROOT_DIR, SELF_NAME, 'pth')
        model.load_state_dict(torch.load(model_snapshot))
        log.info('loaded the old image for the model from :{}'.format(model_snapshot))
    except:
        log.exception('failed to load the model  from :{}'.format(model_snapshot))

    if config.CONFIG.cuda:
        model = model.cuda()        
        if config.CONFIG.multi_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    print('**** the model', model)
    
    if args.task == 'train':
        train(config, args, SELF_NAME, ROOT_DIR, model, dataset)
        
    if args.task == 'predict':
        print('=========== PREDICTION ==============')
        model.eval()
        count = 0
        while True:
            count += 1
            input_string = input('?')
            if not input_string:
                continue
            
            label = predict(config, args, model, input_string, dataset)            
            print(input_string.replace('@@ ', ''), '==', label)
                
                        
    if 'service' in sys.argv:
        model.eval()
        from flask import Flask,request,jsonify
        from flask_cors import CORS
        app = Flask(__name__)
        CORS(app)

        @app.route('/ade-genentech',methods=['POST'])
        def _predict():
           print(' requests incoming..')
           sentence = []
           try:
               input_string = word_tokenize(request.json["text"].lower())
               sentence.append([VOCAB[w] for w in input_string] + [VOCAB['EOS']])
               dummy_label = LongVar([0])
               sentence = LongVar(sentence)
               input_ = [0], (sentence,), (0, )
               output, attn = model(input_)
               #print(LABELS[output.max(1)[1]], attn)
               nwords = len(input_string)
               return jsonify({
                   "result": {
                       'sentence': input_string,
                       'attn': ['{:0.4f}'.format(i) for i in attn.squeeze().data.cpu().numpy().tolist()[:-1]],
                       'probs': ['{:0.4f}'.format(i) for i in output.exp().squeeze().data.cpu().numpy().tolist()],
                       'label': LABELS[output.max(1)[1].squeeze().data.cpu().numpy()]
                   }
               })
           
           except Exception as e:
               print(e)
               return jsonify({"result":"model failed"})

        print('model running on port:5010')
        app.run(host='0.0.0.0',port=5010)
