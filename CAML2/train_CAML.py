from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import csv
import argparse
from keras.preprocessing import sequence
from datetime import datetime
import numpy as np
import random
import codecs
np.random.seed(1337)
random.seed(1337)
import os
from tqdm import tqdm
from utilities import *
from tylib.exp.metrics import *
import time
import tensorflow as tf
import sys
from sklearn.utils import shuffle
from collections import Counter
from collections import defaultdict
import pickle
import string
import re
import math
import operator

from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from tf_models.model_caml import Model
from tylib.exp.experiment_caml import Experiment
from tylib.exp.exp_ops import *
from parser_CAML import *
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

PAD = "<PAD>"
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"


########################################################################################
class CFExperiment(Experiment):
    def __init__(self, inject_params=None):
        print("Starting Rec Experiment")
        super(CFExperiment, self).__init__()
        self.parser = build_parser()
        self.no_text_mode = False
        self.args = self.parser.parse_args()

        # For hierarchical setting
        self.args.qmax = self.args.smax * self.args.dmax
        self.args.amax = self.args.smax * self.args.dmax

        original = self.args.rnn_type
        self.args.rnn_type = 'RAW_MSE_CAML_FN_FM'
        self.args.base_encoder = 'NBOW'
        self.model_name = self.args.rnn_type
        self._setup()

        print('='*40, '加载数据', '='*40)  
        t1 = time.time()
        self._load_sets()
        print('Finish！ time (s):', round(time.time()-t1))

        print('='*40, '构造模型', '='*40)
        t1 = time.time()
        self.mdl = Model(self.vocab, self.args,
                            #char_vocab=len(self.char_index),
                            #pos_vocab=len(self.pos_index),
                            mode='HREC', num_item=self.num_items,
                            num_user=self.num_users)
        print('Finish！ time (s):', round(time.time()-t1))
        self._setup_tf(load_embeddings=not self.no_text_mode)
        print('CFExperiment 初始化完成！')

    ##################################### 加载数据 START ##################################### 
    def _prepare_set(self, data):
        user = [x[0] for x in data]
        items = [x[1] for x in data]
        labels = [x[2] for x in data]

        # Raw user-item ids
        user_idx = user
        item_idx = items

        user_list = []
        item_list = []
        user_concept_list = []
        item_concept_list = []
        user_len = []
        item_len = []
        user_concept_len = []
        item_concept_len = []
        for i in range(len(user)):
            user_reviews = []
            item_reviews = []
            user_concepts = []
            item_concepts = []
            user_r_len = []
            user_c_len = []
            item_r_len = []
            item_c_len = []
 
            for x in self.ui_review_dict[user[i]]:
                user_reviews.append(self.ui_review_dict[user[i]][x])
                user_concepts.append(self.ui_concept_dict[user[i]][x])
                user_r_len.append(len(self.ui_review_dict[user[i]][x]))
                user_c_len.append(len(self.ui_concept_dict[user[i]][x]))
                if len(user_reviews) == self.args.dmax:
                    break
            user_list.append(user_reviews)
            user_concept_list.append(user_concepts)
            for x in self.iu_review_dict[items[i]]:
                item_reviews.append(self.iu_review_dict[items[i]][x])
                item_concepts.append(self.iu_concept_dict[items[i]][x])
                item_r_len.append(len(self.iu_review_dict[items[i]][x]))
                item_c_len.append(len(self.iu_concept_dict[items[i]][x]))
                if len(item_reviews) == self.args.dmax:
                    break
            item_list.append(item_reviews)
            item_concept_list.append(item_concepts)
            user_len.append(user_r_len)
            item_len.append(item_r_len)
            user_concept_len.append(user_c_len)
            item_concept_len.append(item_c_len)
        
        user_concept, user_concept_len = prep_hierarchical_data_list_new(user_concept_list,
                                                                         user_concept_len, 
                                                                         self.args.smax, 
                                                                         self.args.dmax)
        items_concept, item_concept_len = prep_hierarchical_data_list_new(item_concept_list,
                                                                          item_concept_len, 
                                                                          self.args.smax,  
                                                                          self.args.dmax)

        user, user_len = prep_hierarchical_data_list_new(user_list, user_len,
                                                              self.args.smax,
                                                              self.args.dmax)
        items, item_len = prep_hierarchical_data_list_new(item_list, item_len,
                                                               self.args.smax,
                                                               self.args.dmax)
            
        output = [user, user_len, items, item_len]
        self.mdl.register_index_map(0, 'q1_inputs')
        self.mdl.register_index_map(1, 'q1_len')
        self.mdl.register_index_map(2, 'q2_inputs')
        self.mdl.register_index_map(3, 'q2_len')

        output.append(user_concept)
        output.append(user_concept_len)
        output.append(items_concept)
        output.append(item_concept_len)
        self.mdl.register_index_map(4, 'c1_inputs')
        self.mdl.register_index_map(5, 'c1_len')
        self.mdl.register_index_map(6, 'c2_inputs')
        self.mdl.register_index_map(7, 'c2_len')
        
        idx = 7
        gen_outputs = [x[3] for x in data]
        gen_len = [x[4] for x in data]

        output.append(gen_outputs)
        output.append(gen_len)
        idx += 1
        self.mdl.register_index_map(idx, 'gen_outputs')
        idx += 1
        self.mdl.register_index_map(idx, 'gen_len')
        output.append(labels)
        output = list(zip(*output))
        return output
    
    
    def load_dataset(self, train_data):
        output = []
        reviews = []

        for p1, p2, c in zip(train_data['paper_id1'], train_data['paper_id2'], train_data['clean_content']):
            linedata = [self.voc.word2idx[str(word)] for word in c.split()]
            output.append((p1, p2, 1))
            reviews.append(linedata)
        return output, reviews

    def load_review_data(self, train_data, data_type):
        #分组，获取历史数据
        p1p2 = train_data.groupby('paper_id1').apply(
            lambda subdf:subdf.paper_id2.values).to_dict()
        p1ctx = train_data.groupby('paper_id1').apply(
            lambda subdf:subdf.clean_content.values).to_dict()
        p2p1= train_data.groupby('paper_id2').apply(
            lambda subdf:subdf.paper_id2.values).to_dict()
        p2ctx = train_data.groupby('paper_id2').apply(
            lambda subdf:subdf.clean_content.values).to_dict()

        ui_dict = {}
        iu_dict = {}
        for p1 in p1p2:
            ui_dict[p1] = {}
            for p2, ctx in zip(p1p2[p1], p1ctx[p1]):
                ctx = [self.voc.word2idx[str(word)] for word in ctx.split()]
                ctx = [self.voc.word2idx['<SOS>']] + ctx + [self.voc.word2idx['<EOS>']]
                #if data_type == 'concept':
                ui_dict[p1][p2] = ctx

        for p2 in p2p1:
            iu_dict[p2] = {} 
            for p1, ctx in zip(p2p1[p2], p2ctx[p2]):
                ctx = [self.voc.word2idx[str(word)] for word in ctx.split()]
                ctx = [self.voc.word2idx['<SOS>']] + ctx + [self.voc.word2idx['<EOS>']]
                #if data_type == 'concept':
                iu_dict[p2][p1] = ctx
        
        length1 = [len(ui_dict[x]) for x in ui_dict]
        length2 = [len(iu_dict[x]) for x in iu_dict]
        length3 = []
        for x in ui_dict:
            length3 += [len(ui_dict[x][y]) for y in ui_dict[x]]
        show_stats('{}:user num review'.format(data_type), length1)
        show_stats('{}:item num review'.format(data_type), length2)
        show_stats('{}:review num word'.format(data_type), length3)
        return ui_dict, iu_dict 
    
    def _load_sets(self):
        # 加载数据：我改在了我的任务/数据集上，参考 utilities.py 里的 load_data
        # train_data/test_data 都是包含4列的dataframe ['paper_id1', 'paper_id1', 'clean_content', 'n_of_number']
        # 这个数据是一个因引文网络数据，节点表示论文，连边表示引用/被引用的关系，clean_content是引用连边上的文本
        
        path = '{}.pkl'.format(self.args.dataset)
        train_data, test_data, _, voc = load_data(path)
        self.voc = voc
        
        #添加100个停用词
        self.stop_concept = {}
        '''
        freq_words = {k:v for k,v in sorted(voc.word_freq.items(), 
                                            key = lambda x:x[1], reverse = True)}
        freq_words = list(freq_words)[0:100]
        for w in freq_words:
            self.stop_concept.add(w)
        '''
        
        self.vocab = len(voc.word2idx)
        print("vocab={}".format(self.vocab))
        word2df = None
        
        self.train_rating_set, self.train_reviews = self.load_dataset(train_data)
        self.test_rating_set, self.test_reviews = self.load_dataset(test_data)
    
        self.ui_review_dict, self.iu_review_dict = self.load_review_data(train_data, "review")
        self.ui_concept_dict, self.iu_concept_dict = self.load_review_data(train_data, "concept")  
        self.num_users = len(self.ui_review_dict)
        self.num_items = len(self.iu_review_dict)

    ####################################### 加载数据 END ####################################### 
    
    def train(self):
        print('='*40, ' Train ', '='*40)
        scores = []
        counter = 0
        min_loss = 1e+7
        epoch_scores = {}
        self.eval_list = []
        
        data = self.train_rating_set
        reviews = self.train_reviews
        print("Training Interactions={}".format(len(data)))
    
        self.sess.run(tf.assign(self.mdl.is_train,self.mdl.true))
                
        for epoch in range(self.args.epochs): 
            all_att_dict = {}
            pos_val, neg_val = [],[]
            losses = []
            review_losses = []
            #random.shuffle(data) 
            num_batches = int(len(data) / self.args.batch_size)
            t1 = time.time()
            for i in tqdm(range(0, num_batches+1)):
                batch = batchify(data, reviews, i, self.args.batch_size, max_sample=len(data))  
                if(len(batch)==0):
                    continue
                batch = self._prepare_set(batch)
                
                feed_dict = self.mdl.get_feed_dict(batch)
                _, loss, gen_loss, gen_acc, att1, att2, word_att1, word_att2 = self.sess.run(
                    [self.mdl.train_op,
                     self.mdl.cost, self.mdl.gen_loss, self.mdl.gen_acc, 
                     self.mdl.att1,self.mdl.att2, self.mdl.word_att1, self.mdl.word_att2], 
                    feed_dict)
                #if i % 50 == 0:
                 #   print('step:{}, loss:{}'.format(i, round(loss/self.args.batch_size, 2)))
                
            print('>> Epoch {} time(s):{:.2f}\tloss = {:.2f}'.format(epoch, time.time()-t1, loss))
              

    def evaluate(self):
        print('Evaluate...')
        self.sess.run(tf.assign(self.mdl.is_train, self.mdl.false)) 
        #self.evaluate(self.test_rating_set, self.test_reviews, epoch)
        data = self.test_rating_set
        reviews = self.test_reviews
     
        rouge = Rouge()
        bleu_score = [[], [], [], []]
        rouge_score = [0.]*9
        #generated_file = open("gen_results/"+self.args.dataset+".txt", "w")
        t1 = time.time()
        for i in tqdm(range(len(data))):
            batch = batchify(data, reviews, i, 1, max_sample=len(data))
            source_sentence = ' '.join([self.voc.idx2word[idx] for idx in batch[0][3]])
            batch = self._prepare_set(batch)
            if(len(batch)==0): continue
                
            feed_dict = self.mdl.get_feed_dict(batch, mode='testing') 
            answer = self.sess.run([self.mdl.answer], feed_dict)[0]
            generated_sentence = ' '.join([self.voc.idx2word[idx] for idx in answer])
            bleu_1, bleu_2, bleu_3, bleu_4 = compute_bleu([source_sentence], 
                                                          [generated_sentence])
            bleu_score[0].append(bleu_1)
            bleu_score[1].append(bleu_2)
            bleu_score[2].append(bleu_3)
            bleu_score[3].append(bleu_4)

            rouge_1 = rouge.get_scores(generated_sentence, source_sentence)
            rouge_score[0:3] = list(map(lambda x :x[0]+x[1], 
                                    zip(rouge_score[0:3], list(rouge_1[0]['rouge-1'].values()))))
            rouge_score[3:6] = list(map(lambda x :x[0]+x[1],
                                    zip(rouge_score[3:6], list(rouge_1[0]['rouge-2'].values()))))
            rouge_score[6:9] = list(map(lambda x :x[0]+x[1],
                                    zip(rouge_score[6:9], list(rouge_1[0]['rouge-l'].values()))))
            #generated_file.write("Case :" + str(i) + "\n")
            #generated_file.write(source_sentence + "\n")
            #generated_file.write(generated_sentence + "\n\n")
   
            #print("Case: " + str(i))
            #print("Origin: " + source_sentence)
            #print("Generated: " + generated_sentence)
            #if i % 100 == 0:
             #   print('evaluate [{}/{}] time (s):{:.2f}'.format(i, len(data), time.time() - t1)) 
     
        bleu_score = [sum(i)/len(i) for i in bleu_score]
        rouge_score = [i/len(data) for i in rouge_score]
        print('bleu_score:', bleu_score)
        print('rouge_score:', rouge_score)
        
        #res_file = open('gen_results/result.txt', "a+")
        #res_file.write('Dataset:{}\tepochs:{}\n'.format(self.args.dataset, self.args.epochs))
        #res_file.write('BLEU:{}\n'.format(bleu_score))
        #res_file.write('rouge_score:{}\n\n'.format(rouge_score))
        #return bleu_score, rouge_score
            
            
            
if __name__ == '__main__':
    print('='*40, '主函数', '='*40)
    exp = CFExperiment(inject_params=None)
    exp.train()
    exp.evaluate()
