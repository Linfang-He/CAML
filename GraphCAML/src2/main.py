from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from rouge import Rouge
from collections import defaultdict
import random
import scipy.sparse as sp

from parser import *
from model import Model
from utilities import *
from data_loader import *

'''
from src2.parser import *
from src2.model import Model
from src2.utilities import *
from src2.data_loader import *
'''
import datetime
import numpy as np
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # filter out INFO and WARNING logs.
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 指定gpu
import sys

DATACNT = 0
KGE = "TransD"
sys.stdout = Logger("../caml_log.txt")

PAD = "<PAD>"  # 补全字符
UNK = "<UNK>"  # 低频词语
SOS = "<SOS>"
EOS = "<EOS>"

np.random.seed(1337)  # for reproducibility
random.seed(1337)


class CFExperiment():
    """ Main experiment class for collaborative filtering.
    Check tylib/exp/experiment.py for base class.
    """

    def __init__(self, inject_params=None):
        # 记录evalation的结果
        self.eval_test = defaultdict(list)
        self.eval_dev = defaultdict(list)
        self.parser = build_parser()
        self.args = self.parser.parse_args()

        self.show_metrics = ['RMSE', 'All_loss', 'bleu_score', 'rouge_score_1', 'rouge_score_2', 'rouge_score_L']
        self.eval_primary = 'All_loss'  # 选择epoch的最后标准是all_loss

        # For hierarchical setting
        self.args.qmax = self.args.smax * self.args.dmax
        self.args.amax = self.args.smax * self.args.dmax

        self.args.rnn_type = 'SIG_MSE_CAML_FN_FM'
        # self.args.rnn_type = 'RAW_MSE_CAML_FN_FM'
        self.args.base_encoder = 'NBOW'

        print('=' * 40, '加载数据', '=' * 40)
        self._load_sets()

        print('=' * 40, '构造模型', '=' * 40)
        self.mdl = Model(self.vocab_len, self.args, self.n_user, self.n_item, self.norm_adj_mat)

        # 设置GPU及其使用量，设置tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.7
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self):

        min_loss = 1e+7
        best_epoch = 1

        for epoch in tqdm(range(1, self.args.epochs + 1)):
            print('========================================Epoch [{}]========================================'.format(
                epoch))
            # np.random.shuffle(self.train_data)

            print("Training......")
            self.sess.run(tf.assign(self.mdl.is_train, self.mdl.true))  # 训练模型
            start = datetime.datetime.now()
            num_batches = int(self.train_len / self.args.batch_size)

            for i in tqdm(range(0, num_batches + 1)):  # tqdm是进度条
                batch = batchify(self.train_rating_set, self.train_reviews, i, self.args.batch_size,
                                 max_sample=self.train_len)

                if (len(batch) == 0):
                    continue

                batch = self._prepare_set(batch)
                feed_dict = self.mdl.get_feed_dict(batch, mode='training')

                train_loss = self.sess.run([self.mdl.cost], feed_dict)

            print('Finish all the batches: time(s):{}\n'.format(datetime.datetime.now() - start))

            print("Evaluting valid dataset......")
            self.sess.run(tf.assign(self.mdl.is_train, self.mdl.false))  # 不训练模型
            valid_loss = self.valid_evaluate()

            if min_loss > valid_loss:
                min_loss = valid_loss
                self.mdl.saver.save(self.sess, '../output/model_best_' + str(DATACNT) + '.ckpt')
                best_epoch = epoch

            if (epoch - best_epoch > self.args.early_stop and self.args.early_stop > 0):
                print("Ended at early stop, epoch [{}]".format(epoch))
                break

            print('Finish one epoch : time(s):{}\n'.format(datetime.datetime.now() - start))

        print('=' * 40, 'End of training', '=' * 40)

    def valid_evaluate(self):
        losses = []
        num_batches = int(self.valid_len / self.args.batch_size)
        for i in tqdm(range(0, num_batches + 1)):  # tqdm是进度条

            batch = batchify(self.dev_rating_set, self.dev_reviews, i, self.args.batch_size,
                             max_sample=self.valid_len)
            if (len(batch) == 0):
                continue

            batch = self._prepare_set(batch)

            feed_dict = self.mdl.get_feed_dict(batch, mode='valid')
            loss = self.sess.run([self.mdl.cost], feed_dict)
            losses.append(loss)

        return np.mean(losses)


    def test_evaluate(self):
        print("Evaluting test dataset......")
        self.mdl.saver.restore(self.sess, '../output/model_best_' + str(DATACNT) + '.ckpt')

        all_preds = []
        losses = []
        actual_labels = [float(line[2]) for line in self.test_rating_set]  # 实际评分

        rouge = Rouge()
        #bleu_score = []
        bleu_score = [[], [], [], [], []]
        rouge_score_1 = [0.0, 0.0, 0.0]  # f , p, r
        rouge_score_2 = [0.0, 0.0, 0.0]
        rouge_score_L = [0.0, 0.0, 0.0]

        start = datetime.datetime.now()
        # batch_size设置为1，方便计算每一个gen_reivew和source_review的bleu和rouge
        writer = open("../output/caml_generate_results" + str(DATACNT) + ".txt", 'w', encoding='utf-8')
        self.sess.run(tf.assign(self.mdl.is_train, self.mdl.false))  # 不训练模型
        for i in tqdm(range(self.test_len)):
            batch = batchify(self.test_rating_set, self.test_reviews, i, 1, max_sample=self.test_len)
            temp = batch[0][3]
            if len(temp) > 70:
                temp = temp[:70]
            #使用review的前30个单词作为ground truth
            source_sentence = ' '.join([self.index_word[idx] for idx in temp])
            writer.write("True review:\n")
            writer.write(source_sentence)
            writer.write("\n\n")
            batch = self._prepare_set(batch)
            if (len(batch) == 0):
                continue

            feed_dict = self.mdl.get_feed_dict(batch, mode='testing')

            gen_results, preds = self.sess.run([self.mdl.gen_results, self.mdl.output_pos], feed_dict)

            all_preds += [x[0] for x in preds]  # preds 预测评分
            # losses.append(loss)

            generated_sentence = ' '.join([self.index_word[idx] for idx in gen_results])
            writer.write("Generate sentence:\n")
            writer.write(generated_sentence)
            writer.write('\n============================================================================\n')

            bleu_0, bleu_1, bleu_2, bleu_3, bleu_4 = compute_bleu([source_sentence], 
                                                          [generated_sentence])
            bleu_score[0].append(bleu_0)

            bleu_score[1].append(bleu_1)
            bleu_score[2].append(bleu_2)
            bleu_score[3].append(bleu_3)
            bleu_score[4].append(bleu_4)

            rouge_1 = rouge.get_scores(generated_sentence, source_sentence)
            for i in range(3):
                rouge_score_1[i] += list(rouge_1[0]['rouge-1'].values())[i]
                rouge_score_2[i] += list(rouge_1[0]['rouge-2'].values())[i]
                rouge_score_L[i] += list(rouge_1[0]['rouge-l'].values())[i]

        # end of all batchs
        print('time(s):{}\n'.format(datetime.datetime.now() - start))
        writer.close()
        # sum = 0.0
        # for score in bleu_score:
        #     sum = sum + score
        # bleu_s = sum / len(bleu_score)
        bleu_score = [sum(i)/len(i) for i in bleu_score]

        rouge_score_1 = [i / self.test_len for i in rouge_score_1]
        rouge_score_2 = [i / self.test_len for i in rouge_score_2]
        rouge_score_L = [i / self.test_len for i in rouge_score_L]

        print('=' * 40, '结果', '=' * 40)
        print("bleu_score: ", bleu_score)
        print("rouge_score_1 f p r : %.4f\t%.4f\t%.4f" % (rouge_score_1[0], rouge_score_1[1], rouge_score_1[2]))
        print("rouge_score_2 f p r : %.4f\t%.4f\t%.4f" % (rouge_score_2[0], rouge_score_2[1], rouge_score_2[2]))
        print("rouge_score_L f p r : %.4f\t%.4f\t%.4f" % (rouge_score_L[0], rouge_score_L[1], rouge_score_L[2]))

        print("\n预测分数为小数时：")
        # all_preds 是0~1之间的数
        all_preds = [rescale(x) for x in all_preds]
        mse = mean_squared_error(actual_labels, all_preds)
        mae = mean_absolute_error(actual_labels, all_preds)
        print("MAE: ", mae)
        print("RMSE: ", mse ** 0.5)  # 开根号

        print("\n预测分数四舍五入转化为整数时：")
        all_preds = [int(x + 0.5) for x in all_preds]
        actual_labels = [int(x) for x in actual_labels]
        mse_int = mean_squared_error(actual_labels, all_preds)
        mae_int = mean_absolute_error(actual_labels, all_preds)
        print("MAE_int: ", mae_int)
        print("RMSE_int: ", mse_int ** 0.5)


    

    def _load_sets(self):
        print('=' * 40, 'Load dataset', '=' * 40)
        print('load vocabulary...')
        voc = load_vocabulary("../data/reviews.csv")
        self.word_index = voc.word2idx
        self.index_word = voc.idx2word
        self.vocab_len = len(self.word_index)
        print("vocab = {}\n".format(self.vocab_len))

        print('loading all dataset ...\n')
        file_path = "../data/reviews.csv"
        train_path = "../data/train.json"
        valid_path = "../data/valid.json"
        test_path = "../data/test.json"
        self.n_user, self.n_item, all_data_len = load_all_dataset(file_path)

        print("loading rating_set, reviews...\n")  # 格式为： (userid, itemid, rating) reviews，review保持原始的长度
        self.train_rating_set, self.train_reviews, self.train_len = load_dataset(train_path, self.word_index)
        self.dev_rating_set, self.dev_reviews, self.valid_len = load_dataset(valid_path, self.word_index)
        self.test_rating_set, self.test_reviews, self.test_len = load_dataset(test_path, self.word_index)
        print("all dataset length: " + str(all_data_len))
        print("trainset length: " + str(self.train_len))
        print("validset length: " + str(self.valid_len))
        print("testset length: " + str(self.test_len))


        adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(file_path, self.n_user, self.n_item)
        self.norm_adj_mat = tf.convert_to_tensor(norm_adj_mat, tf.float32)
        self.adj_mat = tf.convert_to_tensor(adj_mat, tf.float32)
        self.mean_adj_mat = tf.convert_to_tensor(mean_adj_mat, tf.float32)

        # review的长度最长为smax
        print("constructing ui_review_dict, iu_review_dict, ui_concept_dict, iu_concept_dict of train_dataset set...\n")
        self.ui_review_dict, self.iu_review_dict = load_review_data(train_path, self.word_index, self.args.smax,
                                                                    "review")
        self.ui_concept_dict, self.iu_concept_dict = load_review_data(train_path, self.word_index, self.args.smax,
                                                                      "concept")
        num_user = len(self.ui_review_dict)
        num_item = len(self.iu_review_dict)
        print("users in train dataset is: " + str(num_user))
        print("items in train dataset is: " + str(num_item) + "\n")
        print("users in all dataset: " + str(self.n_user))
        print("items in all dataset: " + str(self.n_item) + "\n")

    # data格式为：[user[i], items[i], labels[i], reviews[i], len(reviews[i])] review是原始的长度
    # results：user, user_len, items, item_len, user_concept, user_concept_len, items_concept, items_concept_len, user_idx, item_idx, gen_outputs(reviews), gen_len(reviews_len) ratings
    def _prepare_set(self, data):

        user = [x[0] for x in data]
        items = [x[1] for x in data]
        labels = [int(x[2]) for x in data]

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

        # 获得每条数据中user对应的所有的reviews item对应的所有的reviews
        for i in range(len(user)):
            user_reviews = []
            item_reviews = []
            user_concepts = []
            item_concepts = []

            user_r_len = []
            user_c_len = []
            item_r_len = []
            item_c_len = []

            # ui_review_dict中review的长度最大时smax
            if user[i] in self.ui_review_dict:
                for x in self.ui_review_dict[user[i]]:
                    user_reviews.append(self.ui_review_dict[user[i]][x])
                    user_concepts.append(self.ui_concept_dict[user[i]][x])
                    user_r_len.append(len(self.ui_review_dict[user[i]][x]))
                    user_c_len.append(len(self.ui_concept_dict[user[i]][x]))
                    # user_clicked_entity.append(x)
                    if len(user_reviews) == self.args.dmax:
                        break

            user_list.append(user_reviews)
            user_concept_list.append(user_concepts)
            user_len.append(user_r_len)
            user_concept_len.append(user_c_len)

            # entity_embs_dataset = np.load(
            #     '../kg/embeddings_dataset/' + 'entity_embeddings_' + KGE + '_' + str(self.args.entity_dim) + '_' + str(
            #         DATACNT) + '.npy')
            # max_len = len(entity_embs_dataset) - 1
            # # TODO padding
            # # 不足max_clicked_entity的填充-1
            # # 要对齐为一个矩阵，之前超过max_clicked_entity的情况没有进行处理
            # padding_len = self.args.max_clicked_entity - len(user_clicked_entity)
            # if padding_len > 0:
            #     for x in range(padding_len):
            #         user_clicked_entity.append(max_len - 1)
            # else:
            #     user_clicked_entity = user_clicked_entity[:self.args.max_clicked_entity]

            # user_clicked_entities_indices.append(user_clicked_entity)

            if items[i] in self.iu_review_dict:
                for x in self.iu_review_dict[items[i]]:
                    item_reviews.append(self.iu_review_dict[items[i]][x])
                    item_concepts.append(self.iu_concept_dict[items[i]][x])
                    item_r_len.append(len(self.iu_review_dict[items[i]][x]))
                    item_c_len.append(len(self.iu_concept_dict[items[i]][x]))
                    if len(item_reviews) == self.args.dmax:
                        break
            item_list.append(item_reviews)
            item_concept_list.append(item_concepts)
            item_len.append(item_r_len)
            item_concept_len.append(item_c_len)

        # 对于每个user和item，其相关的review padding到dmax条，每条smax个单词
        user_concept, user_concept_len = prep_hierarchical_data_list_new(user_concept_list, user_concept_len,
                                                                         self.args.smax,
                                                                         self.args.dmax)
        items_concept, item_concept_len = prep_hierarchical_data_list_new(item_concept_list, item_concept_len,
                                                                          self.args.smax,
                                                                          self.args.dmax)

        user, user_len = prep_hierarchical_data_list_new(user_list, user_len,
                                                         self.args.smax,
                                                         self.args.dmax)
        items, item_len = prep_hierarchical_data_list_new(item_list, item_len,
                                                          self.args.smax,
                                                          self.args.dmax)

        output = [user, user_len, items, item_len]
        register_index_map(self.mdl.imap, 0, 'q1_inputs')
        register_index_map(self.mdl.imap, 1, 'q1_len')
        register_index_map(self.mdl.imap, 2, 'q2_inputs')
        register_index_map(self.mdl.imap, 3, 'q2_len')

        output.append(user_concept)
        output.append(user_concept_len)
        output.append(items_concept)
        output.append(item_concept_len)
        register_index_map(self.mdl.imap, 4, 'c1_inputs')
        register_index_map(self.mdl.imap, 5, 'c1_len')
        register_index_map(self.mdl.imap, 6, 'c2_inputs')
        register_index_map(self.mdl.imap, 7, 'c2_len')

        output.append(user_idx)
        output.append(item_idx)
        # output.append(entity_idx)
        # output.append(user_clicked_entities_indices)
        register_index_map(self.mdl.imap, 8, 'user_indices')
        register_index_map(self.mdl.imap, 9, 'item_indices')
        # register_index_map(self.mdl.imap, 10, 'entity_indices')
        # register_index_map(self.mdl.imap, 11, 'user_clicked_entities_indices')

        # 保存的是原始的review和其长度
        gen_outputs = [x[3] for x in data]
        gen_len = [x[4] for x in data]

        output.append(gen_outputs)
        output.append(gen_len)
        register_index_map(self.mdl.imap, 10, 'gen_outputs')
        register_index_map(self.mdl.imap, 11, 'gen_len')

        output.append(labels)
        output = list(zip(*output))
        return output


if __name__ == '__main__':
    srt = datetime.datetime.now()
    exp = CFExperiment(inject_params=None)
    exp.train()
    exp.test_evaluate()
    print('all time(s):{}\n'.format(datetime.datetime.now() - srt))
