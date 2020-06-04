import json
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from time import time


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
        self.idx2word = {0: "", 1: "", 2: " ", 3: " "}
        # self.n_words = 4
        self.word_freq = {}

    def build_sentence(self, sentence):
        li = re.findall(r'\b\w+\b', sentence)
        for word in li:
            if word not in self.word_freq:
                self.word_freq[word] = 1
            else:
                self.word_freq[word] += 1


# 统计review中词频，并返回
def load_vocabulary(file_name):
    voc = Vocabulary()
    f = pd.read_csv(file_name)
    for r in f['review']:
        voc.build_sentence(r)
    ordered_word_freq = sorted(voc.word_freq.items(), key=lambda x: x[1], reverse=True)
    n_words = 4
    for triple in ordered_word_freq:
        key = triple[0]
        if key not in voc.word2idx.keys():
            voc.word2idx[key] = n_words
            voc.idx2word[n_words] = key
            n_words = n_words + 1
            if n_words > 40003:
                break
    return voc


def load_all_dataset(file_name):
    # output:
    # N: the number of users
    # M: the number of items
    # data: the list of rating information
    reviews = pd.read_csv(file_name)
    return len(set(list(reviews['user']))), len(set(list(reviews['item']))), reviews.shape[0]


# data: train_dataset/ valid/ test 数据集格式：item, rating, userid, review
# 返回：(userid, itemid, rating) reviews
def load_dataset(train_path, word_index):
    output = []
    reviews = []
    reader = open(train_path, 'r', encoding="utf-8")
    alllist = json.load(reader)
    cnt = 0
    for line in alllist:
        cnt = cnt + 1
        list = re.findall(r'\b\w+\b', line["review"])
        linedata = []
        linedata.append(word_index["<EOS>"])
        for word in list:
            if word in word_index:
                linedata.append(word_index[word])
            else:
                linedata.append(word_index["<UNK>"])
        linedata.append(word_index["<SOS>"])
        output.append((line["user"], line["item"], line["rating"]))
        reviews.append(linedata)
    reader.close()
    return output, reviews, cnt


# data_type: review/ concept
# results: ui_dict iu_dict
def load_review_data(train_path, word_index, smax, data_type):
    users = []
    items = []
    reviews = []

    reader = open(train_path, 'r', encoding="utf-8")
    list = json.load(reader)
    for line in list:
        users.append(line["user"])
        items.append(line["item"])
        reviews.append(line["review"])

    ui_dict = {}
    iu_dict = {}

    for i in range(len(reviews)):
        review = reviews[i]
        user = users[i]
        item = items[i]
        linedata = []
        list = re.findall(r'\b\w+\b', review)
        linedata.append(word_index["<EOS>"])
        for word in list:
            if word in word_index:
                linedata.append(word_index[word])
            else:
                linedata.append(word_index["<UNK>"])
        linedata.append(word_index["<SOS>"])

        if len(linedata) > smax:
            linedata = linedata[:smax]

        if user not in ui_dict:
            ui_dict[user] = {}
        if item not in iu_dict:
            iu_dict[item] = {}

        ui_dict[user][item] = linedata
        iu_dict[item][user] = linedata
    reader.close()
    return ui_dict, iu_dict


# def get_adj_mat(path, file_name, n_users, n_items):
#     try:
#         t1 = time()
#         adj_mat = sp.load_npz(path + '/s_adj_mat.npz')
#         norm_adj_mat = sp.load_npz(path + '/s_norm_adj_mat.npz')
#         mean_adj_mat = sp.load_npz(path + '/s_mean_adj_mat.npz')
#         print('already load adj matrix', adj_mat.shape, time() - t1)

#     except Exception:
#         adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(file_name, n_users, n_items)
#         sp.save_npz(path + '/s_adj_mat.npz', adj_mat)
#         sp.save_npz(path + '/s_norm_adj_mat.npz', norm_adj_mat)
#         sp.save_npz(path + '/s_mean_adj_mat.npz', mean_adj_mat)
#     return adj_mat, norm_adj_mat, mean_adj_mat


# def _convert_sp_mat_to_sp_tensor(self, X):
#         coo = X.tocoo().astype(np.float32)
#         indices = np.mat([coo.row, coo.col]).transpose()
#         return tf.SparseTensor(indices, coo.data, coo.shape)


def create_adj_mat(file_name, n_users, n_items):
    # R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    R = np.zeros((n_users, n_items))
    f = pd.read_csv(file_name)

    for r in range(f.shape[0]):
        # print('$'*40, type(f['user'][r]))
        R[f['user'][r], f['item'][r]] = 1.
    t1 = time()
    # adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = np.zeros((n_users + n_items, n_users + n_items))
    # adj_mat = adj_mat.tolil()
    # R = R.tolil()

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    # adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape, time() - t1)

    t2 = time()

    def normalized_adj_single(adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        # return norm_adj.tocoo()
        return norm_adj

    # def check_adj_if_equal(adj):
    #     dense_A = np.array(adj.todense())
    #     degree = np.sum(dense_A, axis=1, keepdims=False)

    #     temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
    #     print('check normalized adjacency matrix whether equal to this laplacian matrix.')
    #     return temp

    # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    norm_adj_mat = normalized_adj_single(adj_mat + np.eye(adj_mat.shape[0]))
    mean_adj_mat = normalized_adj_single(adj_mat)

    print('already normalize adjacency matrix', time() - t2)
    # return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
    return adj_mat.astype(np.float32), norm_adj_mat.astype(np.float32), mean_adj_mat.astype(np.float32)


'''
#统计review中词频，并返回
def load_vocabulary(file_pos):
    voc = Vocabulary()
    reader = open(file_pos, 'r', encoding="utf-8")
    list = json.load(reader)
    n_words = 4
    for dict in list:
        review = re.findall(r'\b\w+\b',dict['review'])
        for word in review:
            if word not in voc.word2idx:
                voc.word2idx[word] = n_words
                voc.idx2word[n_words] = word
                n_words = n_words + 1
    reader.close()
    return voc
'''
