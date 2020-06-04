#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops.rnn_cell import GRUCell

import sys
sys.path.append(r"/home/ubuntu/PJ/")

from tylib.lib.att_op import *
from tylib.lib.cnn import *
from tylib.lib.compose_op import *
from utilities import pad_to_max
# from src2.utilities import pad_to_max

import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



# 模型
class Model:
    def __init__(self, vocab_size, args, num_user, num_item, norm_adj):
        self.norm_adj = norm_adj
        self.build_parse_args(vocab_size, args, num_user, num_item)

        # with self.graph.as_default():
        self.build_inputs()
        self.build_model()
        self.build_train()

    def build_parse_args(self, vocab_size, args, num_user, num_item):
        self.vocab_size = vocab_size
        self.args = args
        self.imap = {}  # 记录的是metric信息
        self.num_user = num_user
        self.num_item = num_item
        self.rnn_type = args.rnn_type
        self.batch_size = args.batch_size
        self.method = "all"

        # 填充词
        self.PAD_tag = 0  # 补全字符
        self.UNK_tag = 1  # 低频词
        self.SOS_tag = 2
        self.EOS_tag = 3

        #TODO
        self.layer_size = [64]
        self.n_layers = 1
        self.emb_dim = self.args.emb_size
        self.weight_size = self.layer_size
        self.n_layers = len(self.weight_size)
        self.weights = self._init_weights()
        self.mess_dropout = [0.1]

        self.initializer = tf.contrib.layers.xavier_initializer()

    def _init_weights(self):
            all_weights = dict()

            initializer = tf.contrib.layers.xavier_initializer()

            self.weight_size_list = [self.emb_dim] + self.weight_size

            for k in range(self.n_layers):
                all_weights['W_gc_%d' %k] = tf.Variable(
                    initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
                all_weights['b_gc_%d' %k] = tf.Variable(
                    initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

                all_weights['W_bi_%d' % k] = tf.Variable(
                    initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
                all_weights['b_bi_%d' % k] = tf.Variable(
                    initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

                all_weights['W_%d' % k] = tf.Variable(
                    initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_%d' % k)
                all_weights['b_%d' % k] = tf.Variable(
                    initializer([1, self.weight_size_list[k + 1]]), name='b_%d' % k)

            return all_weights


    def _create_gatne_embed(self):
        norm_adj = self.norm_adj
        ego_embeddings = tf.concat([self.user_emb_matrix, self.item_emb_matrix], axis=0)

        for k in range(0, self.n_layers):

            temp_embeddings = tf.matmul(norm_adj, ego_embeddings)

            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(temp_embeddings, self.weights['W_%d' % k]) + self.weights['b_%d' % k])

            ego_embeddings = sum_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            ego_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

        u_g_embeddings, i_g_embeddings = tf.split(ego_embeddings, [self.num_user, self.num_item], 0)
        return u_g_embeddings, i_g_embeddings


    def _create_ngcf_embed(self):

        norm_adj = self.norm_adj

        ego_embeddings = tf.concat([self.user_emb_matrix, self.item_emb_matrix], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []

            a = tf.matmul(norm_adj, ego_embeddings)

            temp_embed.append(a)

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)

            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_user, self.num_item], 0)
        return u_g_embeddings, i_g_embeddings


    def build_inputs(self):

        with tf.name_scope('q1_input'):
            self.q1_inputs = tf.placeholder(tf.int32, shape=[None,
                                                             self.args.qmax],
                                            name='q1_inputs')
        with tf.name_scope('q2_input'):
            self.q2_inputs = tf.placeholder(tf.int32, shape=[None,
                                                             self.args.amax],
                                            name='q2_inputs')

        with tf.name_scope('c1_input'):
            self.c1_inputs = tf.placeholder(tf.int32, shape=[None,
                                                             self.args.qmax],
                                            name='c1_inputs')
        with tf.name_scope('c2_input'):
            self.c2_inputs = tf.placeholder(tf.int32, shape=[None,
                                                             self.args.amax],
                                            name='c2_inputs')

        with tf.name_scope('gen_output'):
            self.gen_outputs = tf.placeholder(tf.int32, shape=[None,
                                                               # self.args.gmax],
                                                               None],
                                              name='gen_outputs')

        with tf.name_scope('dropout'):
            self.dropout = tf.placeholder(tf.float32,
                                          name='dropout')
            self.rnn_dropout = tf.placeholder(tf.float32,
                                              name='rnn_dropout')
            self.emb_dropout = tf.placeholder(tf.float32,
                                              name='emb_dropout')
        with tf.name_scope('q1_lengths'):
            self.q1_len = tf.placeholder(tf.int32, shape=[None, None])
        with tf.name_scope('q2_lengths'):
            self.q2_len = tf.placeholder(tf.int32, shape=[None, None])
        with tf.name_scope('c1_lengths'):
            self.c1_len = tf.placeholder(tf.int32, shape=[None, None])
        with tf.name_scope('c2_lengths'):
            self.c2_len = tf.placeholder(tf.int32, shape=[None, None])

        with tf.name_scope('user_indices'):
            self.user_indices = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('item_indices'):
            self.item_indices = tf.placeholder(tf.int32, shape=[None])
        # TODO
        # with tf.name_scope('entity_indices'):
        #     self.entity_indices = tf.placeholder(tf.int32, shape=[None])
        # with tf.name_scope('user_clicked_entities_indices'):
        #     self.user_clicked_entities_indices = tf.placeholder(tf.int32, shape=[None, self.args.max_clicked_entity])

        with tf.name_scope('gen_len'):
            self.gen_len = tf.placeholder(tf.int32, shape=[None])

        with tf.name_scope('learn_rate'):
            self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')

        with tf.name_scope("sig_labels"):  # sigmoid cross entropy
            self.sig_labels = tf.placeholder(tf.float32,
                                             shape=[None],
                                             name='sigmoid_labels')

    def build_model(self):
        with tf.variable_scope('embedding_layer'):
            self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
            self.true = tf.constant(True, dtype=tf.bool)
            self.false = tf.constant(False, dtype=tf.bool)

            self.embeddings = tf.get_variable('embedding',
                                              [self.vocab_size,
                                               self.args.emb_size],
                                              initializer=self.initializer)

            self.user_emb_matrix = tf.get_variable('user_emb_matrix',
                                                   [self.num_user,
                                                    self.args.emb_size],
                                                   initializer=self.initializer)

            self.item_emb_matrix = tf.get_variable('item_emb_matrix',
                                                   [self.num_item,
                                                    self.args.emb_size],
                                                   initializer=self.initializer)

            # [batch_size, emb_size] 使用查找层将用户u（或项目v）转换为隐式表示
            self.user_batch = tf.nn.embedding_lookup(self.user_emb_matrix,
                                                     self.user_indices)

            self.item_batch = tf.nn.embedding_lookup(self.item_emb_matrix,
                                                     self.item_indices)

        # Hierarchical mode
        self.prepare_hierarchical_input()
        q1_len = tf.cast(tf.count_nonzero(self.q1_len, axis=1), tf.int32)
        q2_len = tf.cast(tf.count_nonzero(self.q2_len, axis=1), tf.int32)

        print('hierarchical q1_embed:', self.q1_embed.get_shape())
        print('hierarchical q2_embed:', self.q2_embed.get_shape())
        print("=================================================")

        print("========================Learning joint_representation===============================")
        # [bs, 50], [bs, 50],结果是图中的x_u x_v

        self.i1_embed, self.i2_embed, self.i3_embed = None, None, None
        self.q1_output, self.q2_output = self._joint_representation(self.q1_embed, self.q2_embed, q1_len, q2_len,
                                                                    self.qmax, self.a1max, score=1, reuse=None,
                                                                    features=None, extract_embed=True, side='POS',
                                                                    o1_embed=self.o1_embed, o2_embed=self.o2_embed,
                                                                    o1_len=self.o1_len, o2_len=self.o2_len,
                                                                    q1_mask=self.q1_mask, q2_mask=self.q2_mask)


        u_embeddings, i_embeddings = self._create_ngcf_embed()
        u_g_embeddings, i_g_embeddings = self._create_gatne_embed()

        u_g_embeddings = tf.nn.embedding_lookup(u_g_embeddings, self.user_indices)
        i_g_embeddings = tf.nn.embedding_lookup(i_g_embeddings, self.item_indices)

        u_embeddings = tf.nn.embedding_lookup(u_embeddings, self.user_indices)
        i_embeddings = tf.nn.embedding_lookup(i_embeddings, self.item_indices)

        # 拼接
        if(self.method == 'all'):
            self.q1_output = tf.concat([self.q1_output, u_g_embeddings, u_embeddings], 1)
            self.q2_output = tf.concat([self.q2_output, i_g_embeddings, i_embeddings], 1)
        elif(self.method == 'ngcf'):
            self.q1_output = tf.concat([self.q1_output, u_embeddings], 1)
            self.q2_output = tf.concat([self.q2_output, i_embeddings], 1)
        elif(self.method == 'gatne'):
            self.q1_output = tf.concat([self.q1_output, u_g_embeddings], 1)
            self.q2_output = tf.concat([self.q2_output, i_g_embeddings], 1)
        # else self.method == 'caml', no operation here
        # self.q1_output = u_g_embeddings
        # self.q2_output = i_g_embeddings

        print("Concatenate additional info which gets from Trans, ")
        print("q1_output: ")
        print(self.q1_output)
        print("q2_output: ")
        print(self.q2_output)
        print("================================================")

        # self.output_neg = None
        # 构造fm，获得分解机fm的prediction
        self.output_pos = self._rec_output(self.q1_output, self.q2_output,
                                           reuse=None,
                                           side='POS')
        print("================================================")

    def build_train(self):
        # Define loss and optimizer
        with tf.name_scope("train_dataset"):
            # 计算cost
            with tf.name_scope("cost_function"):
                sig = self.output_pos  # sig代表的是预测评分
                target = tf.expand_dims(self.sig_labels, 1)  # sig_labels是rating
                self.cost = tf.reduce_mean(
                    tf.square(tf.subtract(target, sig)))

                with tf.name_scope('generation_results'):
                    self.gen_results = self._beam_search_infer(self.q1_output, self.q2_output)

                with tf.name_scope('generation_loss'):
                    self.gen_loss, self.gen_acc = self._gen_review(self.q1_output, self.q2_output)
                    self.cost += self.args.gen_lambda * self.gen_loss

                # self.task_cost = self.cost

                with tf.name_scope('regularization'):
                    if (self.args.l2_reg > 0):
                        vars = tf.trainable_variables()
                        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars \
                                           if 'bias' not in v.name])
                        lossL2 *= self.args.l2_reg
                        self.cost += lossL2

            control_deps = []
            with tf.name_scope('optimizer'):
                self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
                tvars = tf.trainable_variables()

                global_step = tf.Variable(0, trainable=False)

                def _none_to_zero(grads, var_list):
                    return [grad if grad is not None else tf.zeros_like(var)
                            for var, grad in zip(var_list, grads)]

                if (self.args.clip_norm > 0):
                    grads, _ = tf.clip_by_global_norm(
                        tf.gradients(self.cost, tvars),
                        self.args.clip_norm)
                    with tf.name_scope('gradients'):
                        gradients = self.opt.compute_gradients(self.cost)

                        def ClipIfNotNone(grad):
                            if grad is None:
                                return grad
                            grad = tf.clip_by_value(grad, -10, 10, name=None)
                            return tf.clip_by_norm(grad, self.args.clip_norm)

                        if (self.args.clip_norm > 0):
                            clip_g = [(ClipIfNotNone(grad), var) for grad, var in gradients]
                        else:
                            clip_g = [(grad, var) for grad, var in gradients]

                    # Control dependency for center loss
                    with tf.control_dependencies(control_deps):
                        self.train_op = self.opt.apply_gradients(clip_g,
                                                                 global_step=global_step)
                else:
                    with tf.control_dependencies(control_deps):
                        self.train_op = self.opt.minimize(self.cost)

            # save model
            self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

    # 将外面传入的原始数据构造为本模型的输入数据 feed_dict
    def get_feed_dict(self, data, mode='training', lr=None):
        """ This is the pointwise feed-dict that is actually used.
        """

        # print("data.type", type(data))
        data = list(zip(*data))
        # print("data.shape", len(data))
        labels = data[-1]  # rating
        sig_labels = labels

        feed_dict = dict()
        feed_dict[self.q1_inputs] = data[self.imap['q1_inputs']]
        feed_dict[self.q2_inputs] = data[self.imap['q2_inputs']]
        feed_dict[self.q1_len] = data[self.imap['q1_len']]
        feed_dict[self.q2_len] = data[self.imap['q2_len']]
        feed_dict[self.c1_inputs] = data[self.imap['c1_inputs']]
        feed_dict[self.c2_inputs] = data[self.imap['c2_inputs']]
        feed_dict[self.c1_len] = data[self.imap['c1_len']]
        feed_dict[self.c2_len] = data[self.imap['c2_len']]
        feed_dict[self.dropout] = self.args.dropout
        feed_dict[self.rnn_dropout] = self.args.rnn_dropout
        feed_dict[self.emb_dropout] = self.args.emb_dropout
        feed_dict[self.sig_labels] = sig_labels
        feed_dict[self.learn_rate] = self.args.learn_rate

        feed_dict[self.user_indices] = data[self.imap['user_indices']]
        feed_dict[self.item_indices] = data[self.imap['item_indices']]
        # feed_dict[self.entity_indices] = data[self.imap['entity_indices']]  # entity_indices和item_indices是相同的
        # feed_dict[self.user_clicked_entities_indices] = data[self.imap['user_clicked_entities_indices']]

        if (mode != 'testing'):
            # 实际的review以及其长度，将其padding到review长度中最长的
            feed_dict[self.gen_len] = data[self.imap['gen_len']]
            max_len = 0
            for i in range(len(feed_dict[self.gen_len])):
                if max_len < feed_dict[self.gen_len][i]:
                    max_len = feed_dict[self.gen_len][i]
            padding_outputs = []
            for i in range(len(data[self.imap['gen_outputs']])):
                padding_outputs.append(pad_to_max(data[self.imap['gen_outputs']][i], max_len))
            feed_dict[self.gen_outputs] = padding_outputs

        if (mode != 'training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.rnn_dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
        return feed_dict

    def _joint_representation(self, q1_embed, q2_embed, q1_len, q2_len, q1_max,
                              q2_max, score=1,
                              reuse=None, features=None, extract_embed=False,
                              side='', o1_embed=None,
                              o2_embed=None, o1_len=None, o2_len=None, q1_mask=None,
                              q2_mask=None):
        """ Learns a joint representation given q1 and q2.
        """

        # q1_embed q2_embed原本输入的 user/item 向量表达
        use_mode = 'FC'

        # Extra projection layer
        # 对最初的embedding先经过一个MLP

        q1_embed = projection_layer(
            q1_embed,
            self.args.emb_size,
            name='trans_proj',
            activation=tf.nn.relu,
            initializer=self.initializer,
            dropout=self.args.dropout,
            reuse=reuse,
            use_mode=use_mode,
            num_layers=self.args.num_proj,
            return_weights=True,
            is_train=self.is_train
        )

        q2_embed = projection_layer(
            q2_embed,
            self.args.emb_size,
            name='trans_proj',
            activation=tf.nn.relu,
            initializer=self.initializer,
            dropout=self.args.dropout,
            reuse=True,
            use_mode=use_mode,
            num_layers=self.args.num_proj,
            is_train=self.is_train
        )

        # 单序列编码器功能
        # rnn_type controls what type of encoder is used.
        # Supports neural bag-of-words (NBOW) and CNN encoder
        _, q1_output = self.learn_single_repr(q1_embed, q1_len, q1_max,
                                              self.args.rnn_type,
                                              reuse=reuse, pool=False,
                                              name='main', mask=q1_mask)

        _, q2_output = self.learn_single_repr(q2_embed, q2_len, q2_max,
                                              self.args.rnn_type,
                                              reuse=True, pool=False,
                                              name='main', mask=q2_mask)

        # q1_output/q2_output分别代表user/item的特征表达
        print('Single Representation:')
        print('q1_output shape:', q1_output.shape)
        print('q2_output shape:', q2_output.shape)
        print("q2_output: ")
        print(q2_output)
        print("===============================================")

        # activate MPCN model
        # 经过co-attention，得到user/item在word/review级别拼接后的特征表达
        # 从多层次的 [?, ?, 50] 转换到 [?, 50]
        # 得到的最后结果是多次进行select后的拼接，是图中的e_u
        q1_output, q2_output = self.multi_pointer_coattention_networks(
            q1_output, q2_output,
            q1_len, q2_len,
            o1_embed, o2_embed,
            o1_len, o2_len,
            rnn_type=self.args.rnn_type,
            reuse=reuse)

        print("After multi_pointer_coattention_networks, ")
        print("q1_output: ")
        print(q1_output)
        print("q2_output: ")
        print(q2_output)
        print("================================================")

        try:
            # For summary statistics
            self.max_norm = tf.reduce_max(tf.norm(q1_output, ord='euclidean', keepdims=True, axis=1))
        except:
            self.max_norm = 0

        if (extract_embed):
            self.q1_extract = q1_output
            self.q2_extract = q2_output

        # userbatch可以看做是userid； self.user_batch = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        # 认为userbatch是# [batch_size, emb_size] ，user implicit facctor
        q1_output = tf.concat([q1_output, self.user_batch], 1)  # 拼接
        q2_output = tf.concat([q2_output, self.item_batch], 1)

        q1_output = tf.nn.dropout(q1_output, self.dropout)
        q2_output = tf.nn.dropout(q2_output, self.dropout)

        print("Concatenate implicit factor: ")
        print("q1_output: ")
        print(q1_output)
        print("q2_output: ")
        print(q2_output)
        print("================================================")

        return q1_output, q2_output

    def learn_single_repr(self, q1_embed, q1_len, q1_max, base_encoder,
                          reuse=None, pool=False, name="", mask=None):
        """ This is the single sequence encoder function.
        rnn_type controls what type of encoder is used.
        Supports neural bag-of-words (NBOW) and CNN encoder
        """
        if ('NBOW' in base_encoder):
            if mask is not None:
                masks = tf.cast(mask, tf.float32)
                masks = tf.expand_dims(masks, 2)
                masks = tf.tile(masks, [1, 1, self.args.emb_size])
                q1_embd = q1_embed * masks

            q1_output = tf.reduce_sum(q1_embed, 1)
            if (pool):
                return q1_embed, q1_output
        elif ('CNN' in base_encoder):
            q1_output = build_raw_cnn(q1_embed, self.args.emb_size,
                                      filter_sizes=3,
                                      initializer=self.initializer,
                                      dropout=self.rnn_dropout, reuse=reuse, name=name)
            if (pool):
                q1_output = tf.reduce_max(q1_output, 1)
                return q1_output, q1_output
        else:
            q1_output = q1_embed
        return q1_embed, q1_output

    def build_single_cell(self):
        # rnn_cell = LSTMCell
        rnn_cell = GRUCell
        cell = rnn_cell(self.args.emb_size)
        return cell

    def _cal_key_loss(self, preds):
        if self.args.word_gumbel == 0:
            return tf.constant(0.0, dtype=tf.float32)

        preds = -tf.log(preds + 1e-7)
        word = tf.concat([self.word_u, self.word_i], 1)
        num_words = word.get_shape().as_list()[1]
        prefix = tf.range(self.batch_size)
        prefix = tf.tile(tf.expand_dims(prefix, 1), [1, num_words])
        indices = tf.concat([tf.expand_dims(prefix, 2), tf.expand_dims(word, 2)], 2)

        l = tf.gather_nd(preds, indices)
        if self.args.word_aggregate == 'MEAN':
            l = tf.reduce_mean(l, 1)
        else:
            if self.args.word_aggregate == 'MAX':
                l = tf.reduce_max(l, 1)
            else:
                l = tf.reduce_min(l, 1)
        l = self.args.key_word_lambda * tf.reduce_mean(l)
        return l

    def _gen_review(self, q1_output, q2_output, reuse=None):
        print("Gen Output")
        dim = q1_output.get_shape().as_list()[1]
        batch_dim = q1_output.get_shape().as_list()[0]

        with tf.variable_scope('gen_review', reuse=reuse) as scope:
            # cal state 把用户完整表示和物品完整表示和评分合并到GRU中，计算出初始隐藏状态
            state = tf.nn.tanh(tf.matmul(q1_output, self.review_user_mapping) + tf.matmul(q2_output,
                                                                                          self.review_item_mapping) + self.review_bias)
            # 希望生成的解释尽可能出现挑选的关键单词
            review_inputs_transpose = tf.transpose(self.gen_outputs, perm=[1, 0])
            max_review_length = tf.reduce_max(self.gen_len)
            masks = tf.transpose(tf.sequence_mask(self.gen_len, maxlen=max_review_length, dtype=tf.float32),
                                 perm=[1, 0])

            # predict first word
            self.dropout_out = tf.nn.dropout(state, self.dropout)
            final_input = self.dropout_out

            logits = tf.layers.dense(final_input, self.vocab_size, kernel_initializer=self.initializer,
                                     bias_initializer=self.initializer, name='review_output_layer', reuse=True)
            # 输出单词的概率分布
            self.preds = tf.nn.softmax(logits)

            # if self.args.concept == 1:
            key_loss = self._cal_key_loss(self.preds)

            init_sum_key_loss = key_loss
            # 每一步选择概率最大的单词 作为该时间步的输出结果
            argm = tf.argmax(self.preds, axis=1, output_type=tf.int32)
            castm = tf.cast(argm, tf.int32)
            correct = tf.equal(castm, tf.gather(review_inputs_transpose, 1))

            init_accuracy = tf.reduce_sum(tf.cast(correct, tf.float32) * tf.gather(masks, 1))

            init_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.gather(review_inputs_transpose, 1),
                                                               logits=logits) * tf.gather(masks, 1))

            init_sum = tf.reduce_sum(tf.gather(masks, 1))
            init_iteration = tf.constant(1, dtype=tf.int32)

            def condition(i, *args):
                return i < max_review_length - 1

            def forward_one_step(i, sum, loss, sum_key_loss, accuracy, state):
                self.review_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.embeddings, ids=tf.gather(review_inputs_transpose, i))
                self.rnnlm_outputs, new_state = self.rnn_cell(self.review_inputs_embedded, state)

                # dropout before output layer
                self.dropout_out = tf.nn.dropout(self.rnnlm_outputs, self.dropout)
                final_input = self.dropout_out

                # output_layer
                logits = tf.layers.dense(final_input, self.vocab_size, kernel_initializer=self.initializer,
                                         bias_initializer=self.initializer, name='review_output_layer', reuse=True)
                # masking loss
                self.preds = tf.nn.softmax(logits)
                # 每一步选择概率最大的单词 作为该时间步的输出结果
                argm = tf.argmax(self.preds, axis=1, output_type=tf.int32)
                castm = tf.cast(argm, tf.int32)

                # if self.args.concept == 1:
                key_loss = self._cal_key_loss(self.preds)
                sum_key_loss += key_loss

                correct = tf.equal(castm, tf.gather(review_inputs_transpose, i + 1))
                accuracy = accuracy + tf.reduce_sum(tf.cast(correct, tf.float32) * tf.gather(masks, i + 1))
                sum = sum + tf.reduce_sum(masks[i + 1])

                loss = loss + tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.gather(review_inputs_transpose, i + 1),
                                                                   logits=logits) * tf.gather(masks, i + 1))
                return i + 1, sum, loss, sum_key_loss, accuracy, new_state

            iteration, sum, loss, sum_key_loss, accuracy, state = tf.while_loop(condition, forward_one_step,
                                                                                [init_iteration, init_sum, init_loss,
                                                                                 init_sum_key_loss, init_accuracy,
                                                                                 state])

            # sum += batch_dim
            review_loss = loss  # / batch_dim# / sum
            review_acc = accuracy / sum
        return review_loss, review_acc

    def prepare_hierarchical_input(self):
        """ Supports hierarchical data input
        Converts word level -> sentence level 
        """
        # Build word-level masks
        self.q1_mask = tf.cast(self.q1_inputs, tf.bool)
        self.q2_mask = tf.cast(self.q2_inputs, tf.bool)

        def make_hmasks(inputs, smax):
            # Hierarchical Masks
            # Inputs are bsz x (dmax * smax)
            inputs = tf.reshape(inputs, [-1, smax])
            masked_inputs = tf.cast(inputs, tf.bool)
            return masked_inputs

        # Build review-level masks
        self.q1_hmask = make_hmasks(self.q1_inputs, self.args.smax)
        self.q2_hmask = make_hmasks(self.q2_inputs, self.args.smax)

        # with tf.device('/gpu:0'):
        # self.q1_inputs: [bs, 600]
        q1_embed = tf.nn.embedding_lookup(self.embeddings, self.q1_inputs)
        q2_embed = tf.nn.embedding_lookup(self.embeddings, self.q2_inputs)

        print("=============================================")
        # This is found in nn.py in tylib
        print("Hierarchical Flattening")
        q1_embed, q1_len = hierarchical_flatten(q1_embed, self.q1_len, self.args.smax)
        q2_embed, q2_len = hierarchical_flatten(q2_embed, self.q2_len, self.args.smax)

        self.o1_embed = q1_embed
        self.o2_embed = q2_embed

        self.o1_len = q1_len
        self.o2_len = q2_len

        # 获取最原始的embedding 我猜测是把多层的转换为1维，再在1维的基础上做

        _, q1_embed = self.learn_single_repr(q1_embed, q1_len, self.args.smax,
                                             self.args.base_encoder,
                                             reuse=None, pool=True,
                                             name='sent', mask=self.q1_hmask)
        _, q2_embed = self.learn_single_repr(q2_embed, q2_len, self.args.smax,
                                             self.args.base_encoder,
                                             reuse=True, pool=True,
                                             name='sent', mask=self.q2_hmask)

        _dim = q1_embed.get_shape().as_list()[1]
        q1_embed = tf.reshape(q1_embed, [-1, self.args.dmax, _dim])
        q2_embed = tf.reshape(q2_embed, [-1, self.args.dmax, _dim])

        self.q1_embed = q1_embed
        self.q2_embed = q2_embed

        self.qmax = self.args.dmax
        self.a1max = self.args.dmax
        self.a2max = self.args.dmax

    def _rec_output(self, q1_output, q2_output, reuse=None, side="",
                    name=''):
        """ This function supports the final layer outputs of
        recommender models.

        Four options: 'DOT','MLP','MF' and 'FM'
        (should be self-explanatory)
        """
        print("Recommender Output")
        with tf.variable_scope('rec_out', reuse=reuse) as scope:
            if (q2_output is None):
                input_vec = q1_output
            else:
                print("Q1_OUTPUT", q1_output.shape)
                print("Q2_OUTPUT", q2_output.shape)
                input_vec = tf.concat([q1_output, q2_output], 1)
            input_vec = tf.nn.dropout(input_vec, self.dropout)
            # output 指的是 fm_prediction
            output, _ = build_fm(input_vec, k=self.args.factor,
                                 reuse=reuse,
                                 name=name,
                                 initializer=self.initializer,
                                 reshape=False)
            if ('SIG' in self.args.rnn_type):
                output = tf.nn.sigmoid(output)
            return output


    # 这是直接取top-1的文本搜索，生成解释的文本
    def _beam_search_infer(self, q1_output, q2_output, reuse=None):
        # [1,50]
        dim = q1_output.get_shape().as_list()[1]  # 50
        # max_review_length = tf.reduce_max(self.gen_len)
        # max_review_length = tf.constant(70, dtype=tf.int32)
        # MODIFIED
        max_review_length = tf.constant(70, dtype=tf.int32)
        with tf.variable_scope('gen_review', reuse=reuse) as scope:
            # cal state 计算初始隐藏层的状态
            self.review_user_mapping = tf.get_variable(name='review_user_mapping',
                                                       shape=[dim, self.args.emb_size],
                                                       initializer=self.initializer)  # , dtype=self.dtype)
            self.review_item_mapping = tf.get_variable(name='review_item_mapping',
                                                       shape=[dim, self.args.emb_size],
                                                       initializer=self.initializer)  # , dtype=self.dtype)

            if not (self.args.feed_rating == 0):
                self.review_rating_embeddings = tf.get_variable(name='review_rating_embeddings',
                                                                shape=[5, self.args.emb_size],
                                                                initializer=self.initializer)  # , dtype=self.dtype)
            self.review_bias = tf.get_variable(name='review_bias',
                                               shape=[self.args.emb_size],
                                               initializer=self.initializer)  # , dtype=self.dtype)
            #GRUCell
            self.rnn_cell = self.build_single_cell()
            #计算初始隐藏层的状态
            state = tf.nn.tanh(tf.matmul(q1_output, self.review_user_mapping) +
                               tf.matmul(q2_output, self.review_item_mapping) + self.review_bias)

            self.dropout_out = tf.nn.dropout(state, self.dropout)
            final_input = state

            # [bs, n_words]
            logits = tf.layers.dense(final_input, self.vocab_size, kernel_initializer=self.initializer,
                                     bias_initializer=self.initializer, name='review_output_layer')
            #得到时间t时的输出单词的概率分布
            self.preds = tf.nn.softmax(logits)

            # [bs, ] 每一步选择概率最大的单词 作为该时间步的输出结果
            argm = tf.argmax(self.preds, axis=1, output_type=tf.int32)

            init_answer = tf.cast(argm, tf.int32)  # [bs,]
            init_lens = tf.constant(1, shape=[1])
            init_lm_inputs = tf.cast(argm, tf.int32)  # [bs,]
            init_state = state  # [ba, dim]

            def condition(answer, lens, lm_inputs, state):
                # 遇到终止符 or 长度超过最大长度
                return tf.logical_and(lm_inputs != self.EOS_tag,
                                      tf.squeeze(lens) <= max_review_length)

            def forward_one_step(answer, lens, lm_inputs, state):
                # [bs, dim]
                self.tip_inputs_embedded = tf.nn.embedding_lookup(params=self.embeddings, ids=lm_inputs)
                # [bs, dim], [bs, dim]
                self.rnnlm_outputs, state = self.rnn_cell(self.tip_inputs_embedded, state)

                final_input = self.rnnlm_outputs
                logits = tf.layers.dense(final_input, self.vocab_size,
                                         kernel_initializer=self.initializer,
                                         bias_initializer=self.initializer,
                                         name='review_output_layer', reuse=True)

                self.preds = tf.nn.softmax(logits)
                argm = tf.argmax(self.preds, axis=1, output_type=tf.int32)
                lm_inputs = tf.cast(argm, tf.int32)
                answer = tf.concat([answer, lm_inputs], axis=0)
                lens = lens + 1
                return answer, lens, lm_inputs, state

            answer, lens, lm_inputs, state = tf.while_loop(
                condition,
                forward_one_step,
                [init_answer, init_lens, init_lm_inputs, init_state])
            return answer



    # 返回经过 co-attentionl 处理的 user/item 的向量表达
    def multi_pointer_coattention_networks(self,
                                           q1_output, q2_output,
                                           q1_len, q2_len,
                                           o1_embed, o2_embed,
                                           o1_len, o2_len,
                                           rnn_type='',
                                           reuse=None):

        """ Multi-Pointer Co-Attention Networks
        q1_output:user
        q2_output:item
        all their meta info (lengths etc.)

        o1_embed and o2_embed are original word embeddings, which have
        not been procesed by review-level encoders.

        Returns q1_output, q2_output, which are the final user/item reprs.
        """

        # q1_output,q2_output都是(10, 20, 50)
        # 在train中，q1_output中每20个条目都是不同的， q2_output是不同的
        # 在valid中，q1_output中每20个条目是相同的， q2_output是不同的
        _odim = o1_embed.get_shape().as_list()[2]

        # for visualisation purposes only
        self.afm = []
        self.afm2 = []
        self.word_att1 = []
        self.word_att2 = []
        self.att1 = []
        self.att2 = []
        self.hatt1 = []
        self.hatt2 = []
        self.word_u = []
        self.word_i = []

        print("Multi-Pointer Co-Attention Network Model")
        o1_embed = tf.reshape(o1_embed, [-1, self.args.dmax, _odim])
        o2_embed = tf.reshape(o2_embed, [-1, self.args.dmax, _odim])
        f1, f2 = [], []
        r1, r2 = [], []

        if self.args.masking == 1:
            q1_mask = tf.sequence_mask(q1_len, self.args.dmax, dtype=tf.float32)
            q2_mask = tf.sequence_mask(q2_len, self.args.dmax, dtype=tf.float32)
        else:
            q1_mask = None
            q2_mask = None

        tmp_reuse = reuse

        for i in range(self.args.num_heads):
            sub_afm = []
            print("Review-level Co-Attention...")
            '''Review-level Co-Attention
            '''

            if self.args.att_reuse == 0:
                name = 'caml_{}'.format(i)
                # reuse = None
            else:
                name = 'caml'
                if i == 1:
                    #    reuse = None
                    # else:
                    tmp_reuse = True

            # q1_output : [bs, 20, 50]
            # _q1是评论层面的向量表示，使用最大池化；
            # a1, a2返回的应该是选择的最重要的评论的权重 [bs, 20]，每行20个中都只有一个是1
            _q1, _q2, a1, a2, sa1, sa2, afm = co_attention(
                q1_output, q2_output, att_type=self.args.att_type,
                pooling='MAX', mask_diag=False,
                kernel_initializer=self.initializer,
                activation=None, dropout=self.dropout,
                seq_lens=None, transform_layers=self.args.num_inter_proj,
                proj_activation=tf.nn.relu, name=name,
                reuse=tmp_reuse, gumbel=True,
                hard=1, model_type=self.args.rnn_type,
                mask_a=q1_mask, mask_b=q2_mask
            )

            # f1保存的是评论层面+单词层面表示的拼接
            if self.args.average_embed == 1:
                _q1 = tf.reduce_sum(_q1, 1)
                _q2 = tf.reduce_sum(_q2, 1)
                f1.append(_q1)
                f2.append(_q2)

            self.att1.append(sa1)
            self.att2.append(sa2)
            self.hatt1.append(a1)
            self.hatt2.append(a2)
            self.afm.append(afm)

            print("Word-level Co-Attention...")
            """ Word-level Co-Attention Layer
            """
            # _dim = o1_embed.get_shape().as_list()[1]
            o1_embed = tf.reshape(o1_embed, [-1, self.args.dmax,
                                             self.args.smax * self.args.emb_size])
            o2_embed = tf.reshape(o2_embed, [-1, self.args.dmax,
                                             self.args.smax * self.args.emb_size])

            # a1, a2: #[bs, 20]，每20个中只有一个是1，其他事0
            # _a1, _a2: [bs, 20, 1]
            self.test1 = a1
            self.test2 = a2
            _a1 = tf.expand_dims(a1, 2)
            _a2 = tf.expand_dims(a2, 2)

            # [bs, 20, 30]
            # self.test1 = tf.reshape(self.c1_inputs, [-1, self.args.dmax, self.args.smax])
            # self.test2 = tf.reshape(self.c2_inputs, [-1, self.args.dmax, self.args.smax])

            # review_concept1: [bs, 30]
            review_concept1 = tf.reduce_sum(
                tf.reshape(self.c1_inputs, [-1, self.args.dmax, self.args.smax]) * tf.cast(_a1, dtype=tf.int32), 1)

            review_concept2 = tf.reduce_sum(
                tf.reshape(self.c2_inputs, [-1, self.args.dmax, self.args.smax]) * tf.cast(_a2, dtype=tf.int32), 1)

            _o1 = tf.nn.embedding_lookup(self.embeddings,
                                         review_concept1)
            _o2 = tf.nn.embedding_lookup(self.embeddings,
                                         review_concept2)

            # bsz x dmax
            # olen should be bsz x dmax
            _o1_len = tf.reshape(self.c1_len, [-1, self.args.dmax])
            _o2_len = tf.reshape(self.c2_len, [-1, self.args.dmax])
            _o1_len = tf.reduce_sum(_o1_len * tf.cast(a1, tf.int32), 1)
            _o2_len = tf.reduce_sum(_o2_len * tf.cast(a2, tf.int32), 1)
            _o1_len = tf.reshape(_o1_len, [-1])
            _o2_len = tf.reshape(_o2_len, [-1])

            if self.args.masking == 1:
                o1_mask = tf.sequence_mask(_o1_len, self.args.smax, dtype=tf.float32)
                o2_mask = tf.sequence_mask(_o2_len, self.args.smax, dtype=tf.float32)
            else:
                o1_mask = None
                o2_mask = None

            if self.args.att_reuse == 0:
                name = 'inner_{}'.format(i)
            else:
                name = 'inner'

            if self.args.word_gumbel == 1:
                word_hard = True
            else:
                word_hard = False

            # 计算co-attention
            # 获得用户或物品单词层面的向量表示，采用平均池化
            z1, z2, wa1, wa2, swa1, swa2, wm = co_attention(
                _o1, _o2, att_type=self.args.att_type,
                pooling=self.args.word_pooling, mask_diag=False,
                kernel_initializer=self.initializer, activation=None,
                dropout=self.dropout, seq_lens=None,
                transform_layers=self.args.num_inter_proj,
                proj_activation=tf.nn.relu, name=name,
                reuse=tmp_reuse, model_type=self.args.rnn_type,
                hard=1, gumbel=word_hard,
                mask_a=o1_mask, mask_b=o2_mask
            )

            sub_afm.append(wm)
            z1 = tf.reduce_sum(z1, 1)
            z2 = tf.reduce_sum(z2, 1)

            if self.args.concept == 1:
                f1.append(z1)
                f2.append(z2)
            # These below are for visualisation only.
            self.afm2.append(wm)
            # 分别对应两个level的attention
            self.word_att1.append(swa1)
            self.word_att2.append(swa2)
            if self.args.word_gumbel == 1:
                word1 = tf.expand_dims(tf.reduce_sum(review_concept1 * tf.cast(wa1, dtype=tf.int32), 1), 1)
                word2 = tf.expand_dims(tf.reduce_sum(review_concept2 * tf.cast(wa2, dtype=tf.int32), 1), 1)
                self.word_u.append(word1)
                self.word_i.append(word2)

        if self.args.word_gumbel == 1:
            self.word_u = tf.concat(self.word_u, 1)
            self.word_i = tf.concat(self.word_i, 1)

        if ('FN' in rnn_type):
            # Neural Network Multi-Pointer Learning

            q1_output = tf.concat(f1, 1)
            q2_output = tf.concat(f2, 1)
            q1_output = ffn(q1_output, _odim,
                            self.initializer, name='final_proj',
                            reuse=reuse,
                            num_layers=self.args.num_com,
                            dropout=None, activation=tf.nn.relu)
            q2_output = ffn(q2_output, _odim,
                            self.initializer, name='final_proj',
                            reuse=True,
                            num_layers=self.args.num_com,
                            dropout=None, activation=tf.nn.relu)
        elif ('ADD' in rnn_type):
            # Additive Multi-Pointer Aggregation
            q1_output = tf.add_n(f1)
            q2_output = tf.add_n(f2)
        else:
            # Concat Multi-Pointer Aggregation
            q1_output = tf.concat(f1, 1)
            q2_output = tf.concat(f2, 1)

        print("================================================")
        return q1_output, q2_output

