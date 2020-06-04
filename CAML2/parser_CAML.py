from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

def build_parser():
    parser = argparse.ArgumentParser()
    ps = parser.add_argument
    ps("--dataset", dest="dataset", type=str, default='NET')
    ps("--rnn_type", dest="rnn_type", type=str, metavar='<str>', default='RAW_MSE_CAML_FN_FM')
    ps("--emb_size", dest="emb_size", type=int, metavar='<int>', default=128)
    ps("--batch_size", dest="batch_size", type=int, metavar='<int>', default=64)
    ps("--lr", dest='learn_rate', type=float, metavar='<float>', default=0.0001)
    ps("--epochs", dest="epochs", type=int, metavar='<int>', default=1)
    # Question
    ps("--qmax", dest="qmax", type=int, metavar='<int>', default=100)  
    # Characters
    ps("--char_max", dest="char_max", type=int, metavar='<int>', default=100)
    # Answer
    ps("--amax", dest="amax", type=int, metavar='<int>', default=100)
    # Review Sentences
    ps("--smax", dest="smax", type=int, metavar='<int>', default=10)
    # Generated Reviews
    ps("--gmax", dest="gmax", type=int, metavar='<int>', default=30)
    # documents number
    ps("--dmax", dest="dmax", type=int, metavar='<int>', default=4)  
    
    

    ################################## 不用改的参数 ##################################
    ps("--key_word_lambda", dest="key_word_lambda", type=float, metavar='<float>',default=1.0)
    ps("--decay_epoch", dest="decay_epoch", type=int, metavar='<int>', default=0)
    ps("--allow_growth", dest="allow_growth", type=int, metavar='<int>', default=0)
    ps("--dropout", dest="dropout", type=float, metavar='<float>', default=0.8)
    ps("--rnn_dropout", dest="rnn_dropout", type=float, metavar='<float>', default=0.8)
    ps("--emb_dropout", dest="emb_dropout", type=float, metavar='<float>', default=0.8)
    
    ps("--num_proj", dest="num_proj", type=int, metavar='<int>',
       default=1, help="Number of projection layers")
    ps("--factor", dest="factor", type=int, metavar='<int>',
       default=10, help="Number of factors (for FM model)")
    ps("--gen_lambda", dest="gen_lambda", type=float, metavar='<float>',
        default=1.0, help="The generationg loss weight.")
    ps("--pretrained", dest="pretrained", type=int, metavar='<int>',
       default=0, help="Whether to use pretrained embeddings or not")
    ps('--gpu', dest='gpu', type=int, metavar='<int>',
       default=0, help="Specify which GPU to use (default=0)")
    ps("--clip_norm", dest='clip_norm', type=int,
       metavar='<int>', default=1, help="Clip Norm value for gradients")
    ps("--trainable", dest='trainable', type=int, metavar='<int>',
       default=1, help="Trainable Word Embeddings (0|1)")
    ps('--l2_reg', dest='l2_reg', type=float, metavar='<float>',
       default=1E-6, help='L2 regularization, default=4E-6')
    ps('--eval', dest='eval', type=int, metavar='<int>',
       default=1, help='Epoch to evaluate results (default=1)')
    ps('--log', dest='log', type=int, metavar='<int>',
       default=1, help='1 to output to file and 0 otherwise')
    ps('--dev', dest='dev', type=int, metavar='<int>',
       default=1, help='1 for development set 0 to train-all')
    ps('--seed', dest='seed', type=int, default=1337, help='random seed (not used)')
    ps('--num_heads', dest='num_heads', type=int, default=1, help='number of heads')
    ps("--hard", dest="hard", type=int, metavar='<int>',
       default=1, help="Use hard att when using gumbel")
    ps('--word_aggregate', dest='word_aggregate', type=str, default='MAX',
        help='pooling type for key word loss')
    ps("--average_embed", dest="average_embed", type=int, metavar='<int>',
       default=1, help="Use average embedding of all reviews")
    ps("--word_gumbel", dest="word_gumbel", type=int, metavar='<int>',
       default=0, help="Use gumbel in the word(concept) level")
    ps("--data_prepare", dest="data_prepare", type=int, metavar='<int>',
       default=0, help="Data preparing type")
    ps("--feed_rating", dest="feed_rating", type=int, metavar='<int>',
       default=0, help="0 no feed, 1 feed groundtruth, 2 feed predicted rate")
    
    ps("--masking", dest="masking", type=int, metavar='<int>',
       default=1, help="Use masking and padding")
    ps("--concept", dest="concept", type=int, metavar='<int>',
       default=1, help="Use concept correlated components or not")
    ps("--len_penalty", dest="len_penalty", type=int, metavar='<int>',
       default=2, help="Regularization type for length balancing in beam search")
    ps("--implicit", dest="implicit", type=int, metavar='<int>',
       default=0, help="Use implicit factor or not")
    
    ps("--att_reuse", dest="att_reuse", type=int, metavar='<int>',
       default=0, help="Re-use attention or not")
    ps('--tensorboard', action='store_true', help='To use tensorboard or not (may not work)')
    ps('--early_stop',  dest='early_stop', type=int,
       metavar='<int>', default=5, help='early stopping')
    ps('--wiggle_lr',  dest='wiggle_lr', type=float,
       metavar='<float>', default=1E-5, help='Wiggle lr')
    ps('--wiggle_after',  dest='wiggle_after', type=int,
       metavar='<int>', default=0, help='Wiggle lr')
    ps('--wiggle_score',  dest='wiggle_score', type=float,
       metavar='<float>', default=0.0, help='Wiggle score')
    ps('--translate_proj', dest='translate_proj', type=int,
       metavar='<int>', default=1, help='To translate project or not')
    ps('--att_type', dest='att_type', type=str, default='SOFT')
    ps('--att_pool', dest='att_pool', type=str, default='MAX')
    ps('--word_pooling', dest='word_pooling', type=str, default='MEAN')
    ps('--num_class', dest='num_class', type=int,
       default=2, help='self explainatory..(not used for recommendation)')
    ps('--all_dropout', action='store_true',
       default=False, help='to dropout the embedding layer or not')    
    ps('--base_encoder', dest='base_encoder',
       default='GLOVE', help='BaseEncoder for hierarchical models')
    ps("--init", dest="init", type=float,
       metavar='<float>', default=0.01, help="Init Params")
    ps("--temperature", dest="temperature", type=float,
      metavar='<float>', default=0.5, help="Temperature")
    ps("--num_inter_proj", dest="num_inter_proj", type=int,
       metavar='<int>', default=1, help="Number of inter projection layers")
    ps("--num_com", dest="num_com", type=int,
       metavar='<int>', default=1, help="Number of compare layers")
    ps("--init_type", dest="init_type", type=str,
       metavar='<str>', default='normal', help="Init Type")
    ps("--decay_lr", dest="decay_lr", type=float,
       metavar='<float>', default=0, help="Decay Learning Rate")
    ps("--decay_steps", dest="decay_steps", type=float,
       metavar='<float>', default=0, help="Decay Steps (manual)")
    ps("--decay_stairs", dest="decay_stairs", type=float,
       metavar='<float>', default=1, help="To use staircase or not")
    ps('--emb_type', dest='emb_type', type=str,
       default='glove', help='embedding type')
    return parser
