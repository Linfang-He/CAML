from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def build_parser():
    """ This function produces the default argparser.
    Note that NOT all args are actively used. I used a generic
    argparser for all my projects, so you will surely see something that
    is not used. The important arguments are provided in the main readme.
    """
    parser = argparse.ArgumentParser()
    ps = parser.add_argument

# ps('--eval', dest='eval', type=int, metavar='<int>',
#        default=1, help='Epoch to evaluate results (default=1)')
    # ps('--dataset', dest='dataset', type=str,
    #    default='books', help='dataset')


    ps("--epochs", dest="epochs", type=int, metavar='<int>',
       default=30, help="Number of epochs (default=30)")

    # ps('--mode', dest='mode', type=str, default='HREC',
    #    help='mode')
    ps("--batch_size", dest="batch_size", type=int, metavar='<int>',
       default=128, help="Batch size (default=128)")

    # ps("--n_iter", dest="n_iter", type=int, default=3, help='number of iterations when computing entity representation')

    # ps("--method", dest="entity_dim", type=str, metavar='<str>',
    #    default='all', help="Four options, 'all', 'gatne', 'ngcf', 'caml' (default='all')")

    ps("--learn_rate", dest='learn_rate', type=float,
       metavar='<float>', default=0.001, help="Learning Rate")

    ps("--emb_size", dest="emb_size", type=int, metavar='<int>',
       default=50, help="Embeddings dimension (default=50)")

    ps("--rnn_size", dest="rnn_size", type=int, metavar='<int>',
       default=50, help="model-specific dimension. (default=50)")
    ps("--rnn_dim", dest="rnn_dim", type=int, metavar='<int>',
       default=50, help="model-specific dimension. (default=50)")
    ps("--latent_size", dest="latent_size", type=int, metavar='<int>',
       default=50, help="latent factor dimension for user/items. (default=50)")

    # MODIFIED
    # ps("--l2_weight", dest="l2_weight", type=float, default=0.01, help='weight of l2 regularization')

    # ps("--qmax", dest="qmax", type=int, metavar='<int>',
    #  default=100, help="Max Length of Question ")

    # ps("--char_max", dest="char_max", type=int, metavar='<int>',
    #   default=8, help="Max length of characters")
    # ps("--amax", dest="amax", type=int, metavar='<int>',
    #  default=100, help="Max Length for Answer")

    ps("--smax", dest="smax", type=int, metavar='<int>',
       default=30, help="Max Length of Sentences (per review)")

    ps("--gmax", dest="gmax", type=int, metavar='<int>',
       default=30, help="Max Length of Generated Reviews (per review)")

    ps("--dmax", dest="dmax", type=int, metavar='<int>',
       default=20, help="Max Number of reviews")

    ps("--max_clicked_entity", dest="max_clicked_entity", type=int, metavar='<int>',
       default=20, help="Max length of clicked entities")

    # ps("--dev_lr", dest='dev_lr', type=int,
    #   metavar='<int>', default=0, help="Dev Learning Rate")
    # ps("--ratio", dest="ratio", type=float, default=1, help='size of training dataset')

    # ps("--opt", dest="opt", type=str, metavar='<str>', default='Adam',
    #  help="Optimization algorithm)")

    ps("--key_word_lambda", dest="key_word_lambda", type=float, metavar='<float>',
       default=1.0, help="The key word generation loss weight.")
    ps("--use_lower", dest="use_lower", type=int, metavar='<int>',
       default=1, help="Use all lowercase")

    ps("--allow_growth", dest="allow_growth", type=int, metavar='<int>',
       default=0, help="Allow Growth")

    ps("--rnn_layers", dest="rnn_layers", type=int,
       metavar='<int>', default=1, help="Number of RNN layers")
    # ps("--decay_epoch", dest="decay_epoch", type=int,
    #   metavar='<int>', default=1, help="Decay everywhere n epochs")
    ps("--num_proj", dest="num_proj", type=int, metavar='<int>',
       default=1, help="Number of projection layers")
    ps("--factor", dest="factor", type=int, metavar='<int>',
       default=10, help="Number of factors (for FM model)")

    ps("--dropout", dest="dropout", type=float, metavar='<float>',
       default=0.8, help="The dropout probability.")
    ps("--gen_lambda", dest="gen_lambda", type=float, metavar='<float>',
       default=1.0, help="The generationg loss weight.")
    ps("--rnn_dropout", dest="rnn_dropout", type=float, metavar='<float>',
       default=0.8, help="The dropout probability.")
    ps("--emb_dropout", dest="emb_dropout", type=float, metavar='<float>',
       default=0.8, help="The dropout probability.")

    # ps("--pretrained", dest="pretrained", type=int, metavar='<int>',
    #  default=0, help="Whether to use pretrained embeddings or not")

    ps("--hdim", dest='hdim', type=int, metavar='<int>',
       default=50, help="Hidden layer size (default=50)")

    ps("--clip_norm", dest='clip_norm', type=int,
       metavar='<int>', default=1, help="Clip Norm value for gradients")

    ps("--trainable", dest='trainable', type=int, metavar='<int>',
       default=1, help="Trainable Word Embeddings (0|1)")
    ps('--l2_reg', dest='l2_reg', type=float, metavar='<float>',
       default=1E-6, help='L2 regularization, default=4E-6')


    ps('--num_heads', dest='num_heads', type=int, default=3, help='number of heads') #多指针的数目
    ps("--hard", dest="hard", type=int, metavar='<int>',
       default=1, help="Use hard att when using gumbel")

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

    ps("--att_reuse", dest="att_reuse", type=int, metavar='<int>',
       default=0, help="Re-use attention or not")

    ps('--tensorboard', action='store_true', help='To use tensorboard or not (may not work)')

    ps('--early_stop', dest='early_stop', type=int,
       metavar='<int>', default=5, help='early stopping')

    ps('--wiggle_lr', dest='wiggle_lr', type=float,
       metavar='<float>', default=1E-5, help='Wiggle lr')
    ps('--wiggle_after', dest='wiggle_after', type=int,
       metavar='<int>', default=0, help='Wiggle lr')
    ps('--wiggle_score', dest='wiggle_score', type=float,
       metavar='<float>', default=0.0, help='Wiggle score')
    ps('--translate_proj', dest='translate_proj', type=int,
       metavar='<int>', default=1, help='To translate project or not')

    ps('--att_type', dest='att_type', type=str, default='SOFT',
       help='attention type')
    ps('--att_pool', dest='att_pool', type=str, default='MAX',
       help='pooling type for attention')
    ps('--word_pooling', dest='word_pooling', type=str, default='MEAN',
       help='pooling type for word attention')
    ps('--num_class', dest='num_class', type=int,
       default=2, help='self explainatory..(not used for recommendation)')

    ps("--all_dropout", dest="all_dropout", type=int, metavar='<int>',
       default=0, help="to dropout the embedding layer or not")

    # MODIFIED 本来是6
    ps("--num_neg", dest="num_neg", type=int, metavar='<int>',
       default=10, help="Number of negative samples for pairwise training")

    ps("--init", dest="init", type=float,
       metavar='<float>', default=0.01, help="Init Params")
    ps("--temperature", dest="temperature", type=float,
       metavar='<float>', default=0.5, help="Temperature")
    ps("--num_inter_proj", dest="num_inter_proj", type=int,
       metavar='<int>', default=1, help="Number of inter projection layers")
    ps("--num_com", dest="num_com", type=int,
       metavar='<int>', default=1, help="Number of compare layers")
    ps("--show_att", dest="show_att", type=int,
       metavar='<int>', default=0, help="Display Attention")
    ps("--write_qual", dest="write_qual", type=int,
       metavar='<int>', default=0, help="write qual")
    ps("--show_affinity", dest="show_affinity", type=int,
       metavar='<int>', default=0, help="Display Affinity Matrix")
    ps("--init_type", dest="init_type", type=str,
       metavar='<str>', default='xavier', help="Init Type")
    ps("--decay_lr", dest="decay_lr", type=float,
       metavar='<float>', default=0, help="Decay Learning Rate")
    ps("--decay_steps", dest="decay_steps", type=float,
       metavar='<float>', default=0, help="Decay Steps (manual)")
    ps("--decay_stairs", dest="decay_stairs", type=float,
       metavar='<float>', default=1, help="To use staircase or not")
    ps('--emb_type', dest='emb_type', type=str,
       default='glove', help='embedding type')
    ps('--log_dir', dest='log_dir', type=str,
       default='logs', help='log directory')

    ps("--beam_size", dest="beam_size", type=int, metavar='<int>',
       default=4, help="beam search size")
    #    ps("--beam_number", dest="beam_number", type=int, metavar='<int>',
    #       default=4, help="beam search number")
    ps('--view_output', dest='view_output', type=str,
       default='logs', help='view output')

    # 我添加的参数
    ps('--rating_lambda', dest='rating_lambda', type=float,
       default='1', help='rating_lambda')

    return parser
