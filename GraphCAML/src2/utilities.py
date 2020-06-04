import sys
import os
import operator
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

import warnings
warnings.filterwarnings('ignore')

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w") #如果是追加，用a+

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

#返回：[user[i], items[i], ratings[i], reviews[i], len(reviews[i])，review是原始的review的长度
def batchify(data, reviews, i, bsz, max_sample):
    # data: [(user, item, rating)]
    start = int(i * bsz)
    end = int(i * bsz) + bsz
    if (end > max_sample):
        end = max_sample
    data = data[start:end]
    reviews = reviews[start:end]

    users = [x[0] for x in data]
    items = [x[1] for x in data]
    ratings = [x[2] for x in data]

    output = []
    for i in range(len(users)):
        output.append([users[i], items[i], ratings[i], reviews[i], len(reviews[i])])

    # u_g_embeddings, i_g_embeddings = tf.split(adj, [num_user, num_item])

    # u_g_embeddings = u_g_embeddings[users]
    # i_g_embeddings = i_g_embeddings[items]

    # print('\n\n\n', i, len(output), '\n\n\n')
    return output

'''
def compute_bleu(references, candidates):
    references = [[item] for item in references]
    #1元组，2元组，3元组和4元组
    score0 = corpus_bleu(references, candidates)
    score1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    score2 = corpus_bleu(references, candidates,  weights=(0, 1, 0, 0))
    score3 = corpus_bleu(references, candidates,  weights=(0, 0, 1, 0))
    score4 = corpus_bleu(references, candidates, weights=(0, 0, 0, 1))
    return score0, score1, score2, score3, score4
'''

#学姐的
def compute_bleu(references, candidates):
    references = [[item] for item in references]
    #1元组，2元组，3元组和4元组
    score0 = corpus_bleu(references, candidates)
    score1 = corpus_bleu(references, candidates, weights=(1.0,))
    score2 = corpus_bleu(references, candidates,  weights=(1.0 / 2,) * 2)
    score3 = corpus_bleu(references, candidates,  weights=(1.0 / 3,) * 3)
    score4 = corpus_bleu(references, candidates, weights=(1.0 / 4,) * 4)
    return score0, score1, score2, score3, score4

'''
def compute_bleu(references, candidates):
    references = [[item] for item in references]
    #1元组，2元组，3元组和4元组
    score0 = corpus_bleu(references, candidates)
    score1 = sentence_bleu(references, candidates, weights=(1, 0, 0, 0))
    score2 = sentence_bleu(references, candidates, weights=(0, 1, 0, 0))
    score3 = sentence_bleu(references, candidates, weights=(0, 0, 1, 0))
    score4 = sentence_bleu(references, candidates, weights=(0, 0, 0, 1))
    return score0, score1, score2, score3, score4
'''



def mkdir_p(path):
    ''' Makes path if path does not exist
    '''
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        pass

def register_eval_score(epoch, eval_type, metric, val, eval_dev, eval_test):
    """ Registers eval metrics to class
    """
    eval_obj = {
        'metric':metric,
        'val':val
    }

    if(eval_type.lower()=='dev'):
        eval_dev[epoch].append(eval_obj)
    elif(eval_type.lower()=='test'):
        eval_test[epoch].append(eval_obj)

def show_metrics(epoch, eval_list, show_metrics, name):
    """ Shows and outputs metrics
    """
    # print("Eval Metrics for [{}]".format(name))
    get_last = eval_list[epoch]
    for metric in get_last:
        if(metric['metric'] in show_metrics):
            print("[{}] {}={}".format(name, metric['metric'], metric['val']))

def _select_test_by_dev(self, epoch, eval_dev, eval_test,
                        no_test=False, lower_is_better=False,
                        name='', has_dev=True):
    #Outputs best test score based on dev score

    primary_metrics = []
    test_metrics = []
    if(lower_is_better):
        reverse=False
    else:
        reverse=True

    for key, value in eval_dev.items():
        _val = [x for x in value if x['metric']==self.eval_primary] #all_loss作为标准
        if(len(_val)==0):
            continue
        primary_metrics.append([key, _val[0]['val']])

    sorted_metrics = sorted(primary_metrics,
                                key=operator.itemgetter(1),
                                reverse=reverse)
    cur_dev_score = primary_metrics[-1][1]
    best_epoch = sorted_metrics[0][0] #选出dev中最好的epoch

    if(no_test):
        print("[{}]\nBest epoch = {}".format(name, best_epoch))
        self._show_metrics(best_epoch, eval_dev,
                            self.show_metrics, name='best')
        print("\n")
        return best_epoch

    print("[{}]\n epoch = {}".format(name, best_epoch))
    self._show_metrics(best_epoch, eval_test,
                        self.show_metrics, name='best')
    print("\n")

    return best_epoch

def prep_hierarchical_data_list_new(data, lens, smax, dmax, threshold=True):
    """ Converts and pads hierarchical data
    """
    # print("Preparing Hiearchical Data list")
    all_data = []
    all_lengths = []
    for i in range(len(data)):
        new_data = []
        data_lengths = []
        #for data_list in doc:
        doc = data[i]
        doc_len = lens[i]

        for j in range(len(doc)):
            data_list = doc[j]
            sent_lens = doc_len[j]
            if(sent_lens==0):
                continue
            if(threshold and sent_lens>smax):
                sent_lens = smax

            _data_list = pad_to_max(data_list, smax)
            new_data.append(_data_list)
            data_lengths.append(sent_lens)
        new_data = pad_to_max(new_data, dmax, pad_token = [0 for i in range(smax)])

        _new_data = []
        for nd in new_data:
            # flatten lists
            _new_data += nd

        data_lengths = pad_to_max(data_lengths, dmax, pad_token=0)
        all_data.append(_new_data)
        all_lengths.append(data_lengths)
    return all_data, all_lengths

def pad_to_max(seq, seq_max, pad_token=0):
    ''' Pad Sequence to sequence max
    '''
    while(len(seq)<seq_max):
        seq.append(pad_token)
    return seq[:seq_max]

def register_index_map(imap, idx, target):
    imap[target] = idx

def clip_labels(x):
    if (x > 5):
        return 5
    elif (x < 1):
        return 1
    else:
        return x

#max_val = 5; min_val = 1
def rescale(x):
    return (x * (5 - 1)) + 1

'''
def dict_to_list(data_dict):
    data_list = []
    for key, value in tqdm(data_dict.items(),
                           desc='dict conversion'):
        for v in value:
            data_list.append([key, v[0], v[1]])
    return data_list


def dictToFile(dict, path):
    print("Writing to {}".format(path))
    with gzip.open(path, 'w') as f:
        f.write(json.dumps(dict))


def dictFromFileUnicode(path):
    #Read js file:
    #key ->  unicode keys
    #string values -> unicode value
    print("Loading {}".format(path))
    with gzip.open(path, 'r') as f:
        return json.loads(f.read())


def load_pickle(fin):
    with open(fin, 'r') as f:
        obj = pickle.load(f)
    return obj


def select_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if (gpu >= 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

def load_set(fin):
    data = []
    with open(fin, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            data.append(r)
    return data


def length_stats(lengths, name=''):
    print("=====================================")
    print("Length Statistics for {}".format(name))
    print("Max={}".format(np.max(lengths)))
    print("Median={}".format(np.median(lengths)))
    print("Mean={}".format(np.mean(lengths)))
    print("Min={}".format(np.min(lengths)))


def show_stats(name, x):
    print("{} max={} mean={} min={}".format(name,
                                            np.max(x),
                                            np.mean(x),
                                            np.min(x)))


def print_args(args, path=None):
    if path:
        output_file = open(path, 'w')
    args.command = ' '.join(sys.argv)
    items = vars(args)
    if path:
        output_file.write('=============================================== \n')
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    if path:
        output_file.write('=============================================== \n')
    if path:
        output_file.close()
    del args.command


def mkdir_p(path):
    if path == '':
        return
    try:
        os.makedirs(path)
    except:
        pass


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor
  (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def model_stats():
    print("============================================================")
    print("List of all Trainable Variables")
    tvars = tf.trainable_variables()
    all_params = []

    for idx, v in enumerate(tvars):
        print(" var {:3}: {:15} {}".format(idx, str(v.get_shape()), v.name))
        num_params = 1
        param_list = v.get_shape().as_list()
        if (len(param_list) > 1):
            for p in param_list:
                if (p > 0):
                    num_params = num_params * int(p)
        else:
            all_params.append(param_list[0])
        all_params.append(num_params)
    print("Total number of trainable parameters {}".format(np.sum(all_params)))

'''
