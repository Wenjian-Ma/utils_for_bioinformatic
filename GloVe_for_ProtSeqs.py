from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from scipy import sparse
from random import shuffle
from math import log





def merge_main_context(vector_matrix, merge_fun=lambda m, c: np.mean([m, c], axis=0),
                       normalize=True):
    """
    Merge the main-word and context-word vectors for a weight matrix
    using the provided merge function (which accepts a main-word and
    context-word vector and returns a merged version).

    By default, `merge_fun` returns the mean of the two vectors.
    """

    vocab_size = int(len(vector_matrix) / 2)
    for i, row in enumerate(vector_matrix[:vocab_size]):
        merged = merge_fun(row, vector_matrix[i + vocab_size])  # 按对应行进行求和
        if normalize:
            merged /= np.linalg.norm(merged)
        vector_matrix[i, :] = merged

    return vector_matrix[:vocab_size]


def run_iter(data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training.

    `data` is a pre-fetched data / weights list where each element is of
    the form

        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)

    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:
        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - log(cooccurrence))

        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost  # 注意这里乘了1/2

        # Compute gradients for word vector terms.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word, biases, etc.
        grad_main = weight * cost_inner * v_context  # 损失函数对vi求导，是不是缺了个因子2呢？ 不缺！
        grad_context = weight * cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))  # 梯度下降法，
        # 问题是为什么要除以np.sqrt(gradsq_W_main) -> 原文使用Adagrad算法， 利用梯度的对学习率约束
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
            gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)  # 向量
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2  # 标量
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def train_glove(vocab, co_occurrences, iter_callback=None, vector_size=100,
                iterations=25, **kwargs):
    """
    co_occurrences: (word_i_id, word_j_id, x_ij)

    If `iter_callback` is not `None`, the provided function will be
    called after each iteration with the learned `W` matrix so far.

    Keyword arguments are passed on to the iteration step function
    `run_iter`.

    Returns the computed word vector matrix .
    """

    vocab_size = len(vocab)

    # Word vector matrix. This matrix is (2V) * d, where V is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5].
    vector_matrix = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    #
    #  this matrix is same size with vector_matrix
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # NB: These are all views into the actual data matrices, so updates
    # to them will pass on to the real data structures
    data = [(vector_matrix[i_main], vector_matrix[i_context + vocab_size],
             biases[i_main: i_main + 1],
             biases[i_context + vocab_size: i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main: i_main + 1],
             gradient_squared_biases[i_context + vocab_size: i_context + vocab_size + 1],
             co_occurrence)
            for i_main, i_context, co_occurrence in co_occurrences]

    for i in range(iterations):

        cost = run_iter(data, **kwargs)
        print('global cost of glove model: %.4f' % cost)
        if iter_callback is not None:
            iter_callback(vector_matrix)

    return vector_matrix


def build_co_occur(vocab, corpus, window_size=10, min_count=None):
    """
    Build a word co-occurrence list for the given corpus.
    return: (i_main, i_context, co_occurrence value)
    i_main -> the main word in the co_occurrence
    i_context -> is the ID of the context word
    co_occurrence` is the `X_{ij}` co_occurrence value as described in Pennington et al.(2014).
    If `min_count` is not `None`, co_occurrence pairs fewer than `min_count` times are ignored.
    """

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    co_occurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                       dtype=np.float64)

    for i, line in enumerate(corpus):

        token_ids = [vocab[word][0] for word in line]

        for center_i, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            # 将窗口左边的内容与窗口右边的内容区分开，
            context_ids = token_ids[max(0, center_i - window_size): center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)  # 原文用词对间的独立来除count

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                co_occurrences[center_id, left_id] += increment
                co_occurrences[left_id, center_id] += increment

    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)
    for i, (row, data) in enumerate(zip(co_occurrences.rows,
                                        co_occurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]


def build_vocab(train_corpus):
    """
    Build a vocabulary with word frequencies for an entire corpus.
    Returns: {word : (ID, frequency)}
    """

    vocab = Counter()
    for line in train_corpus:
        for tokens in line:
            vocab.update([tokens])

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}

def data_partition(sample_size_list):
    num_sum = 0
    seed_ = 42
    folds_num = 5
    label_all = []
    for i in range(len(sample_size_list)):
        tmp_labels = [float(i)] * sample_size_list[i]
        label_all += tmp_labels
        num_sum += sample_size_list[i]
    label_all = np.array(label_all)
    pse_data = np.random.normal(loc=0.0, scale=1.0, size=[num_sum, num_sum])
    folds = StratifiedKFold(folds_num, shuffle=True, random_state=np.random.RandomState(seed_))
    folds_temp = list(folds.split(pse_data, label_all))
    folds = []
    for i in range(folds_num):
        train_index = folds_temp[i][0]
        test_index = folds_temp[i][1]
        folds.append((train_index, test_index))
    return folds
def glove(sentence_list, sample_size_list, fixed_len, word_size, win_size, vec_dim=10):
    n_row = (fixed_len - word_size + 1) * vec_dim
    corpus_out = -np.ones((len(sentence_list), n_row))

    folds = data_partition(sample_size_list)
    print('Glove processing ...')

    for i, (train_index, test_index) in enumerate(folds):
        print('Round [%s]' % (i + 1))
        train_sentences = []
        test_sentences = []
        for x in train_index:
            train_sentences.append(sentence_list[x])
        for y in test_index:
            test_sentences.append(sentence_list[y])
        # The core stone of Glove
        vocab = build_vocab(train_sentences)  # 词汇表
        # print(vocab): {'CTT': (0, 8), 'TTC': (1, 6), 'TCG': (2, 2), 'CGC': (3, 2), ...}
        # exit()
        co_occur = build_co_occur(vocab, train_sentences, window_size=win_size)  # “共现矩阵”

        vector_matrix = train_glove(vocab, co_occur, vector_size=vec_dim, iterations=50)  # 词向量矩阵(main + context)

        # Merge and normalize word vectors
        vector_matrix = merge_main_context(vector_matrix)  # 对词向量矩阵(main + context)按对应行平均并归一化
        vectors = []
        for sentence in test_sentences:
            vector = []
            for j in range(len(sentence)):
                try:
                    vec_temp = np.array(vector_matrix[vocab[sentence[j]][0]])
                    # vocab={'word': (id, frequency), ...}
                except KeyError:
                    vec_temp = np.zeros(vec_dim)
                if len(vector) == 0:
                    vector = vec_temp
                else:
                    vector = np.hstack((vector, vec_temp))
            vectors.append(vector)
        corpus_out[test_index] = np.array(vectors)
    print('....................')
    return corpus_out




def glove4vec(corpus, sample_size_list, fixed_len):

    corpus_out = glove(corpus, sample_size_list, fixed_len, word_size=2,
                       win_size=5, vec_dim=10)
    return corpus_out


def read_fasta(file_path):
    fasta_dict = {}
    with open(file_path, 'r') as f:
        sequence = ''
        header = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # If sequence already exists, store it
                if header and sequence:
                    fasta_dict[header] = sequence
                # Update header and reset sequence for new entry
                header = line[1:]
                sequence = ''
            else:
                sequence += line
        # Store last sequence
        if header and sequence:
            fasta_dict[header] = sequence
    return fasta_dict

def km_words(input_file, fixed_len, word_size, fixed=True):
    """ convert sequence to corpus """
    seq_list = []
    for seq in input_file:
        if fixed:
            if len(seq)<fixed_len:
                seq = seq + 'X'*fixed_len

            seq_list.append(seq[:fixed_len])
        else:
            seq_list.append(seq)
    corpus = []
    for sequence in seq_list:
        word_list = []
        # windows slide along sequence to generate gene/protein words
        for i in range(len(sequence) - word_size + 1):
            word = sequence[i:i + word_size]
            word_list.append(word)
        corpus.append(word_list)
    return corpus

#######Parameters####该程序将序列裁剪至400后计算每个单词的嵌入（10dim），展平后为蛋白表征，也可改为不裁剪，求平均作为蛋白表征##
fixed_len = 400
fixed=True# 是否裁剪序列长度到400
word_size=2#2-mer

#############
species_15 = ['511145','284812','237561','224308','160488','99287','44689','10116','10090','9606','8355','7955','7227','4932','3702']
for specie in tqdm(species_15):

    seq_dict = read_fasta('/media/ST-18T/Ma/HIF2GO/data/CAFA/'+specie+'/'+specie+'.protein.sequences.v12.0.fasta')

    prot_id = []
    with open('/media/ST-18T/Ma/HIF2GO/data/CAFA/' + specie + '/' + specie + '_proteins_id.txt') as f_id:
        for line in f_id:
            prot_id.append(line.strip())

    seq = [seq_dict[i] for i in prot_id]#正常的存储序列的列表

    corpus = km_words(seq, fixed_len, word_size, fixed)#转化为k-mer序列

    emb_vectors = glove4vec(corpus, [len(corpus)], fixed_len)

    if emb_vectors.shape[0]!=len(seq):
        raise Exception('Error!')

    np.save('/media/ST-18T/Ma/HIF2GO/data/CAFA/'+specie+'/glove.npy',emb_vectors)
