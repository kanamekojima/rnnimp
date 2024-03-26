from argparse import ArgumentParser
import os
import random
import shutil
import sys
import time

import numpy as np

from common_utils import mkdir
from train_utils import (
    AugmentedDataFeeder,
    get_a1_freqs,
    get_random_hap_id_ordering,
    load_data,
    one_hot,
    set_weights_for_cross_entropy,
)
import rnn_model
import train_rnn_model


def kernel_PCA(K, num_components):
    n = len(K)
    C = np.full((n, n), - 1.0 / float(n))
    for i in range(n):
        C[i, i] += 1.0
    K_tilde = np.dot(C, np.dot(K, C))
    l, V = np.linalg.eig(K_tilde)
    l_indices = np.argsort(l)[::-1]
    l = l[l_indices]
    V = V[:, l_indices]
    for i in range(num_components):
        if l[i] <= 0:
            num_components = i
            break
    if num_components == 0:
        return None, None, 0
    return l[:num_components], V[:,:num_components], num_components


def get_normalized_kernel_PC_scores(L, K, V, l):
    K_mean = np.mean(K, axis=1)
    L_mean = np.mean(L, axis=1, keepdims=True)
    K_mean_all = np.mean(K_mean, keepdims=True)
    M = L - K_mean - L_mean + K_mean_all
    T = np.dot(M, V) * np.sqrt(len(K)) / l.reshape(1, -1)
    return T


def kernel_function(x, y):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    norm_xy = norm_x * norm_y
    if norm_xy == 0:
        return 0
    u = np.dot(x, y) / norm_xy
    sigma = 1.0
    return norm_xy * np.exp((u - 1.0) / sigma)


def get_features(haplotypes, num_components, a1_freqs):
    n = len(a1_freqs)
    if num_components == 0:
        empty_feature = [[], []] * n
        return empty_feature, 0
    X = np.zeros(haplotypes.T.shape, dtype=np.float64)
    for i in range(n):
        for j, haplotype in enumerate(haplotypes):
            X[i, j] = one_hot(haplotype[i], a1_freqs[i])[1]
    K = np.zeros([n, n], dtype=np.float64)
    L = np.zeros([2 * n, n], dtype=np.float64)
    for i in range(n):
        K[i, i] = kernel_function(X[i], X[i])
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = kernel_function(X[i], X[j])
    l, V, num_components = kernel_PCA(K, num_components)
    if num_components == 0:
        empty_feature = [[], []] * n
        return empty_feature, 0
    for i in range(n):
        for j in range(n):
            L[2 * i, j] = kernel_function(1 - X[i], X[j])
        L[2 * i + 1] = K[i]
    T = get_normalized_kernel_PC_scores(L, K, V, l)
    features = np.reshape(T, [-1, 2, num_components]).astype(np.float32)
    return features, num_components


def main():
    description = 'train'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str, required=True, dest='hap_file',
                        help='hap file')
    parser.add_argument('--legend', type=str, required=True,
                        dest='legend_file', help='legend file')
    parser.add_argument('--output-prefix', type=str, required=True,
                        dest='output_prefix', help='output prefix')
    parser.add_argument('--random-seed', type=int, default=3141592653,
                        dest='random_seed', help='seed for random')
    parser.add_argument('--rnn-cell-type', type=str, default='GRU',
                        dest='rnn_cell_type', help='RNN cell type')
    parser.add_argument('--num-units', type=int,  default=40, dest='num_units',
                        help='output dimension of rnn cells')
    parser.add_argument('--num-layers', type=int, default=5, dest='num_layers',
                        help='number of layers')
    parser.add_argument('--feature-size', type=int, default=10,
                        dest='feature_size', help='feature size')
    parser.add_argument('--scope', type=str, required=True,
                        dest='scope', help='model scope')
    parser.add_argument('--batch-size', type=int, default=500,
                        dest='batch_size', help='batch size')
    parser.add_argument('--max-iteration-count', type=int, default=100000,
                        dest='max_iteration_count',
                        help='max iteration count')
    parser.add_argument('--validation-sample-size', type=int, default=100,
                        dest='validation_sample_size',
                        help='validation sample size')
    parser.add_argument('--gamma', type=float, default=0, dest='gamma',
                        help='gamma value')
    parser.add_argument('--rsquare', action='store_true', default=False,
                        dest='use_rsquare',
                        help='use r-square values for validation')
    parser.add_argument('--num-threads', type=int, default=8,
                        dest='num_threads', help='num threads')
    args = parser.parse_args()

    start_time = time.time()
    mkdir(os.path.dirname(args.output_prefix))
    haplotypes, array_marker_flags, positions = load_data(
        args.hap_file, args.legend_file)
    input_indexes = array_marker_flags.nonzero()[0]
    output_indexes = np.logical_not(array_marker_flags).nonzero()[0]

    output_points_fw = []
    t = None
    for flag in array_marker_flags:
        if flag:
            if t is None:
                t = 0
            else:
                t += 1
        else:
            output_points_fw.append(t)
    output_points_bw = []
    t = None
    for flag in reversed(array_marker_flags):
        if flag:
            if t is None:
                t = len(input_indexes) - 1
            else:
                t -= 1
        else:
            output_points_bw.append(t)
    output_points_bw.reverse()

    config = rnn_model.Config(
        input_dim=2,
        num_inputs=len(input_indexes),
        num_classes=2,
        output_points_fw=output_points_fw,
        output_points_bw=output_points_bw,
        rnn_cell_type=args.rnn_cell_type,
        num_units=args.num_units,
        num_layers=args.num_layers,
        feature_size=args.feature_size,
        scope=args.scope,
        )
    hap_id_ordering = get_random_hap_id_ordering(
        len(haplotypes) // 2, args.random_seed)
    validation_hap_sample_size = 2 * args.validation_sample_size
    train_hap_ids = hap_id_ordering[validation_hap_sample_size:]
    haplotypes_train = haplotypes[train_hap_ids]
    a1_freqs = get_a1_freqs(haplotypes_train)
    features, feature_size = get_features(
        haplotypes_train[:, array_marker_flags], config.feature_size,
        a1_freqs[array_marker_flags])
    config.feature_size = features.shape[2]
    inputs = np.array([
        [one_hot(haplotype[i], a1_freqs[i]) for i in input_indexes]
        for haplotype in haplotypes
    ], dtype=np.float32)
    outputs = np.array([
        [one_hot(haplotype[i]) for i in output_indexes]
        for haplotype in haplotypes
    ], dtype=np.float32)
    train_data_feeder = AugmentedDataFeeder(
        inputs[train_hap_ids], outputs[train_hap_ids], positions,
        input_indexes, output_indexes)
    validation_hap_ids = hap_id_ordering[:validation_hap_sample_size]
    validation_inputs = inputs[validation_hap_ids]
    validation_outputs = outputs[validation_hap_ids]
    config_file = args.output_prefix + '_config.json'
    config.write(config_file)
    shutil.copy(args.legend_file, args.output_prefix + '.legend.gz')
    train_config = train_rnn_model.TrainConfig(
        args.output_prefix,
        args.batch_size,
        args.max_iteration_count,
        args.use_rsquare,
        args.num_threads,
        )
    set_weights_for_cross_entropy(
        a1_freqs[output_indexes], args.gamma, train_config)
    train_rnn_model.train(
        train_data_feeder, validation_inputs, validation_outputs,
        features, config, train_config)
    elapsed_time = time.time() - start_time
    with open(args.output_prefix + '.time', 'wt'):
        fout.write('Elapsed time: {:f} [s]\n'.format(elapsed_time))


if __name__ == '__main__':
    main()
