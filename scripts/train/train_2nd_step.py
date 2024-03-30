from argparse import ArgumentParser
import os
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


def main():
    description = 'train 2nd step'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str, required=True,
                        dest='hap_file', help='hap file')
    parser.add_argument('--legend', type=str, required=True,
                        dest='legend_file', help='legend file')
    parser.add_argument('--config1', type=str, required=True,
                        dest='config_file1', help='config file1')
    parser.add_argument('--config2', type=str, required=True,
                        dest='config_file2', help='config file2')
    parser.add_argument('--init-checkpoint1', type=str, required=True,
                        dest='init_checkpoint_file1',
                        help='init checkpoint file1')
    parser.add_argument('--init-checkpoint2', type=str, required=True,
                        dest='init_checkpoint_file2',
                        help='init checkpoint file2')
    parser.add_argument('--output-prefix', type=str, required=True,
                        dest='output_prefix', help='output prefix')
    parser.add_argument('--batch-size', type=int, default=500,
                        dest='batch_size', help='batch size')
    parser.add_argument('--max-iteration-count', type=int, default=100000,
                        dest='max_iteration_count', help='max iteration count')
    parser.add_argument('--validation-sample-size', type=int, default=100,
                        dest='validation_sample_size',
                        help='validation sample size')
    parser.add_argument('--gamma', type=float, default=0, dest='gamma',
                        help='gamma value')
    parser.add_argument('--rsquare', action='store_true', default=False,
                        dest='use_rsquare',
                        help='use r-square values for validation')
    parser.add_argument('--random-seed', type=int, default=3141592653,
                        dest='random_seed', help='seed for random')
    parser.add_argument('--num-threads', type=int, default=1,
                        dest='num_threads', help='number of threads')
    args = parser.parse_args()

    start_time = time.time()
    mkdir(os.path.dirname(args.output_prefix))
    haplotypes, array_marker_flags, positions = load_data(
        args.hap_file, args.legend_file)
    input_indexes = array_marker_flags.nonzero()[0]
    output_indexes = np.logical_not(array_marker_flags).nonzero()[0]
    hap_id_ordering = get_random_hap_id_ordering(
        len(haplotypes) // 2, args.random_seed)
    validation_hap_sample_size = 2 * args.validation_sample_size
    train_hap_ids = hap_id_ordering[validation_hap_sample_size:]
    a1_freqs = get_a1_freqs(haplotypes[train_hap_ids])
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
    config1 = rnn_model.Config.load(args.config_file1)
    config2 = rnn_model.Config.load(args.config_file2)
    shutil.copy(args.legend_file, args.output_prefix + '.legend.gz')
    train_config = train_rnn_model.TrainConfig(
        args.output_prefix,
        args.batch_size,
        args.max_iteration_count,
        args.use_rsquare,
        args.num_threads,
        )
    train_config.init_checkpoint_file1 = args.init_checkpoint_file1
    train_config.init_checkpoint_file2 = args.init_checkpoint_file2
    set_weights_for_cross_entropy(
        a1_freqs[output_indexes], args.gamma, train_config)
    train_rnn_model.train_hybrid_model(
        train_data_feeder, validation_inputs, validation_outputs,
        config1, config2, train_config)
    elapsed_time = time.time() - start_time
    with open(args.output_prefix + '.time', 'wt'):
        fout.write('Elapsed time: {:f} [s]\n'.format(elapsed_time))


if __name__ == '__main__':
    main()
