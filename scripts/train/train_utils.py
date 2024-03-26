import math
import os
import sys

import numpy as np

from common_utils import reading
import train_rnn_model


NA_ALLELE = -9
NUM_RECOMBINATION_POINTS = 2
BLOCK_SIZE = 3


def load_data(hap_file, legend_file):
    array_marker_flags = []
    positions = []
    with reading(legend_file) as fin:
        items = fin.readline().strip().split()
        try:
            marker_flag_col = items.index('array_marker_flag')
            position_col = items.index('position')
        except ValueError:
            print(
                'Some header items not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin:
            items = line.strip().split()
            array_marker_flag = items[marker_flag_col] == '1'
            position = items[position_col]
            positions.append(int(position))
            array_marker_flags.append(array_marker_flag)
    array_marker_flags = np.array(array_marker_flags, bool)
    positions = np.array(positions, np.int32)

    print('array marker count   : {:d}'.format(
        np.count_nonzero(array_marker_flags)))
    print('imputation site count: {:d}'.format(
        len(array_marker_flags) - np.count_nonzero(array_marker_flags)))

    num_haplotypes = 0
    with reading(hap_file) as fin:
        items = fin.readline().rstrip().split(' ')
        num_haplotypes = len(items)

    haplotypes = np.full(
        [num_haplotypes, len(positions)], NA_ALLELE, dtype=np.int32)
    with reading(hap_file) as fin:
        for i, line in enumerate(fin):
            items = line.rstrip().split(' ')
            for j, item in enumerate(items):
                if item == '0' or item == '1':
                    haplotypes[j][i] = int(item)
    return haplotypes, array_marker_flags, positions


def one_hot(allele, a1_freq=None):
    if allele == NA_ALLELE:
        if a1_freq is None:
            return [0.5, 0.5]
        return [1.0 - a1_freq, a1_freq]
    return [1 - allele, allele]


def get_random_hap_id_ordering(num_samples, seed):
    ordering = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(ordering)
    return np.stack([2 * ordering, 2 * ordering + 1], axis=1).reshape(-1)


def get_a1_freqs(haplotypes):
    a1_freqs = []
    for i in range(haplotypes.shape[1]):
        count = (haplotypes[:, i] != -1).astype(np.int64).sum()
        pos_count = (haplotypes[:, i] == 1).astype(np.int64).sum()
        a1_freq = pos_count / float(count)
        a1_freqs.append(a1_freq)
    return np.array(a1_freqs, np.float64)


def set_weights_for_cross_entropy(a1_freqs, gamma, train_config):

    def softened_freq(freq, gamma):
        maf = max(min(freq, 1.0 - freq), 0.005)
        return 0.5 * math.pow(2.0 * maf, gamma)

    def get_pos_weight(a1_freq, zeta):
        if a1_freq < 0.5:
            return 1.0 - softened_freq(a1_freq, zeta)
        return softened_freq(1.0 - a1_freq, zeta)

    def get_weights(a1_freq, gamma):
        pos_weight = get_pos_weight(a1_freq, 0.5)
        class_weight = [1.0 - pos_weight, pos_weight]
        base_weight = softened_freq(a1_freq, gamma)
        base_weight /= class_weight[0] * (1.0 - a1_freq) \
                       + class_weight[1] * a1_freq
        return [base_weight * class_weight[0], base_weight * class_weight[1]]

    train_config.weights_for_cross_entropy = [
        softened_freq(a1_freq, gamma) for a1_freq in a1_freqs
    ]


def get_recombination_probs(positions, k, recombination_rate, Ne):
    distances = positions[1:] - positions[:-1]
    rho = 4 * Ne * recombination_rate
    recombination_probs = 1.0 - np.exp(- distances * rho / k)
    return recombination_probs


class AugmentedDataFeeder:

    def __init__(
            self,
            inputs,
            outputs,
            positions,
            input_indexes,
            output_indexes,
            recombination_rate=1.0e-8,
            Ne=1e4,
            block_size=BLOCK_SIZE,
            num_recombination_points=NUM_RECOMBINATION_POINTS,
            ):
        assert len(input_indexes) + len(output_indexes) == len(positions)
        assert len(inputs) == len(outputs)
        self.input_indexes = input_indexes
        self.output_indexes = output_indexes
        self.haplotypes = np.zeros(
            [inputs.shape[0], len(positions), 2], dtype=inputs.dtype)
        self.haplotypes[:, self.input_indexes] = inputs
        self.haplotypes[:, self.output_indexes] = outputs

        self.block_size = block_size
        self.num_recombination_points = num_recombination_points

        r = self.haplotypes.shape[0] % self.block_size
        if r == 0:
            self.augmented_haplotypes = np.zeros_like(self.haplotypes)
        else:
            self.augmented_haplotypes = np.zeros_like(self.haplotypes[:-r])
            assert self.augmented_haplotypes.shape[0] % self.block_size == 0
        self.recombination_prob_dist = get_recombination_probs(
            positions, self.block_size, recombination_rate, Ne)
        self.recombination_prob_dist /= self.recombination_prob_dist.sum()

        self.input_shape = inputs.shape[1:]
        self.output_shape = outputs.shape[1:]
        self.sample_size = self.augmented_haplotypes.shape[0]

    def get_augmented_data(self):
        recombination_points = np.zeros(
            self.num_recombination_points + 1, np.int64)
        recombination_points[-1] = self.haplotypes.shape[1]
        r_indexes = np.arange(self.haplotypes.shape[0])
        s_indexes = np.arange(self.block_size)
        np.random.shuffle(r_indexes)
        for i in range(self.augmented_haplotypes.shape[0] // self.block_size):
            if self.num_recombination_points >= 1:
                recombination_points[:-1] = np.random.choice(
                    len(self.recombination_prob_dist),
                    self.num_recombination_points, replace=False,
                    p=self.recombination_prob_dist) + 1
                recombination_points.sort()
                assert recombination_points[-1] == self.haplotypes.shape[1]
            p = 0
            for p_next in recombination_points:
                np.random.shuffle(s_indexes)
                for j, s_index in enumerate(s_indexes):
                    index = self.block_size * i + j
                    augmented_haplotype = self.augmented_haplotypes[index]
                    index = r_indexes[self.block_size * i + s_index]
                    haplotype = self.haplotypes[index]
                    augmented_haplotype[p:p_next] = haplotype[p:p_next]
                p = p_next
        inputs = self.augmented_haplotypes[:, self.input_indexes]
        outputs = self.augmented_haplotypes[:, self.output_indexes]
        return inputs, outputs

    def get_next(self):
        inputs, outputs = self.get_augmented_data()
        return inputs, outputs
