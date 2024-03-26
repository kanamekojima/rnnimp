from argparse import ArgumentParser
from collections import namedtuple
import sys

import numpy as np
import onnxruntime

from common_utils import mkdir, reading
from gen_utils import write_gen
from vcf_utils import write_vcf


NA_ALLELE = -9


LegendRecord = namedtuple(
    'LegendRecord', (
        'id',
        'position',
        'a0',
        'a1',
        'array_marker_flag',
        'a1_freq',
    )
)


def load_legend(legend_file):
    legend_record_list = []
    with reading(legend_file) as fin:
        items = fin.readline().strip().split()
        try:
            id_col = items.index('id')
            a0_col = items.index('a0')
            a1_col = items.index('a1')
            position_col = items.index('position')
            marker_flag_col = items.index('array_marker_flag')
            a1_freq_col = items.index('a1_freq')
        except ValueError:
            print(
                'Some header items not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin:
            items = line.strip().split()
            a1_freq = items[a1_freq_col]
            legend_record = LegendRecord(
                items[id_col],
                items[position_col],
                items[a0_col],
                items[a1_col],
                items[marker_flag_col] == '1',
                None if a1_freq == 'NA' else float(a1_freq)
            )
            legend_record_list.append(legend_record)
    return legend_record_list


def load_data(hap_file, legend_file, legend_record_list):

    def allele_check(a0, a1, legend_record):
        if a0 == legend_record.a0 and (a1 == legend_record.a1 or a1 == '0'):
            return True, False
        if a0 == legend_record.a1 and (a1 == legend_record.a0 or a1 == '0'):
            return True, True
        return False, None

    legend_record_dict = {}
    marker_id_dict = {}
    for i, legend_record in enumerate(legend_record_list):
        legend_record_dict[legend_record.position] = legend_record
        marker_id_dict[legend_record.position] = i

    load_info_list = []
    with reading(legend_file) as fin:
        items = fin.readline().strip().split()
        try:
            a0_col = items.index('a0')
            a1_col = items.index('a1')
            position_col = items.index('position')
        except ValueError:
            print(
                'Some header items not found in '+ legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin:
            items = line.strip().split()
            position = items[position_col]
            if position in legend_record_dict:
                a0 = items[a0_col]
                a1 = items[a1_col]
                legend_record = legend_record_dict[position]
                match_flag, swap_flag = allele_check(a0, a1, legend_record)
                if match_flag:
                    marker_id = marker_id_dict[position]
                    load_info_list.append([marker_id, swap_flag])
                else:
                    load_info_list.append(None)
            else:
                load_info_list.append(None)

    num_haplotypes = 0
    with reading(hap_file) as fin:
        items = fin.readline().strip().split()
        num_haplotypes = len(items)

    haplotypes = np.full(
        [num_haplotypes, len(legend_record_list)], NA_ALLELE, dtype=np.int32)
    with reading(hap_file) as fin:
        for line, load_info in zip(fin, load_info_list):
            if load_info is None:
                continue
            items = line.strip().split()
            marker_id, swap_flag = load_info
            for i, item in enumerate(items):
                if item == '0' or item == '1':
                    allele = int(item)
                    if swap_flag:
                        allele = 1 - allele
                    haplotypes[i, marker_id] = allele
    return haplotypes


def one_hot(allele, a1_freq=None):
    if allele == NA_ALLELE:
        if a1_freq is None:
            return [0.5, 0.5]
        return [1.0 - a1_freq, a1_freq]
    return [1 - allele, allele]


def predict(inputs, model_file):
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    sess_options.enable_mem_reuse = False
    sess = onnxruntime.InferenceSession(
        model_file, sess_options, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    predictions = sess.run(None, {input_name: inputs})[0]
    return predictions


def main():
    description = 'inference'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str, required=True,
                        dest='hap_file', help='hap file')
    parser.add_argument('--legend', type=str, required=True,
                        dest='legend_file', help='legend file')
    parser.add_argument('--model-prefix', type=str, required=True,
                        dest='model_prefix', help='model file prefix')
    parser.add_argument('--output-format', type=str, default='gen',
                        dest='output_format', help='output format')
    parser.add_argument('--output-file', type=str, required=True,
                        dest='output_file', help='output file')
    args = parser.parse_args()

    legend_record_list = load_legend(args.model_prefix + '.legend.gz')
    marker_legend_record_list = [
        legend_record for legend_record in legend_record_list
        if legend_record.array_marker_flag
    ]
    haplotypes = load_data(
        args.hap_file, args.legend_file, marker_legend_record_list)
    a1_freqs = [
        legend_record.a1_freq for legend_record in marker_legend_record_list
    ]
    inputs = np.array([
        [one_hot(haplotype[i], a1_freq) for i, a1_freq in enumerate(a1_freqs)]
        for haplotype in haplotypes
    ], dtype=np.float32)
    predictions = predict(inputs, args.model_prefix + '.ort')
    imputed_legend_record_list = [
        legend_record for legend_record in legend_record_list
        if not legend_record.array_marker_flag
    ]
    if args.output_format == 'gen':
        write_gen(predictions, imputed_legend_record_list, args.output_file)
    elif args.output_format == 'vcf':
        write_vcf(predictions, imputed_legend_record_list, args.output_file)
    else:
        print('Unsupported format: ' + args.output_format, file=sys.stderr)


if __name__ == '__main__':
    main()
