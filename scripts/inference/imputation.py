from argparse import ArgumentParser
import glob
import os
import sys
import time

from common_utils import system
import split_hapslegend


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    description = 'RNN-IMP'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str, required=True,
                        dest='hap_file', help='hap file')
    parser.add_argument('--legend', type=str, required=True,
                        dest='legend_file', help='legend file')
    parser.add_argument('--sample', type=str, default=None,
                        dest='sample_file', help='sample file')
    parser.add_argument('--chromosome', type=str, default=None,
                        dest='chromosome', help='chromosome')
    parser.add_argument('--model-prefix', type=str, required=True,
                        dest='model_prefix', help='model file prefix')
    parser.add_argument('--output-prefix', type=str, required=True,
                        dest='output_prefix', help='output prefix')
    parser.add_argument('--output-format', type=str, default='gen',
                        dest='output_format', help='output format')
    parser.add_argument('--python3-bin', type=str, default='python3',
                        dest='python3_bin', help='path to Python3 binary')
    args = parser.parse_args()

    time1 = time.time()
    assert os.path.exists(args.hap_file)
    assert os.path.exists(args.legend_file)
    assert args.output_format in {'gen', 'vcf'}
    if args.output_format == 'vcf' and args.chromosome is None:
        print('Chromosome must be specified for vcf output', file=sys.stderr)
        sys.exit(-1)
    time2 = time.time()
    model_legend_file_list = sorted(
        glob.glob(args.model_prefix + '*.legend.gz'),
        key=lambda x: int(x[:-10].split('_')[-1]))
    splitted_data_dict_list = split_hapslegend.split_data(
        args.hap_file, args.legend_file, model_legend_file_list,
        args.output_prefix)
    time3 = time.time()
    print('Elapsed time for data preparation: {:f} [s]'.format(time3 - time2))

    for i, splitted_data_dict in enumerate(splitted_data_dict_list, start=1):
        print('{:d} / {:d}'.format(i, len(splitted_data_dict_list)))
        command = args.python3_bin
        command += ' ' + os.path.join(SCRIPT_DIR, 'inference.py')
        command += ' --hap ' + splitted_data_dict.hap_file
        command += ' --legend ' + splitted_data_dict.legend_file
        command += ' --model-prefix ' + splitted_data_dict.model_prefix
        command += ' --output-format ' + args.output_format
        if args.output_format == 'vcf':
            output_file = '{:s}_{:d}.vcf.gz'.format(args.output_prefix, i)
        elif args.output_format == 'gen':
            output_file = '{:s}_{:d}.gen.gz'.format(args.output_prefix, i)
        command += ' --output-file ' + output_file
        system(command)
    time4 = time.time()
    print('Elapsed time for inference: {:f} [s]'.format(time4 - time3))

    if args.output_format == 'gen':
        command = args.python3_bin
        command += ' ' + os.path.join(
            SCRIPT_DIR, 'merge_hapslegend_and_gens.py')
        command += ' --hap ' + args.hap_file
        command += ' --legend ' + args.legend_file
        command += ' --gen-prefix ' + args.output_prefix
        command += ' --start-index 1'
        command += ' --end-index {:d}'.format(len(splitted_data_dict_list))
        command += ' --output-file {:s}.gen'.format(args.output_prefix)
        system(command)
    elif args.output_format == 'vcf':
        command = args.python3_bin
        command += ' ' + os.path.join(
            SCRIPT_DIR, 'merge_hapslegend_and_vcfs.py')
        command += ' --hap ' + args.hap_file
        command += ' --legend ' + args.legend_file
        if args.sample_file is not None:
            command += ' --sample ' + args.sample_file
        command += ' --chromosome ' + args.chromosome
        command += ' --vcf-prefix ' + args.output_prefix
        command += ' --start-index 1'
        command += ' --end-index {:d}'.format(len(splitted_data_dict_list))
        command += ' --output-file {:s}.vcf'.format(args.output_prefix)
        system(command)
    else:
        print('Unsupported format: ' + args.output_format, file=sys.stderr)
        sys.exit(0)
    time5 = time.time()
    print('Elapsed time for merge files: {:f} [s]'.format(time5 - time4))
    print('Total elapsed time: {:f} [s]'.format(time5 - time1))


if __name__ == '__main__':
    main()
