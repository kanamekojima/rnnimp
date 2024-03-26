from argparse import ArgumentParser
import os
import sys

from common_utils import mkdir, reading, writing
from vcf_utils import hapslegend2vcf, merge


def main():
    description = 'merge hapslegend and vcfs'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str,
                        dest='hap_file', required=True,
                        help='hap file')
    parser.add_argument('--legend', type=str,
                        dest='legend_file', required=True,
                        help='legend file')
    parser.add_argument('--sample', type=str,
                        dest='sample_file', default=None,
                        help='sample file')
    parser.add_argument('--vcf-prefix', type=str,
                        dest='vcf_prefix', required=True,
                        help='vcf prefix')
    parser.add_argument('--start-index', type=int,
                        dest='start_index', required=True, help='start index')
    parser.add_argument('--end-index', type=int,
                        dest='end_index', required=True, help='start index')
    parser.add_argument('--chromosome', type=str,
                        dest='chromosome', required=True, help='chromosome')
    parser.add_argument('--output-file', type=str,
                        dest='output_file', required=True,
                        help='output file')
    args = parser.parse_args()

    mkdir(os.path.dirname(args.output_file))
    hapslegend2vcf(
        args.hap_file, args.legend_file, args.output_file + '.tmp1.gz')
    with writing(args.output_file + '.tmp2.gz') as fout:
        for index in range(args.start_index, args.end_index + 1):
            vcf_file = '{:s}_{:d}.vcf.gz'.format(args.vcf_prefix, index)
            if not os.path.exists(vcf_file):
                print('Warning: {:s} not found'.format(vcf_file))
                continue
            with reading(vcf_file) as fin:
                for line in fin:
                    fout.write(line)
    merge(
        args.output_file + '.tmp1.gz',
        args.output_file + '.tmp2.gz',
        args.chromosome,
        args.sample_file,
        args.output_file,
        )
    os.remove(args.output_file + '.tmp1.gz')
    os.remove(args.output_file + '.tmp2.gz')


if __name__ == '__main__':
    main()
