from argparse import ArgumentParser
import os

from common_utils import mkdir, reading, writing
from gen_utils import hapslegend2gen, merge


def main():
    description = 'merge hap and gens'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str,
                        dest='hap_file', required=True,
                        help='hap file')
    parser.add_argument('--legend', type=str,
                        dest='legend_file', required=True,
                        help='legend file')
    parser.add_argument('--gen-prefix', type=str,
                        dest='gen_prefix', required=True,
                        help='gen prefix')
    parser.add_argument('--start-index', type=int,
                        dest='start_index', required=True, help='start index')
    parser.add_argument('--end-index', type=int,
                        dest='end_index', required=True, help='start index')
    parser.add_argument('--output-file', type=str,
                        dest='output_file', required=True,
                        help='output file')
    args = parser.parse_args()

    mkdir(os.path.dirname(args.output_file))
    hapslegend2gen(
        args.hap_file, args.legend_file, args.output_file + '.tmp1.gz')
    with writing(args.output_file + '.tmp2.gz') as fout:
        for index in range(args.start_index, args.end_index + 1):
            gen_file = '{:s}_{:d}.gen.gz'.format(args.gen_prefix, index)
            with reading(gen_file) as fin:
                for line in fin:
                    fout.write(line)
    merge(
        args.output_file + '.tmp1.gz', args.output_file + '.tmp2.gz',
        args.output_file)
    os.remove(args.output_file + '.tmp1.gz')
    os.remove(args.output_file + '.tmp2.gz')


if __name__ == '__main__':
    main()
