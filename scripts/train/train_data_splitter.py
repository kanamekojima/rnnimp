from argparse import ArgumentParser
import gc
import gzip
import os
import sys

from common_utils import mkdir, reading, writing


class HapHandler():

    def __init__(self, hap_file):
        _, ext = os.path.splitext(hap_file)
        if ext == '.gz':
            self.fp = gzip.open(hap_file, 'rt')
        else:
            self.fp = open(hap_file, 'rt')
        self.max_index_in_line_buffer = -1
        self.min_index_in_line_buffer = -1
        self.line_buffer = []

    def get_line(self, index):
        assert index >= self.min_index_in_line_buffer, (
            'index {:s} must be >= min_index_in_line_buffer {:s}'.format(
                index, self.min_index_in_line_buffer))
        assert index <= self.max_index_in_line_buffer, (
            'index {:s} must be <= max_index_in_line_buffer {:s}'.format(
                index, self.max_index_in_line_buffer))
        return self.line_buffer[index - self.min_index_in_line_buffer]

    def load_to_buffer(self, min_index, max_index):
        assert min_index >= self.min_index_in_line_buffer, (
            'index {:s} must be >= min_index_in_line_buffer {:s}'.format(
                min_index, self.min_index_in_line_buffer))

        if max_index < self.max_index_in_line_buffer:
            max_index = self.max_index_in_line_buffer

        new_line_buffer = [None] * (max_index - min_index + 1)
        new_line_buffer_count = 0
        if min_index <= self.max_index_in_line_buffer:
            for index in range(min_index, self.max_index_in_line_buffer + 1):
                line = self.line_buffer[index - self.min_index_in_line_buffer]
                new_line_buffer[new_line_buffer_count] = line
                new_line_buffer_count += 1
            min_index = self.max_index_in_line_buffer + 1
        else:
            for _ in range(self.max_index_in_line_buffer + 1, min_index):
                self.fp.readline()

        self.line_buffer = new_line_buffer
        self.max_index_in_line_buffer = max_index
        self.min_index_in_line_buffer = (
            self.max_index_in_line_buffer - len(self.line_buffer) + 1)
        gc.collect()
        for _ in range(min_index, max_index + 1):
            line = self.fp.readline().rstrip()
            self.line_buffer[new_line_buffer_count] = line
            new_line_buffer_count += 1

    def close(self):
        self.fp.close()


def prepare_modified_hap(
        hap_file,
        legend_file,
        maf_threshold,
        suppress_allele_flip,
        output_prefix,
        ):
    mkdir(os.path.dirname(output_prefix))
    a1_freq_list = []
    swap_flag_list = []
    skip_flag_list = []
    array_marker_flag_list = []
    with reading(legend_file) as fin:
        items = fin.readline().rstrip().split()
        try:
            array_marker_flag_col = items.index('array_marker_flag')
        except ValueError:
            print(
                'header \'array_marker_flag\' not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin:
            items = line.rstrip().split()
            array_marker_flag = items[array_marker_flag_col] == '1'
            array_marker_flag_list.append(array_marker_flag)
    with reading(hap_file) as fin, \
         writing(output_prefix + '.hap.gz') as fout:
        for i, line in enumerate(fin):
            items = line.rstrip().split()
            a0_count = items.count('0')
            a1_count = items.count('1')
            swap_flag = a0_count < a1_count and array_marker_flag_list[i]
            if suppress_allele_flip:
                swap_flag = False
            if swap_flag:
                for j, item in enumerate(items):
                    if item == '0':
                        items[j] = '1'
                    elif item == '1':
                        items[j] = '0'
            a1_freq = 0
            if a0_count + a1_count > 0:
                if swap_flag:
                    a1_freq = a0_count / float(a0_count + a1_count)
                else:
                    a1_freq = a1_count / float(a0_count + a1_count)
            a1_freq_list.append(str(a1_freq))
            swap_flag_list.append(swap_flag)
            if a1_freq < maf_threshold and array_marker_flag_list[i] == False:
                skip_flag_list.append(True)
            else:
                skip_flag_list.append(False)
                fout.write(' '.join(items))
                fout.write('\n')
    with reading(legend_file) as fin, \
         writing(output_prefix + '.legend.gz') as fout:
        header = fin.readline().rstrip()
        items = header.split()
        try:
            a0_col = items.index('a0')
            a1_col = items.index('a1')
        except ValueError:
            print(
                'some header item not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        a1_freq_col = items.index('a1_freq') if 'a1_freq' in items else None
        swap_col = items.index('swap') if 'swap' in items else None
        fout.write(header)
        if a1_freq_col is None:
            fout.write(' a1_freq')
        if swap_col is None:
            fout.write(' swap')
        fout.write('\n')
        for i, line in enumerate(fin):
            if skip_flag_list[i]:
                continue
            items = line.rstrip().split()
            swap_flag = swap_flag_list[i]
            if swap_flag:
                items[a0_col], items[a1_col] = items[a1_col], items[a0_col]
            if a1_freq_col is None:
                items.append(a1_freq_list[i])
            else:
                items[a1_freq_col] = a1_freq_list[i]
            if swap_col is None:
                items.append('1' if swap_flag else '0')
            else:
                items[swap_col] = '1' if swap_flag else '0'
            fout.write(' '.join(items))
            fout.write('\n')


def load_legend_info_list(legend_file):
    legend_info_list = []
    with reading(legend_file) as fin:
        items = fin.readline().rstrip().split(' ')
        try:
            position_col = items.index('position')
            array_marker_flag_col = items.index('array_marker_flag')
        except ValueError:
            print(
                'header \'array_marker_flag\' not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin:
            items = line.rstrip().split(' ')
            position = int(items[position_col])
            array_marker_flag = items[array_marker_flag_col] == '1'
            legend_info_list.append({
                'position': position,
                'array_marker_flag': array_marker_flag,
            })
    return legend_info_list


def load_partition_position_list(partition_file):
    partition_position_list = [1]
    if partition_file is not None:
        with open(partition_file, 'rt') as fin:
            fin.readline()
            for line in fin:
                _, _, stop = line.rstrip().split()
                partition_position_list.append(int(stop) + 1)
    return partition_position_list


def get_next_imp_region_info(
        start_index,
        legend_info_list,
        imp_site_count_limit,
        body_marker_count_limit,
        ):
    imp_start_index = None
    imp_end_index = None
    imp_site_count = 0
    marker_count = 0
    for index in range(start_index, len(legend_info_list)):
        if legend_info_list[index]['array_marker_flag']:
            if imp_start_index is None:
                continue
            if imp_site_count >= imp_site_count_limit:
                break
            if marker_count >= body_marker_count_limit:
                break
            marker_count += 1
        else:
            if imp_start_index is None:
                imp_start_index = index
            imp_end_index = index
            imp_site_count += 1

    if imp_start_index is None:
        return None
    return {
        'imp_start_index': imp_start_index,
        'imp_end_index': imp_end_index,
        'imp_start_position': legend_info_list[imp_start_index]['position'],
        'imp_end_position': legend_info_list[imp_end_index]['position'],
        'remaining_marker_count_limit': body_marker_count_limit - marker_count,
        'imp_site_count': imp_site_count,
    }


def get_local_imp_region_info_list(
        legend_info_list,
        imp_site_count_limit,
        body_marker_count_limit,
        index_origin,
        ):
    imp_region_info_list = []
    start_index = 0
    while True:
        imp_region_info = get_next_imp_region_info(
            start_index, legend_info_list, imp_site_count_limit,
            body_marker_count_limit)
        if imp_region_info is None:
            if len(imp_region_info_list) > 1:
                imp_region_info1 = imp_region_info_list[-2]
                imp_region_info2 = imp_region_info_list[-1]
                imp_site_count = imp_region_info2['imp_site_count']
                if imp_site_count < 0.2 * imp_site_count_limit:
                    imp_start_index = imp_region_info1['imp_start_index']
                    imp_end_index = imp_region_info2['imp_end_index']
                    marker_count = 0
                    imp_site_count = 0
                    for index in range(imp_start_index, imp_end_index + 1):
                        if legend_info_list[index]['array_marker_flag']:
                            marker_count += 1
                        else:
                            imp_site_count += 1
                    remaining_count = body_marker_count_limit - marker_count
                    imp_start_position = imp_region_info1['imp_start_position']
                    imp_end_position = imp_region_info2['imp_end_position']
                    if marker_count <= body_marker_count_limit:
                        imp_region_info_list = imp_region_info_list[:-2]
                        imp_region_info = {
                            'imp_start_index': imp_start_index,
                            'imp_end_index': imp_end_index,
                            'imp_start_position': imp_start_position,
                            'imp_end_position': imp_end_position,
                            'remaining_marker_count_limit': remaining_count,
                            'imp_site_count': imp_site_count,
                        }
                        imp_region_info_list.append(imp_region_info)
            break
        imp_region_info_list.append(imp_region_info)
        start_index = imp_region_info['imp_end_index'] + 1
    for imp_region_info in imp_region_info_list:
        imp_region_info['imp_start_index'] += index_origin
        imp_region_info['imp_end_index'] += index_origin
    return imp_region_info_list


def split_legend_info_list(legend_info_list, partition_position_list):
    if len(partition_position_list) <= 1:
        return [0, len(legend_info_list) - 1]
    assert partition_position_list[-1] > legend_info_list[-1]['position']
    index_pair_list = []
    index = 0
    for partition_position in partition_position_list[1:]:
        start_index = index
        for legend_info in legend_info_list[start_index:]:
            if legend_info['position'] >= partition_position:
                break
            index += 1
        if index > start_index:
            index_pair_list.append([start_index, index - 1])
        if index >= len(legend_info_list):
            break
    assert index_pair_list[-1][1] == len(legend_info_list) - 1
    return index_pair_list


def get_imp_region_info_list(
        legend_info_list,
        imp_site_count_limit,
        body_marker_count_limit,
        flanking_marker_count_limit,
        partition_position_list,
        ):
    last_imp_index = None
    for index in range(len(legend_info_list) - 1, -1, -1):
        if not legend_info_list[index]['array_marker_flag']:
            last_imp_index = index
            break
    if last_imp_index is None:
        return None
    last_partition_position = legend_info_list[last_imp_index]['position'] + 1
    partition_position_list[-1] = last_partition_position
    index_pair_list = split_legend_info_list(
        legend_info_list, partition_position_list)
    imp_region_info_list = []
    for start_index, end_index in index_pair_list:
        imp_region_info_list.extend(
            get_local_imp_region_info_list(
                legend_info_list[start_index: end_index + 1],
                imp_site_count_limit, body_marker_count_limit,
                start_index))

    for i, imp_region_info in enumerate(imp_region_info_list):
        imp_start_index = imp_region_info['imp_start_index']
        remaining_count_limit = imp_region_info['remaining_marker_count_limit']
        margin_start_index = imp_start_index
        marker_count = 0
        count_limit = flanking_marker_count_limit
        count_limit += remaining_count_limit // 2
        for index in range(imp_start_index - 1, -1, -1):
            if legend_info_list[index]['array_marker_flag']:
                marker_count += 1
                if marker_count > count_limit:
                    marker_count -= 1
                    break
                margin_start_index = index
        assert count_limit - marker_count >= 0

        count_limit = flanking_marker_count_limit + count_limit - marker_count
        count_limit += remaining_count_limit // 2
        imp_end_index = imp_region_info['imp_end_index']
        margin_end_index = imp_end_index
        marker_count = 0
        for index in range(imp_end_index + 1, len(legend_info_list)):
            if legend_info_list[index]['array_marker_flag']:
                marker_count += 1
                if marker_count > count_limit:
                    marker_count -= 1
                    break
                margin_end_index = index
        assert count_limit - marker_count >= 0
        count_limit = count_limit - marker_count

        marker_count = 0
        for index in range(margin_start_index - 1, -1, -1):
            if legend_info_list[index]['array_marker_flag']:
                marker_count += 1
                if marker_count > count_limit:
                    marker_count -= 1
                    break
                margin_start_index = index
        assert count_limit - marker_count >= 0

        legend_info = legend_info_list[margin_start_index]
        margin_start_position = legend_info['position']
        legend_info = legend_info_list[margin_end_index]
        margin_end_position = legend_info['position']

        imp_region_info['margin_start_index'] = margin_start_index
        imp_region_info['margin_end_index'] = margin_end_index
        imp_region_info['margin_start_position'] = margin_start_position
        imp_region_info['margin_end_position'] = margin_end_position
    return imp_region_info_list


def get_indices(imp_region_info, legend_info_list):
    indices = []
    margin_start_index = imp_region_info['margin_start_index']
    margin_end_index = imp_region_info['margin_end_index']
    imp_start_index = imp_region_info['imp_start_index']
    imp_end_index = imp_region_info['imp_end_index']
    for index in range(margin_start_index, imp_start_index):
        if legend_info_list[index]['array_marker_flag']:
            indices.append(index)
    for index in range(imp_start_index, imp_end_index + 1):
        indices.append(index)
    for index in range(imp_end_index + 1, margin_end_index + 1):
        if legend_info_list[index]['array_marker_flag']:
            indices.append(index)
    return indices


def get_imp_region_stats(imp_region_info, legend_info_list):
    margin_start_index = imp_region_info['margin_start_index']
    margin_end_index = imp_region_info['margin_end_index']
    imp_start_index = imp_region_info['imp_start_index']
    imp_end_index = imp_region_info['imp_end_index']
    marker_count = 0
    imp_site_count = 0
    for index in range(margin_start_index, imp_start_index):
        if legend_info_list[index]['array_marker_flag']:
            marker_count += 1
    for index in range(imp_start_index, imp_end_index + 1):
        if legend_info_list[index]['array_marker_flag']:
            marker_count += 1
        else:
            imp_site_count += 1
    for index in range(imp_end_index + 1, margin_end_index + 1):
        if legend_info_list[index]['array_marker_flag']:
            marker_count += 1
    return imp_site_count, marker_count


def main():
    description = 'train data splitter'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument(
        '-h', '--hap', type=str, dest='hap_file', required=True,
        help='hap file')
    parser.add_argument(
        '-l', '--legend', type=str, dest='legend_file', required=True,
        help='legend file')
    parser.add_argument(
        '-p', '--partition', type=str, dest='partition_file', default=None,
        help='partition file')
    parser.add_argument(
        '-o', '--output-prefix', type=str, dest='output_prefix',
        required=True, help='output prefix')
    parser.add_argument(
        '-m', '--flanking-marker-count-limit', type=int,
        dest='flanking_marker_count_limit', default=50,
        help='flanking marker count limit')
    parser.add_argument(
        '-b', '--body-marker-count-limit', type=int,
        dest='body_marker_count_limit', default=100,
        help='body marker_count limit')
    parser.add_argument(
        '-i', '--imp-site-count-limit', type=int,
        dest='imp_site_count_limit', default=100,
        help='imp_site_count limit')
    parser.add_argument(
        '-f', '--maf-threshold', type=float, dest='maf_threshold',
        default=0.005, help = 'MAF threshold')
    parser.add_argument(
        '--suppress-allele-flip', action='store_true', default=False,
        dest='suppress_allele_flip', help='suppress allele flip')
    args = parser.parse_args()

    prepare_modified_hap(
        args.hap_file, args.legend_file, args.maf_threshold,
        args.suppress_allele_flip, args.output_prefix + '_mod')

    legend_info_list = load_legend_info_list(
        args.output_prefix + '_mod.legend.gz')
    partition_position_list = load_partition_position_list(
        args.partition_file)
    imp_region_info_list = get_imp_region_info_list(
        legend_info_list, args.imp_site_count_limit,
        args.body_marker_count_limit, args.flanking_marker_count_limit,
        partition_position_list)

    print('No. of splitted regions: {:d}'.format(len(imp_region_info_list)))

    hap_handler = HapHandler(args.output_prefix + '_mod.hap.gz')
    legend_lines = []
    with reading(args.output_prefix + '_mod.legend.gz') as fin:
        legend_header = fin.readline()
        for line in fin:
            legend_lines.append(line)
    for i, imp_region_info in enumerate(imp_region_info_list, start=1):
        indices = get_indices(imp_region_info, legend_info_list)
        hap_handler.load_to_buffer(indices[0], indices[-1])
        output_file = '{:s}_{:d}.hap.gz'.format(args.output_prefix, i)
        print(output_file, flush=True)
        with writing(output_file) as fout:
            for index in indices:
                fout.write(hap_handler.get_line(index))
                fout.write('\n')
        output_file = '{:s}_{:d}.legend.gz'.format(args.output_prefix, i)
        with writing(output_file) as fout:
            fout.write(legend_header)
            for index in indices:
                fout.write(legend_lines[index])
    hap_handler.close()

    mkdir(os.path.dirname(args.output_prefix))
    with open(args.output_prefix + '_region_info.txt', 'wt') as fout:
        line = '\t'.join([
            'imp_start_pos',
            'imp_end_pos',
            'margin_start_pos',
            'margin_end_pos',
            'imp_site_count',
            'marker_count',
            'total_count',
        ])
        fout.write(line)
        fout.write('\n')
        for imp_region_info in imp_region_info_list:
            imp_site_count, marker_count = get_imp_region_stats(
                imp_region_info, legend_info_list)
            assert imp_site_count == imp_region_info['imp_site_count']
            line = '\t'.join(map(str, [
                imp_region_info['imp_start_position'],
                imp_region_info['imp_end_position'],
                imp_region_info['margin_start_position'],
                imp_region_info['margin_end_position'],
                imp_site_count,
                marker_count,
                imp_site_count + marker_count,
            ]))
            fout.write(line)
            fout.write('\n')
    with open(args.output_prefix + '.list', 'wt') as fout:
        for i in range(1, len(imp_region_info_list) + 1):
            fout.write('{:s}_{:d}\n'.format(args.output_prefix, i))


if __name__ == '__main__':
    main()
