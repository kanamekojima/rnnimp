from collections import namedtuple
import os
import sys

from common_utils import mkdir, reading, writing


SplittedDataInfo = namedtuple(
    'SplittedDataInfo', (
        'hap_file',
        'legend_file',
        'model_prefix',
        'start_index',
        'end_index',
    )
)


def split_legend(legend_file, model_legend_file_list, output_prefix):
    with reading(legend_file) as fin:
        legend_header = fin.readline().strip()
        header_items = legend_header.split(' ')
        try:
            position_col = header_items.index('position')
        except ValueError:
            print(
                'header \'position\' not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        legend_lines = []
        for line in fin:
            legend_lines.append(line.strip())
    positions = [
        int(line.rstrip().split(' ')[position_col]) for line in legend_lines
    ]
    splitted_data_info_list = []
    previous_start_index = 0
    max_index = len(positions) - 1
    for i, model_legend_file in enumerate(model_legend_file_list, start=1):
        with reading(model_legend_file) as fin:
            header_items = fin.readline().rstrip().split(' ')
            try:
                position_col = header_items.index('position')
            except ValueError:
                print(
                    'header \'position\' not found in ' + model_legend_file,
                    file=sys.stderr)
                sys.exit(0)
            items = fin.readline().rstrip().split(' ')
            start_position = int(items[position_col])
            for line in fin:
                items = line.rstrip().split(' ')
                end_position = int(items[1])
        start_index = max_index
        end_index = max_index
        for index in range(previous_start_index, max_index + 1):
            if positions[index] >= start_position:
                start_index = index
                break
        for index in range(start_index, max_index + 1):
            if positions[index] > end_position:
                end_index = index - 1
                break
        previous_start_index = start_index
        output_hap_file = '{:s}_{:d}.hap.gz'.format(output_prefix, i)
        output_legend_file = '{:s}_{:d}.legend.gz'.format(output_prefix, i)
        mkdir(os.path.dirname(output_legend_file))
        with writing(output_legend_file) as fout:
            fout.write(legend_header)
            fout.write('\n')
            for index in range(start_index, end_index + 1):
                fout.write(legend_lines[index])
                fout.write('\n')
        assert model_legend_file.endswith('.legend.gz')
        splitted_data_info = SplittedDataInfo(
            hap_file=output_hap_file,
            legend_file=output_legend_file,
            model_prefix='.'.join(model_legend_file.split('.')[:-2]),
            start_index=start_index,
            end_index=end_index,
        )
        splitted_data_info_list.append(splitted_data_info)
    return splitted_data_info_list


def split_hap(hap_file, splitted_data_info_list):
    with reading(hap_file) as fin:
        f_index = 0
        for i, splitted_data_info in enumerate(splitted_data_info_list):
            start_index = splitted_data_info.start_index
            end_index = splitted_data_info.end_index
            if i == len(splitted_data_info_list) - 1:
                next_start_index = end_index
            else:
                next_start_index = splitted_data_info_list[i + 1].start_index
            next_line_buffer = []
            output_hap_file = splitted_data_info.hap_file
            mkdir(os.path.dirname(output_hap_file))
            with writing(output_hap_file) as fout:
                for index in range(start_index, end_index + 1):
                    if f_index > index:
                        line = line_buffer[index - start_index]
                    else:
                        while f_index < index:
                            line = fin.readline()
                            if f_index >= next_start_index:
                                next_lines.append(line)
                            f_index += 1
                        line = fin.readline()
                        f_index += 1
                    if index >= next_start_index:
                        next_line_buffer.append(line)
                    fout.write(line)
            for index in range(end_index + 1, f_index):
                line = line_buffer[index - start_index]
                if index >= next_start_index:
                    next_line_buffer.append(line)
            line_buffer = next_line_buffer
            assert len(line_buffer) == f_index - next_start_index


def split_data(hap_file, legend_file, model_legend_file_list, output_prefix):
    splitted_data_info_list = split_legend(
        legend_file, model_legend_file_list, output_prefix)
    split_hap(hap_file, splitted_data_info_list)
    return splitted_data_info_list
