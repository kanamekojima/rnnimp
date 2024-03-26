import os

import numpy as np

from common_utils import mkdir, reading, writing


def get_genotype_probs(allele_probs):
    genotype_probs0 = allele_probs[:, 0, 0] * allele_probs[:, 1, 0]
    genotype_probs2 = allele_probs[:, 0, 1] * allele_probs[:, 1, 1]
    return np.stack([
        genotype_probs0,
        1 - genotype_probs0 - genotype_probs2,
        genotype_probs2,
    ], axis=1)


def hapslegend2gen(hap_file, legend_file, output_file):

    def to_genotype_prob(allele1, allele2):
        if allele1 == 'NA' or allele2 == 'NA':
            return '0.25 0.5 0.25'
        if allele1 == '0' and allele2 == '0':
            return '1 0 0'
        if allele1 == '0' and allele2 == '1':
            return '0 1 0'
        if allele1 == '1' and allele2 == '0':
            return '0 1 0'
        return '0 0 1'

    with reading(hap_file) as fin_hap, \
         reading(legend_file) as fin_legend, \
         writing(output_file) as fout:
        header_items = fin_legend.readline().rstrip().split(' ')
        try:
            id_col = header_items.index('id')
            a0_col = header_items.index('a0')
            a1_col = header_items.index('a1')
            position_col = header_items.index('position')
        except ValueError:
            print(
                'Some of header items not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin_hap:
            l_items = fin_legend.readline().rstrip().split(' ')
            a0 = l_items[a0_col]
            a1 = l_items[a1_col]
            snp_name = l_items[id_col]
            position = l_items[position_col]
            items = line.strip().split()
            sample_size = int(len(items) / 2)
            fout.write(
                '--- {:s} {:s} {:s} {:s}'.format(snp_name, position, a0, a1))
            for i in range(sample_size):
                fout.write(' ')
                fout.write(to_genotype_prob(*items[2 * i: 2 * i + 2]))
            fout.write('\n')


def merge(gen_file1, gen_file2, output_file):

    def get_position(line):
        space_count = 0
        previous_space_position = -1
        for i, c in enumerate(line):
            if c == ' ':
                space_count += 1
                if space_count == 3:
                    return int(line[previous_space_position + 1 : i])
                previous_space_position = i
        return int(line[previous_space_position + 1 :])

    with reading(gen_file1) as fin1, \
         reading(gen_file2) as fin2, \
         writing(output_file) as fout:
        position2 = -1
        line2 = None
        for line1 in fin1:
            position1 = get_position(line1)
            if position1 > position2:
                if line2 is not None:
                    fout.write(line2)
                    line2 = None
                for line2 in fin2:
                    position2 = get_position(line2)
                    if position1 <= position2:
                        break
                    fout.write(line2)
                    line2 = None
            fout.write(line1)
        if line2 is not None:
            fout.write(line2)
        for line2 in fin2:
            fout.write(line2)


def write_gen(predictions, legend_record_list, output_file):
    mkdir(os.path.dirname(output_file))
    num_samples = predictions.shape[1] // 2
    with writing(output_file) as fout:
        for i, allele_probs in enumerate(predictions):
            legend_record = legend_record_list[i]
            line = '--- {:s} {:s} {:s} {:s}'.format(
                legend_record.id, legend_record.position, legend_record.a0,
                legend_record.a1)
            fout.write(line)
            if allele_probs.shape[0] == 0:
                fout.write('\n')
                continue
            fout.write(' ')
            allele_probs = allele_probs.reshape(-1, 2, 2).astype(np.float64)
            genotype_probs = get_genotype_probs(allele_probs).reshape(-1)
            fout.write(' '.join(map(str, genotype_probs)))
            fout.write('\n')
