from contextlib import contextmanager
import gzip
import os
import pathlib
import re
import shutil
import subprocess
import sys

from pyfaidx import Fasta


def mkdir(dirname):
    os.makedirs(pathlib.Path(dirname).resolve(), exist_ok=True)


def system(command):
    subprocess.call(command, shell=True)


@contextmanager
def reading(filename):
    root, ext = os.path.splitext(filename)
    fp = gzip.open(filename, 'rt') if ext == '.gz' else open(filename, 'rt')
    try:
        yield fp
    finally:
        fp.close()


@contextmanager
def writing(filename):
    root, ext = os.path.splitext(filename)
    fp = gzip.open(filename, 'wt') if ext == '.gz' else open(filename, 'wt')
    try:
        yield fp
    finally:
        fp.close()


def complement_base(base):
    if base == 'A':
        return 'T'
    if base == 'T':
        return 'A'
    if base == 'G':
        return 'C'
    if base == 'C':
        return 'G'
    return 'N'


def align(ref_seq, seq):
    for i in range(len(ref_seq) - len(seq)):
        mismatch_flag = False
        for j in range(len(seq)):
            if ref_seq[i + j] != seq[j]:
                mismatch_flag = True
                break
        if not mismatch_flag:
            return i
    return -1


def parse_manifest(manifest_file, chr_num, hg19_fasta_file):
    assert os.path.exists(manifest_file)
    assert type(chr_num) == str
    reference = Fasta(hg19_fasta_file)['chr' + chr_num]
    marker_dict = {}
    with open(manifest_file, 'rt') as fin:
        for line in fin:
            if line.startswith('[Assay]'):
                break
        items = fin.readline().strip().split(',')
        IlmnStrand_col = items.index('IlmnStrand')
        Chr_col = items.index('Chr')
        MapInfo_col = items.index('MapInfo')
        SNP_col = items.index('SNP')
        SourceStrand_col = items.index('SourceStrand')
        SourceSeq_col = items.index('SourceSeq')
        RefStrand_col = items.index('RefStrand')
        for line in fin:
            if line.startswith('[Controls]'):
                break
            items = line.strip().split(',')
            if items[Chr_col] != chr_num:
                continue
            position = items[MapInfo_col]
            alleles = items[SNP_col][1:-1].split('/')
            if alleles[0] == 'I' or alleles[0] == 'D':
                source_seq = items[SourceSeq_col]
                upstream_seq, a0, a1, downstream_seq = re.split(
                    '[\[\/\]]', source_seq)
                upstream_seq = upstream_seq.upper()
                downstream_seq = downstream_seq.upper()
                if a0 == '-':
                    a0 = ''
                if a1 == '-':
                    a1 = ''
                a0_seq = upstream_seq + a0 + downstream_seq
                a1_seq = upstream_seq + a1 + downstream_seq
                margin = 10
                region_start = int(position) - len(upstream_seq) - margin
                region_end = (
                    region_start + max(len(a0_seq), len(a1_seq)) + 2 * margin
                )
                local_sequence = reference[region_start - 1:region_end - 1]
                a0_align = align(local_sequence, a0_seq)
                a1_align = align(local_sequence, a1_seq)
                indel_position = (
                    max(a0_align, a1_align) + len(upstream_seq)
                    + region_start - 1
                )
                alleles[0] = upstream_seq[-1] + a0
                alleles[1] = upstream_seq[-1] + a1
                position = str(indel_position)
            else:
                if items[RefStrand_col] == '-':
                    alleles[0] = complement_base(alleles[0])
                    alleles[1] = complement_base(alleles[1])
            if position not in marker_dict:
                marker_dict[position] = []
            marker_dict[position].append(alleles)
    return marker_dict


def set_marker_flags(legend_file, marker_dict, output_file):
    mkdir(os.path.dirname(output_file))
    with reading(legend_file) as fin, \
         writing(output_file) as fout:
        fout.write(fin.readline().rstrip() + ' array_marker_flag\n')
        for line in fin:
            items = line.rstrip().split()
            position = items[1]
            flag = '0'
            if position in marker_dict:
                a0 = items[2]
                a1 = items[3]
                for alleles in marker_dict[position]:
                    if alleles[0] == a0 and alleles[1] == a1:
                        flag = '1'
                        break
                    if alleles[1] == a0 and alleles[0] == a1:
                        flag = '1'
                        break
            fout.write('{:s} {:s}\n'.format(line.rstrip(), flag))


def vcf2haplegend(vcf_file, output_prefix):

    def is_missing_genotype(genotype):
        if genotype == '.':
            return True
        if genotype == './.':
            return True
        if genotype == '.|.':
            return True
        return False

    mkdir(os.path.dirname(output_prefix))
    with reading(vcf_file) as fin, \
         writing(output_prefix + '.hap.gz') as fout_hap, \
         writing(output_prefix + '.legend.gz') as fout_legend:
        for line in fin:
            if line.startswith('#CHROM'):
                sample_list = line.rstrip().split()[9:]
                break
        sample_size = len(sample_list)
        fout_legend.write('id position a0 a1\n')
        allele_id_list = [None] * (2 * sample_size)
        hap_record = [None] * (2 * sample_size)
        for line in fin:
            items = line.rstrip().split()
            chrom = items[0]
            position = items[1]
            snp = items[2]
            ref = items[3]
            alts = items[4].split(',')
            genotypes = items[9:]
            for i, genotype in enumerate(genotypes):
                if is_missing_genotype(genotype):
                    allele_id_list[2 * i] = None
                    allele_id_list[2 * i + 1] = None
                else:
                    allele_id_pair = map(int, genotype.split('|'))
                    for j, allele_id in enumerate(allele_id_pair):
                        allele_id_list[2 * i + j] = allele_id
            for alt_allele_id, alt in enumerate(alts, start=1):
                if alt == '.' or alt.startswith('<'):
                    continue
                for i, allele_id in enumerate(allele_id_list):
                    if allele_id is None:
                        hap_record[i] = '?'
                    elif allele_id == alt_allele_id:
                        hap_record[i] = '1'
                    else:
                        hap_record[i] = '0'
                fout_hap.write(' '.join(hap_record))
                fout_hap.write('\n')
                snp_id = snp
                if snp_id == '.':
                    snp_id = ':'.join([chrom, position, ref, alt])
                elif len(alts) >= 2:
                    snp_id = ':'.join([snp, ref, alt])
                fout_legend.write('{:s} {:s} {:s} {:s}\n'.format(
                    snp_id, position, ref, alt))

    with open(output_prefix + '.sample', 'wt') as fout:
        fout.write('ID_1 ID_2 missing\n')
        fout.write('0 0 0\n')
        for sample in sample_list:
            fout.write('{0:s} {0:s} 0\n'.format(sample))


def prepare_train_hap(
        hap_file,
        legend_file,
        sample_file,
        test_sample_name_set,
        output_prefix):
    col_list = []
    mkdir(os.path.dirname(output_prefix))
    with open(sample_file, 'rt') as fin, \
         open(output_prefix + '.sample', 'wt') as fout:
        fout.write(fin.readline())
        fout.write(fin.readline())
        for i, line in enumerate(fin):
            sample_name, *_ = line.rstrip().split()
            if sample_name not in test_sample_name_set:
                fout.write(line)
                col_list.append(2 * i)
                col_list.append(2 * i + 1)
    shutil.copy(legend_file, output_prefix + '.legend.gz')
    with reading(hap_file) as fin, \
         writing(output_prefix + '.hap.gz') as fout:
        for line in fin:
            items = line.rstrip().split(' ')
            fout.write(' '.join([items[col] for col in col_list]))
            fout.write('\n')


def prepare_true_test_hap(
        hap_file,
        legend_file,
        sample_file,
        test_sample_name_set,
        output_prefix):
    col_list = []
    mkdir(os.path.dirname(output_prefix))
    with open(sample_file, 'rt') as fin, \
         open(output_prefix + '.sample', 'wt') as fout:
        fout.write(fin.readline())
        fout.write(fin.readline())
        for i, line in enumerate(fin):
            sample_name, *_ = line.rstrip().split()
            if sample_name in test_sample_name_set:
                fout.write(line)
                col_list.append(2 * i)
                col_list.append(2 * i + 1)
    a1_freq_list = []
    with reading(hap_file) as fin, \
         writing(output_prefix + '.hap.gz') as fout:
        for line in fin:
            items = line.rstrip().split(' ')
            a0_count = items.count('0')
            a1_count = items.count('1')
            items = [items[col] for col in col_list]
            a0_count -= items.count('0')
            a1_count -= items.count('1')
            if a0_count + a1_count == 0:
                a1_freq_list.append('NA')
            else:
                a1_freq = a1_count / (a0_count + a1_count)
                a1_freq_list.append(str(a1_freq))
            fout.write(' '.join(items))
            fout.write('\n')
    with reading(legend_file) as fin, \
         writing(output_prefix + '.legend.gz') as fout:
        header_items = fin.readline().rstrip().split(' ')
        if 'a1_freq' not in header_items:
            header_items.append('a1_freq')
        a1_freq_col = header_items.index('a1_freq')
        fout.write(' '.join(header_items))
        fout.write('\n')
        for i, line in enumerate(fin):
            items = line.rstrip().split(' ')
            if a1_freq_col == len(items):
                items.append(a1_freq_list[i])
            else:
                items[a1_freq_col] = a1_freq_list[i]
            fout.write(' '.join(items))
            fout.write('\n')


def prepare_test_hap(hap_file, legend_file, output_prefix):
    array_marker_flag_list = []
    with reading(legend_file) as fin, \
         writing(output_prefix + '.legend.gz') as fout:
        line = fin.readline()
        items = line.rstrip().split()
        try:
            array_marker_flag_col = items.index('array_marker_flag')
        except ValueError:
            print_error()
            print_error('Error: Header "array_marker_flag" not found in '
                        + legend_file)
            print_error()
            sys.exit(0)
        fout.write(line)
        for line in fin:
            items = line.rstrip().split()
            array_marker_flag = items[array_marker_flag_col] == '1'
            array_marker_flag_list.append(array_marker_flag)
            if array_marker_flag:
                fout.write(line)
    with reading(hap_file) as fin, \
         writing(output_prefix + '.hap.gz') as fout:
        for i, line in enumerate(fin):
            if array_marker_flag_list[i]:
                fout.write(line)


def main():
    test_sample_list_file = 'org_data/test_samples.txt'
    vcf_file = (
        'org_data/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.'
        '20130502.genotypes.vcf.gz')
    manifest_file = 'org_data/InfiniumOmni2-5-8v1-4_A1.csv'
    hg19_fasta_file = 'org_data/hg19.fa'
    output_dir = 'example_data'
    output_basename = 'chr22'

    all_hap_prefix = os.path.join(output_dir, 'all', output_basename)
    vcf2haplegend(vcf_file, all_hap_prefix)
    marker_dict = parse_manifest(manifest_file, '22', hg19_fasta_file)
    set_marker_flags(
        all_hap_prefix + '.legend.gz', marker_dict,
        all_hap_prefix + '.Omni2.5.legend.gz')
    test_sample_name_set = set()
    with open(test_sample_list_file, 'rt') as fin:
        for line in fin:
            sample_name = line.rstrip()
            test_sample_name_set.add(sample_name)

    train_hap_prefix = os.path.join(
        output_dir, 'train', output_basename)
    test_hap_prefix = os.path.join(
        output_dir, 'test', output_basename)

    prepare_train_hap(
        all_hap_prefix + '.hap.gz',
        all_hap_prefix + '.Omni2.5.legend.gz',
        all_hap_prefix + '.sample',
        test_sample_name_set,
        train_hap_prefix)
    prepare_true_test_hap(
        all_hap_prefix + '.hap.gz',
        all_hap_prefix + '.Omni2.5.legend.gz',
        all_hap_prefix + '.sample',
        test_sample_name_set,
        test_hap_prefix + '_true')
    prepare_test_hap(
        test_hap_prefix + '_true.hap.gz',
        test_hap_prefix + '_true.legend.gz',
        test_hap_prefix)


if __name__ == '__main__':
    main()
