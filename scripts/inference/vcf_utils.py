import datetime
import os

import numpy as np

from common_utils  import mkdir, reading, writing
from imputation_scores import calculate_INFO, calculate_Minimac_R2


VCF_SEP = '\t'
VCF_IMP_INFO_TEMPLATE = (
    'IMP;AC={:d};AN={:d};AF={:.5f};MAF={:.5f};R2={:.5f};INFO={:.5f}')
VCF_INFO_TEMPLATE = (
    'AC={:d};AN={:d};AF={:.5f};MAF={:.5f};R2={:.5f};INFO={:.5f}')
SOURCE = 'RNN-IMP'


def get_genotype_probs(allele_probs):
    genotype_probs0 = allele_probs[:, 0, 0] * allele_probs[:, 1, 0]
    genotype_probs2 = allele_probs[:, 0, 1] * allele_probs[:, 1, 1]
    return np.stack([
        genotype_probs0,
        1 - genotype_probs0 - genotype_probs2,
        genotype_probs2,
    ], axis=1)


def hapslegend2vcf(hap_file, legend_file, output_file):

    def to_GT_DS_GP(allele1, allele2):
        if allele1 == 'NA' or allele2 == 'NA':
            return '.'
        if allele1 == '0' and allele2 == '0':
            return '0|0:0.000:1.000,0.000,0.000'
        if allele1 == '0' and allele2 == '1':
            return '0|1:0.500:0.000,1.000,0.000'
        if allele1 == '1' and allele2 == '0':
            return '1|0:0.500:0.000,1.000,0.000'
        return '1|1:1.000:0.000,0.000,1.000'

    mkdir(os.path.dirname(output_file))
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
            non_NA_haps = np.array(
                list(map(int, filter(lambda x: x != 'NA', items))))
            AC = non_NA_haps.sum()
            AN = len(non_NA_haps)
            AF = non_NA_haps.mean()
            MAF = min(AF, 1 - AF)
            R2 = calculate_Minimac_R2(non_NA_haps, AF)
            INFO = 1
            fout.write(VCF_SEP.join([position, snp_name, a0, a1, '.', 'PASS']))
            fout.write(VCF_SEP)
            fout.write(VCF_INFO_TEMPLATE.format(AC, AN, AF, MAF, R2, INFO))
            fout.write(VCF_SEP)
            fout.write('GT:DS:GP')
            for i in range(sample_size):
                fout.write(VCF_SEP)
                fout.write(to_GT_DS_GP(*items[2 * i: 2 * i + 2]))
            fout.write('\n')


def get_vcf_headers(chromosome, sample_list):
    date = datetime.date.today()
    header_lines = [
        '##fileformat=VCFv4.2',
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##filedate={:d}.{:d}.{:d}'.format(date.year, date.month, date.day),
        '##source=' + SOURCE,
        '##contig=<ID={:s}>'.format(chromosome),
        '##INFO=<ID=IMP,Number=0,Type=Flag,Description="Imputed marker">',
        '##INFO=<ID=AC,Number=A,Type=Integer,Description="ALT allele count from GT across target samples">',
        '##INFO=<ID=AN,Number=A,Type=Integer,Description="Total number of alleles in called genotypes">',
        '##INFO=<ID=AF,Number=A,Type=Float,Description="ALT allele frequency computed from DS/GP field across target samples">',
        '##INFO=<ID=MAF,Number=1,Type=Float,Description="Estimated Minor Allele Frequency">',
        '##INFO=<ID=R2,Number=1,Type=Float,Description="Estimated Imputation Accuracy">',
        '##INFO=<ID=INFO,Number=A,Type=Float,Description="IMPUTE info quality score">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotypes">',
        '##FORMAT=<ID=DS,Number=A,Type=Float,Description="Genotype dosage">',
        '##FORMAT=<ID=GP,Number=G,Type=Float,Description="Genotype posterior probabilities">',
    ]
    header_items = [
        'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT',
    ] + sample_list
    header_lines.append('#' + VCF_SEP.join(header_items))
    return header_lines


def merge(vcf_file1, vcf_file2, chromosome, sample_file, output_file):

    def get_position(line):
        for i, c in enumerate(line):
            if c == VCF_SEP:
                return int(line[:i])
        return int(line.rstrip())

    with reading(vcf_file1) as fin:
        items = fin.readline().split(VCF_SEP)
        num_samples = len(items) - 8
    if sample_file is None:
        sample_list = ['Sample_{:d}'.format(i + 1) for i in range(num_samples)]
    else:
        sample_list = []
        with open(sample_file, 'rt') as fin:
            fin.readline()
            fin.readline()
            for line in fin:
                sample_ID1, sample_ID2, *_ = line.rstrip().split(' ')
                sample_list.append(sample_ID1)
        assert len(sample_list) == num_samples
    with reading(vcf_file1) as fin1, \
         reading(vcf_file2) as fin2, \
         writing(output_file) as fout:
        for header_line in get_vcf_headers(chromosome, sample_list):
            fout.write(header_line)
            fout.write('\n')
        position2 = -1
        line2 = None
        for line1 in fin1:
            position1 = get_position(line1)
            if position1 > position2:
                if line2 is not None:
                    fout.write(chromosome)
                    fout.write(VCF_SEP)
                    fout.write(line2)
                    line2 = None
                for line2 in fin2:
                    position2 = get_position(line2)
                    if position1 <= position2:
                        break
                    fout.write(chromosome)
                    fout.write(VCF_SEP)
                    fout.write(line2)
                    line2 = None
            fout.write(chromosome)
            fout.write(VCF_SEP)
            fout.write(line1)
        if line2 is not None:
            fout.write(chromosome)
            fout.write(VCF_SEP)
            fout.write(line2)
        for line2 in fin2:
            fout.write(chromosome)
            fout.write(VCF_SEP)
            fout.write(line2)


def write_vcf(predictions, legend_record_list, output_file):
    mkdir(os.path.dirname(output_file))
    with writing(output_file) as fout:
        for i, allele_probs in enumerate(predictions):
            legend_record = legend_record_list[i]
            a1_probs = allele_probs[:, 1]
            alleles = (a1_probs >= 0.5).astype(np.int32)
            AC = alleles.sum()
            AN = a1_probs.shape[0]
            AF = a1_probs.mean()
            MAF = min(AF, 1 - AF)
            R2 = calculate_Minimac_R2(a1_probs)
            allele_probs = allele_probs.reshape(-1, 2, 2).astype(np.float64)
            genotype_probs = get_genotype_probs(allele_probs)
            dosages = allele_probs[:, :, 1].sum(axis=1)
            INFO = calculate_INFO(genotype_probs, dosages, AF)
            fout.write(VCF_SEP.join([
                legend_record.position,
                legend_record.id,
                legend_record.a0,
                legend_record.a1,
                '.',
                'PASS',
            ]))
            fout.write(VCF_SEP)
            fout.write(VCF_IMP_INFO_TEMPLATE.format(AC, AN, AF, MAF, R2, INFO))
            fout.write(VCF_SEP)
            fout.write('GT:DS:GP')
            alleles = alleles.reshape(-1, 2)
            for i in range(alleles.shape[0]):
                fout.write(VCF_SEP)
                fout.write('{:d}|{:d}:{:.5f}:{:.5f},{:.5f},{:.5f}'.format(
                    *alleles[i], dosages[i], *genotype_probs[i]))
            fout.write('\n')
