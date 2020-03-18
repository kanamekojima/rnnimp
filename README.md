# RNN-IMP

## OVERVIEW

RNN-IMP is a recurrent neural network based genotype imputation program implemented in Python. RNN-IMP takes phased genotypes in HAP/LEGEND format as input data and outputs imputation results in Oxford GEN format.

## REQUIREMENT

- Python 3.5 or Python 3.6 (please set the path in python3)
- Python packages
  - NumPy
  - TensorFlow (versions 1.11 - 1.15)

## INPUT

- Phased genotype data in HAP/LEGEND format
- Trained model files (for details, please see EXAMPLE USAGE)

## OUTPUT

Genotype imputation results in Oxford GEN format

## EXAMPLE USAGE

### Preparation of Example Dataset for Imputation

The example dataset in the uploaded file does not contain the genotype data files for test samples, `chr22.hap.gz` and `chr22.legend.gz`.
The following three files and a software program are required for the preparation of `chr22.hap.gz` and `chr22.legend.gz`:
- A VCF file of 1000 Genome Project (1KGP) phase3 integrated dataset for chromosome22 (`ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz`)
  - Download the VCF file from the following ftp site and put it to `org data` directory:
    - [http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502])
- A manifest file for Infinium Omni2.5-8 BeadChip (`InfiniumOmni2-5-8v1-4_A1.csv`)
  - Download `infinium-omni-2-5-8v1-4-a1-manifest-file-csv.zip` from the following web site:
    - [https://support.illumina.com/array/array_kits/humanomni2_5-8_beadchip_kit/downloads.html](https://support.illumina.com/array/array_kits/humanomni2_5-8_beadchip_kit/downloads.html)
  - Decompress the downloaded file and copy `InfiniumOmni2-5-8v1-4_A1.csv` to `org data` directory
- Hg19 fasta file (`hg19.fa`)
  - Download `hg19.fa.gz` from the following web site:
    - [https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/)
  - Decompress the downloaded file and copy `hg19.fa` to `org data` directory
- SAMtools
  - See [http://samtools.sourceforge.net/](http://samtools.sourceforge.net/) for the installation of SAMtools
  - Set the path in `samtools` after the installation

After the preparation of the above software program and files, execute the following command to generate `chr22.hap.gz` and `chr22.legend.gz`, in `example_data/hap` directory:

```sh
python3 scripts/test_data_preparation.py
```

`chr22.hap.gz` contains phased genotype data of 100 individuals for chromosome 22 obtained from 1KGP phase3 integrated dataset. Genotype data in `chr22.hap.gz` are only for the marker sites designed in Infinium Omni2.5-8 BeadChip. 100 individuals for `chr22.hap.gz` were selected randomly from 2,504 individuals comprising the phase3 integrated dataset. Sample names of the 100 individuals are in `example_data/hap/test_samples.txt`.

### Imputation for Example Dataset

Imputation with RNN-IMP using pre-trained model parameters for `chr22.hap.gz` in `example_data/hap` directory starts by the following command, which is also described in `example.sh`:

```sh
python3 scripts/imputation_all.py \
    --hap example_data/hap/chr22.hap.gz \
    --legend example_data/hap/chr22.legend.gz \
    --model-file-list example_data/model_data/model_files.txt \
    --output-prefix results/chr22
```

RNN-IMP computes imputation for small regions separately, and the final output in Oxford GEN format is obtained by merging the results from the small regions. `--model-file-list` option specifies a text file that contains the paths of the files required for each region at each row. Each region requires the following four files:

- Model file
  - TensorFlow model parameter file
- Panel legend file
  - LEGEND file for marker sites and imputation target sites. Each record of the file contains physical position, allele information, a1 allele frequency, and binary flag that takes one for SNP array marker sites and zero for imputation target sites.
- Config file 1
  - JSON file for the structure information of model 1
- Config file 2
  - JSON file for the structure information of model 2

The files for each region are in `example_data/model_data` directory of this example dataset. Note that this example dataset contains the files for 10 regions, which cover only a part of chromosome 22 (chr22:1-17885697), due to the limitation of the file upload size. The files for all the regions, which cover whole chromosome 22, are available in [https://jmorp.megabank.tohoku.ac.jp/dj1-storage/code/rnn-imp/rnnimp_model_data_chr22.tar.gz](https://jmorp.megabank.tohoku.ac.jp/dj1-storage/code/rnn-imp/rnnimp_model_data_chr22.tar.gzh). Model parameters in this example dataset were trained using phased genotype data for 2,404 individuals who are not in `chr22.hap.gz` from 1KGP phase3 integrated dataset.

## OPTIONS

| Option | Default value | Summary |
|:-------|--------------:|:--------|
| --hap STRING_VALUE | - | input HAP file |
| --legend STRING_VALUE | - | input LEGEND file |
| --model-files STRING_VALUE | - | model file list file |
| --output-prefix STRING_VALUE | - | output file prefix |
| --num-threads INT_VALUE | 1 | number of threads |
| --qsub | False | use Univa Grid Engine for computation (this option only works for the environment where Univa Grid Engine is available) |
| --job-name STRING_VALUE | imputation | job name prefix for Univa Grid Engine job |
| --memory-size STRING_VALUE | 10GB | memory size for Univa Grid Engine job |

## LICENSE

Scripts in this repository are licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## CONTACT

Developer: Kaname Kojima, Ph.D.

E-mail: kojima [AT] megabank [DOT] tohoku [DOT] ac [DOT] jp or kengo [AT] ecei [DOT] tohoku [DOT] ac [DOT] jp
