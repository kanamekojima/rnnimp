# RNN-IMP

RNN-IMP is a Python program for reference-free genotype imputation using recurrent neural networks (RNNs).
RNN-IMP takes phased genotypes in HAPSLEGEND format as input and outputs imputation results in either VCF or Oxford GEN format.

## Installation

Requirements: Python versions 3.5 to 3.10 (ensure python3 is in your path)

```sh
git clone https://github.com/kanamekojima/rnnimp.git
cd rnnimp
python3 -m pip install -r requirements.txt
```

## Example Usage

### Preparation of Example Dataset

To prepare the example dataset, the following files are required:

- A VCF file from the 1000 Genomes Project (1KGP) phase 3 dataset for chromosome 22 (`ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz`):
  - Download the VCF file from the following website and place it in the `org_data` directory:
    - [https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502)
- A manifest file for the Infinium Omni2.5-8 BeadChip (`InfiniumOmni2-5-8v1-4_A1.csv`):
  - Download `infinium-omni-2-5-8v1-4-a1-manifest-file-csv.zip` from the following website:
    - [https://support.illumina.com/array/array_kits/humanomni2_5-8_beadchip_kit/downloads.html](https://support.illumina.com/array/array_kits/humanomni2_5-8_beadchip_kit/downloads.html)
    - Unzip the file and place `InfiniumOmni2-5-8v1-4_A1.csv` in the `org_data` directory.
- An Hg19 fasta file (hg19.fa):
  - Download `hg19.fa.gz` from the following website:
    - [https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/)
    - Unzip the file and place `hg19.fa` in the `org_data` directory.

After preparing the above files, execute the following command in the `rnnimp` directory:

```sh
python3 scripts/test_data_preparation.py
```

This process generates the example dataset, including:

- `example_data/test/chr22_true.[hap.gz/legend.gz]`: Phased genotype data for 100 individuals from `org_data/test_samples.txt` for chromosome 22 in HAPSLEGEND format, derived from the 1KGP phase 3 dataset. These individuals are randomly selected from the 2,504 individuals in the 1KGP phase 3 dataset.
- `example_data/test/chr22.[hap.gz/legend.gz]`: Phased genotype data extracted from `example_data/test/chr22_true.[hap.gz/legend.gz]` for marker sites designed for the Infinium Omni2.5-8 BeadChip. This dataset simulates input data from the Omni2.5 array.
- `example_data/train/chr22.[hap.gz/legend.gz]`: Phased genotype data for the remaining 2,404 individuals not included in `org_data/test_samples.txt` for chromosome 22, obtained from the 1KGP phase 3 dataset in HAPSLEGEND format. This dataset is used for training RNN-IMP.

### Imputation for the Example Dataset

RNN-IMP performs imputation on small regions separately and combines these results to produce an imputation result for an entire chromosome in VCF format or Oxford GEN format.
For each small region, specific RNN model structures and their parameters, stored in ONNX Runtime (ORT) format, along with target information in legend format, are required.

Legend files from the example training data are located in the `results/train/models` directory, and ORT files can be obtained with the following commands in the `rnnimp` directory:

```sh
wget https://github.com/kanamekojima/rnnimp/raw/master/results/train/models/chr22_onnx_files.tbz -P org_data
tar jxf org_data/chr22_onnx_files.tbz -C results/train/models

for onnx_file in $(ls results/train/models/chr22*.onnx)
do
  python3 -m onnxruntime.tools.convert_onnx_models_to_ort $onnx_file
done
```

These files can also be generated through the training process, as described in the subsequent section.
To perform imputation on `example_data/test/chr22.[hap.gz/legend.gz]` using these model information files, execute the following command in the `rnnimp` directory:

```sh
python3 scripts/inference/imputation.py \
    --hap example_data/test/chr22.hap.gz \
    --legend example_data/test/chr22.legend.gz \
    --model-prefix results/train/models/chr22 \
    --output-prefix results/imputation/chr22
```

This command generates the imputation result for `example_data/test/chr22.[hap.gz/legend.gz]` as `results/imputation/chr22.gen`.
To produce the results in VCF format, use the `--output-format vcf` option.

### Training RNN Models for the Example Dataset

In RNN-IMP, the whole chromosome is divided into small regions, and RNN models are trained separately for each region.
To begin, divide the training data, `example_data/train/chr22.[hap.gz/legend.gz]`, into smaller segments using the following commands in the `rnnimp` directory:

```sh
wget https://raw.githubusercontent.com/stephenslab/ldshrink/main/inst/test_gdsf/fourier_ls-all.bed -P org_data
head -n 1 org_data/fourier_ls-all.bed > org_data/fourier_ls-chr22.bed
grep "^chr22 " org_data/fourier_ls-all.bed >> org_data/fourier_ls-chr22.bed
python3 scripts/train/train_data_splitter.py \
    --hap example_data/train/chr22.hap.gz \
    --legend example_data/train/chr22.legend.gz \
    --output-prefix example_data/train/split/chr22 \
    --body-marker-count-limit 200 \
    --flanking-marker-count-limit 50 \
    --imp-site-count-limit 1000 \
    --partition org_data/fourier_ls-chr22.bed
```

The `--partition` option specifies a file containing a list of regions into which the chromosome is divided, facilitating segmentation based on these predefined regions.
For this example, a list of regions segmented at high recombination rate points is used.
This list is available on the Stephens lab GitHub page as a file named `fourier_ls-all.bed`:
[https://github.com/stephenslab/ldshrink](https://github.com/stephenslab/ldshrink)

Based on the specified criteria, these segmented regions are further divided in the above commands, resulting in the generation of training data files for 225 divided regions in this example.
The prefixes for these files are listed in `example_data/train/split/chr22.list`.

To train RNN models for these regions using the segmented training data, execute the following command in the `rnnimp` directory:

```sh
python3 scripts/train/train.py \
    --data-list example_data/train/split/chr22.list \
    --rnn-cell-type GRU \
    --num-layers-higher 4 \
    --num-layers-lower 4 \
    --num-units 40 \
    --gamma1 0.75 \
    --gamma2 -0.75 \
    --feature-size 40 \
    --output-prefix results/train/chr22
```

**Warning:** Running the training command as described may require over six months to complete on a single thread, even using high-end CPUs.
To significantly reduce computation time, the use of supercomputing resources, which allow for parallel processing, is strongly recommended.
The `--slurm` option facilitates the parallel training of RNN models across different regions by leveraging supercomputing resources managed with Slurm.
Due to the variability in Slurm configurations, please refer to the `scripts/train/slurm.py` script for usage details and make necessary adjustments to fit your computing environment.
For computing environments using job schedulers other than Slurm, modifications to the `scripts/train/train.py` script will be required to enable parallel processing.

## Options

### Options for `scripts/inference/imputation.py`

| Option | Default Value | Summary |
|:-------|:-------------:|:-------|
| --hap STRING_VALUE | - | Input hap file |
| --legend STRING_VALUE | - | Input legend file |
| --sample STRING_VALUE | None | Input sample file (optional) |
| --chromosome | None | Chromosome name. Required for VCF output format. |
| --model-prefix STRING_VALUE | - | Model name prefix |
| --output-prefix STRING_VALUE | - | Output file name prefix |
| --output-format STRING_VALUE | gen | Output format [gen / vcf] |
| --python3-bin STRING_VALUE | python3 | Path to the Python3 binary |

Options for `scripts/train/train.py`

| Option | Default Value | Summary |
|:-------|:-------------:|:-------|
| --data-list STRING_VALUE | - | Input data list file |
| --output-prefix STRING_VALUE | - | Output file name prefix |
| --rnn-cell-type STRING_VALUE | GRU | RNN cell type. Available options: GRU / LSTM |
| --num-units INT_VALUE | 40 | Vector size in RNN cells |
| --num-layers-higher INT_VALUE | 4 | RNN layer size for the higher MAF model |
| --num-layers-lower INT_VALUE | 4 | RNN layer size for the lower MAF model |
| --feature-size INT_VALUE | 40 | Input feature vector size |
| --gamma1 FLOAT_VALUE | 0.75 | Loss weight parameter for the higher MAF model |
| --gamma2 FLOAT_VALUE | 0.75 | Loss weight parameter for the lower MAF model |
| --batch-size INT_VALUE | 500 | Training batch size |
| --max-iteration-count INT_VALUE | 100000 | Maximum iteration count |
| --validation-sample-size INT_VALUE | 100 | Validation sample size |
| --num-threads INT_VALUE | 1 | Number of threads in TensorFlow |
| --slurm | False | Enables the use of Slurm for distributed computation (this option is only effective in environments where Slurm is available) |
| --job-name-prefix STRING_VALUE | train | Job name prefix for Slurm jobs |
| --memory-size STRING_VALUE | 20GB | Memory size limit for Slurm jobs |
| --python3-bin STRING_VALUE | python3 | Path to the Python3 binary |

## Citation

If you find RNN-IMP or any of the scripts in this repository useful for your research, please cite:

> Kojima, K., Tadaka, S., Katsuoka, F., Tamiya, G., Yamamoto, M. & Kinoshita, K. (2020).
> A genotype imputation method for de-identified haplotype reference information by using recurrent neural network.
> *PLoS Computational Biology*, **16**(10): e1008207.
> https://doi.org/10.1371/journal.pcbi.1008207

## License

The scripts in this repository are available under the MIT License.
For more details, see the [LICENSE.md](LICENSE.md) file.

## Contact

Developer: Kaname Kojima, Ph.D.

E-mail: kojima [AT] megabank [DOT] tohoku [DOT] ac [DOT] jp or kengo [AT] ecei [DOT] tohoku [DOT] ac [DOT] jp
