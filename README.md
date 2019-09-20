# RNN-IMP

## OVERVIEW

RNN-IMP is a recurrent neural network based genotype imputation program implemented in Python. RNN-IMP takes phased genotypes in HAPLEGEND format as input data and outputs imputation results in Oxford GEN format.

## REQUIREMENT

- Python 3.5 or Python 3.6
- Python packages
  - NumPy
  - TensorFlow (versions 1.11 - 1.15)

Also, please set path to python3.

## INPUT

- Phased genotype data in HAPLEGEND format
- Trained model files (for details, please see EXAMPLE USAGE)

## OUTPUT

Genotype imputation results in Oxford GEN format

## EXAMPLE USAGE

### Preparation of Example Data for Imputation

Download a VCF file of 1000 Genome Project (1KGP) phase3 integrated data set for chromosome22 (ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz) from the following ftp site:

- http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502

Also download a manifest file for Infinium Omni2.5-8 BeadChip (infinium-omni-2-5-8v1-4-a1-manifest-file-csv.zip) from the following web site:

- https://support.illumina.com/array/array_kits/humanomni2_5-8_beadchip_kit/downloads.html

By unzipping the manifest file, you will get a CSV file (InfiniumOmni2-5-8v1-4_A1.csv).
Put the VCF file and the CSV file in `org_data` directory. Then, from the following command, `chr22.hap.gz` and `chr22.legend.gz` are created in `example_data/hap` directory:

~~~~
python3 scripts/test_data_preparation.py
~~~~

`chr22.hap.gz` contains phased genotype data of 100 individuals for chromosome 22 obtained from 1KGP phase3 integrated data set. Genotype data in `chr22.hap.gz` are only for the marker sites designed in Infinium Omni2.5-8 BeadChip. 100 individuals for `chr22.hap.gz` were selected randomly from 2,504 individuals comprising the phase3 integrated data set. Sample names of the 100 individuals are in `example_data/hap/test_samples.txt`.

### Imputation of Example Data

Imputation with RNN-IMP using pre-trained model parameters for `chr22.hap.gz` in `example_data/hap` directory starts by the following command, which is also described in `example.sh`:

~~~~
python3 scripts/imputation_all.py \
    --hap example_data/hap/chr22.hap.gz \
    --legend example_data/hap/chr22.legend.gz \
    --model-file-list example_data/model_data/model_files.txt \
    --output-prefix results/chr22
~~~~

Since RNN-IMP cannot be applied to the whole chromosome currently, imputation is computed in small regions separately, and the results from the small regions are merged as a final output in Oxford GEN format. `model_files.txt` given to `--model-file-list` option is a text file in which path of files required for each region is specified at each row. Each region requires the following four types files:

- model file
  - TensorFlow model parameter file
- panel legend file
  - LEGEND file for marker sites and imputation target sites. Each record of the file contains position and allele information as well as a1 allele frequency and binary flag that takes one for maker site and zero for imputation target site.
- config file 1
  - JSON file for structure information of model 1
- config file 2
  - JSON file for structure information of model 2

The example data set contains 10 regions for chr22:1-17885697 (Due to the limitation of the file size, trained data only for the 10 regions are currently uploaded). Model parameters for these regions were trained using phased genotype data for 2,404 individuals not in `chr22.hap.gz` from 1KGP phase3 integrated data set.

## OPTIONS

| Option | Value Type | Default | Summary |
|--------|:-----------|:--------|:--------|
| --hap | STRING | - | input HAP file |
| --legend | STRING | - | input LEGEND file |
| --model-files | STRING | - | model file list file |
| --output-prefix | STRING | - | output file prefix |
| --num-threads | INT | 1 | number of threads |
| --qsub | BOOLEAN | false | use Univa Grid Engine for computation (this option only works for the environment where Univa Grid Engine is available) |
| --job-name | STRING | imputation | job name prefix for Univa Grid Engine job |
| --memory-size | STRING | 10G | memory size for Univa Grid Engine job |

## LICENSE

RNN-IMP is free for academic use only. For commercial use please contact both of the following e-mail addresses:

kojima [AT] megabank [DOT] tohoku [DOT] ac [DOT] jp

kengo [AT] ecei [DOT] tohoku [DOT] ac [DOT] jp

## DISLAIMER

RNN-IMP (THE "SOFTWARE") IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## CONTACT

Developer: Kaname Kojima

kojima [AT] megabank [DOT] tohoku [DOT] ac [DOT] jp
