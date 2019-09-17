# RNN-IMP

## OVERVIEW

RNN-IMP is a recurrent neural network based genotype imputation program implemented in Python. RNN-IMP takes phased genotypes in HAPLEGEND format as input data and outputs imputation results in Oxford GEN format.

## REQUIREMENT

- Python 3.5 or Python 3.6
- Python packages
  - NumPy
  - TensorFlow (versions 1.11 - 1.15)

Please set path to python3

## INPUT

- Phased genotype data in HAPLEGEND format
- Trained model files (see usage part for details)

## OUTPUT

Genotype imputation results in Oxford GEN format

## EXAMPLE USAGE

In `example.sh`, imputation of genotype data in `example_data/hap/chr22.hap.gz` by the following command is considered.

~~~~
python3 scripts/imputation_all.py \  
    --hap example_data/hap/chr22.hap.gz \
    --legend example_data/hap/chr22.legend.gz \
    --model-file-list example_data/model_data/model_files.txt \
    --output-prefix results/chr22
  
~~~~

Input HAP file `chr22.hap.gz` contains phased genotype data of 100 individuals for chromosome 22 obtained from 1000 Genomes Project phase3 integrated data set (http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/). Only the marker sites designed in Infinium Omni2.5-8 BeadChip (https://www.illumina.com/products/by-type/microarray-kits/infinium-omni25-8.html) are contained in the data. Those 100 individuals were selected randomly from 2,504 individuals comprising the phase3 integrated data set. (Sorry, we are now preparing an instruction and a script for example data)

Since RNN-IMP cannot be applied to the entire chromosome currently, imputaion is computed in small regions seperately, and the result of each region is merged as a final output. `model_files.txt` given to `--model-file-list` option is a text file in which path of files required for each region is specified at each row. Each region requires the following four types files:

- model file
  - TensorFlow model parameter file
- panel legend file
  - LEGEND file for marker and imputation sites that contains position and allele information as well as a1 allele frequency and binary flag denoting maker site by one and imputation site by zero.
- config file 1
  - JSON file for structure information of model 1
- config file 2
  - JSON file for structure information of model 2

The example data set contains 20 regions for chr22:1-18978417. Model parameters for these regions were trained using phased genotype data for remaining 2,404 individuals.

## OPTIONS

| Option | Value Type | Default | Summary |
|--------|:-----------|:--------|:--------|
| --hap | STRING | - | input HAP file |
| --legend | STRING | - | input LEGEND file |
| --model-files | STRING | - | model file list file |
| --output-prefix | STRING | - | output file prefix |
| --num-threads | INT | 1 |  number of threads |
| --qsub | BOOLEAN | false | use Univa Grid Engine for computation (this option only works for the environment where Univa Grid Engine is available) |
| --job-name | STRING | imputation | job name prefix for Univa Grid Engine job |
| --memory-size | STRING | 10G |  memory size for Univa Grid Engine job |

## LICENSE

RNN-IMP is free for academic use only. For commercial use please contact both of the following e-mail addresses:

kojima [AT] megabank [DOT] tohoku [DOT] ac [DOT] jp

kengo [AT] ecei [DOT] tohoku [DOT] ac [DOT] jp

## DISLAIMER

RNN-IMP (THE "SOFTWARE") IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## CONTACT

Developer: Kaname Kojima

kojima [AT] megabank [DOT] tohoku [DOT] ac [DOT] jp
