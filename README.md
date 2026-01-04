# A Deep Learning–Based Computational Tool for Automatic Trimming of Transposable Elements in Large-Scale Genomic Projects

First of all, create 3 environments according to requirements: auto_trimming_agp, dataset_agp and teaid_agp.


dataset_library.py contains functions to generate PDFs and images from an fasta file. 
headers need to include the species at the end, separated by a space from the rest of the header. In this way, the species will be recognized and its genome will be downloaded, which is need to run TEAid. 


## Generation of simulated data
For data generation, headers need to include the species at the end, separated by a space from the rest of the header. 
This will generate sequences with one or two TEs combined with randomly generated DNA sequences of a total length of 15000 bp. This script does not require an environment in particular.
Example:

>DR000395818#CLASSI/LINE/CR1 Bucorvus abyssinicus
>
From this file, that can be, for example, a curated library, we will run the script:

sbatch data_generation/run_generation.sh
### executes data_generation/GenerationData.py

### Parameters
--fasta: Path to FASTA file from which simulated data will be generated (required)
--seq_per_case: number of sequences to generate per case (4 cases)

## Create images from FASTA file with TE-Aid
This script requires the environment teaid_agp. 
```
sbatch run_teaid.sh
```
### executes Auto_trimming.py with --mode teaid

### Parameters
--input_fasta: Path to FASTA file from which the images will be created (required)

Requires environment teaid_agp

The following scripts are needed: TE-Aid, Run-c2g.R, consensus2genome.R and blastndotplot.R (by default, they will be included in the folder TEAid).

### Output
It will create a folder called te_aid, in which the PDF files and images will be generated. 

## Create a dataset
This will generate 4 matrices: features_data.npy, labels_data.npy

```
sbatch create_dataset.sh
```
### Parameters
--input_fasta: Path to FASTA file from which the dataset will be created (required)
--dataset_dir: Directory where the dataset will be saved

After generating the PDFs (and images) and the dataset, we can choose in the script Auto_trimming.py if we want to do training, testing or trimming. 

## Trimming
Previously, we will need to generate the dataset from the FASTA files with the sequences that we want to trim. This will generate a .txt file where the trimmed sequences will be saved with the original header name (until # symbol) and indicates the positions in which the sequence was cut.

### Parameters
--input_fasta: Path to FASTA file from which the dataset was created (required)
--dataset_dir: Directory where the dataset is saved
--model:
--scaler

## Training

### Parameters
--dataset_dir: Directory where the dataset is saved

## Testing

# Parameters
--dataset_dir: Directory where the dataset is saved
--model:
--scaler

