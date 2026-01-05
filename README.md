# A Deep Learning–Based Computational Tool for Automatic Trimming of Transposable Elements in Large-Scale Genomic Projects

First of all, create 3 environments according to requirements: auto_trimming_agp, dataset_agp and teaid_agp.


dataset_library.py contains functions to generate PDFs and images from an fasta file. 
headers need to include the species at the end, separated by a space from the rest of the header. In this way, the species will be recognized and its genome will be downloaded, which is need to run TEAid. 

## Pipeline workflow
This computational tool offers two options:
- Option 1: Training a model and testing with synthetic data, generated, for example, from a curated library of TE sequences.
- Option 2: TE trimming from an input FASTA file containing sequences to curate.

For both options, FASTA headers must follow a specific format, with the species indicated at the end of the header, separated by a space from the rest of the text. Example:
```
>DR000395818#CLASSI/LINE/CR1 Bucorvus abyssinicus
```

### Option 1: Generation of synthetic data
This option generates sequences containing one or two TEs, combined with randomly generated DNA sequences, for a total length of 15,000 bp. This script does not require a specific environment.

**Batch processing (HPC/SLURM):**
```batch
sbatch data_generation/run_generation.sh
```
**Executes** `data_generation/GenerationData.py`

**Output**
It will generate a FASTA file `simulated_data_merged.fasta`.

**Parameters**
- `--fasta`: Path to FASTA file from which synthetic data will be generated (required)
- `--seq_per_case`: number of sequences to generate per case (4 cases)

For next steps after synthetic data generation, the `teaid_agp` environment is required.

### Create images from FASTA file with TE-Aid
For this step, the following scripts from Goubert et al. (2022) are needed: TE-Aid, Run-c2g.R, consensus2genome.R and blastndotplot.R.
By default, they will be included in the folder `TEAid`.

**Batch processing (HPC/SLURM):**
```batch
sbatch run_teaid.sh
```
**Executes** `Auto_trimming.py with --mode teaid`

**Parameters**
- `--input_fasta`: Path to FASTA file from which the images will be created (required)

**Output**
It will create a folder called `te_aid`, in which the PDF files and images will be generated. 

### Create a dataset

**Batch processing (HPC/SLURM):**
```batch
sbatch create_dataset.sh
```
**Executes** `Auto_trimming.py with --mode dataset`

**Parameters**
- `--input_fasta`: Path to FASTA file from which the dataset will be created (required)
- `--dataset_dir`: Directory where the dataset will be saved

**Output**
In the specified directory, this will generate 4 Numpy matrices: features_data.npy, labels_data.npy, case_labels.npy and species_labels.npy.

After generating the PDFs (and images) and the dataset, we can choose in the script Auto_trimming.py if we want to do training or testing (for option 1) or trimming (for option 2). 

**Batch processing (HPC/SLURM):**
```batch
sbatch auto_trimming.sh
```

### Training

**Executes** `Auto_trimming.py with --mode train`

**Parameters**
- `--dataset_dir`: Directory where the dataset is saved

**Output**


### Testing

**Executes** `Auto_trimming.py with --mode test`

**Parameters**
- `--dataset_dir`: Directory where the dataset is saved
- `--model`:
- `--scaler`:

### Trimming
Previously, we will need to generate the dataset from the FASTA files with the sequences that we want to trim. This will generate a .txt file where the trimmed sequences will be saved with the original header name (until # symbol) and indicates the positions in which the sequence was cut.

**Executes** `Auto_trimming.py with --mode trimming`

**Parameters**
- `--input_fasta`: Path to FASTA file from which the dataset was created (required)
- `--dataset_dir`: Directory where the dataset is saved
- `--model`:
- `--scaler`:
