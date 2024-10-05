<div align="center">

# GCDM-SBDD

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Paper](http://img.shields.io/badge/arXiv-2302.04313-B31B1B.svg)](https://arxiv.org/abs/2302.04313)
[![Checkpoints DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13375913.svg)](https://doi.org/10.5281/zenodo.13375913)

</div>

## Description

This is the official structure-based drug design (**SBDD**) codebase of the paper

**Geometry-Complete Diffusion for 3D Molecule Generation and Optimization**, *Nature CommsChem*

[[arXiv](https://arxiv.org/abs/2302.04313)] [[Nature CommsChem](https://www.nature.com/articles/s42004-024-01233-z)]

<div align="center">

![Animation of a diffusion model-generated 3D binding pocket molecule visualized iteratively](img/GCDM_Sampled_4OZ2_Binding_Pocket_Molecule_Long_Trajectory.gif)

</div>

## Contents

- [System requirements](#system-requirements)
- [Installation guide](#installation-guide)
- [Tutorials](#tutorials)
- [Demo](#demo)
  - [Sample molecules for a given pocket](#sample-molecules-for-a-given-pocket)
- [Instructions for use](#instructions-for-use)
  - [Training](#training)
  - [Reproduce paper results](#reproduce-paper-results)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)

## System requirements

### OS requirements
This package supports Linux. The package has been tested on the following Linux system:
`Description: AlmaLinux release 8.9 (Midnight Oncilla)`

### Python dependencies
This package is developed and tested under Python 3.10.x. The primary Python packages and their versions are as follows. For more details, please refer to the `environment.yaml` file.
```python
hydra-core=1.3.2
matplotlib-base=3.7.1
numpy=1.24.3
pyg=2.3.0=py310_torch_2.0.0_cu118
python=3.10.11
pytorch=2.0.1=py3.10_cuda11.8_cudnn8.7.0_0
pytorch-scatter=2.1.1=py310_torch_2.0.0_cu118
pytorch-lightning=2.0.2
scikit-learn=1.2.2
torchmetrics=0.11.4
```

## Installation guide

Install `mamba` (~500 MB: ~1 minute)

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

Install dependencies (~15 GB: ~10 minutes)

```bash
# clone project
git clone https://github.com/BioinfoMachineLearning/GCDM-SBDD
cd GCDM-SBDD

# create conda environment
mamba env create -f environment.yaml
conda activate GCDM-SBDD  # note: one still needs to use `conda` to (de)activate environments

# install local project as package
pip3 install -e .
```

Download checkpoints (~500 MB extracted: ~2 minutes)

**Note**: Make sure to be located in the project's root directory beforehand (e.g., `~/GCDM-SBDD/`)
```bash
# fetch and extract model checkpoints directory
wget https://zenodo.org/record/13375913/files/GCDM_SBDD_Checkpoints.tar.gz
tar -xzf GCDM_SBDD_Checkpoints.tar.gz
rm GCDM_SBDD_Checkpoints.tar.gz
```

**NOTE**: Trained EGNN baseline checkpoint files are also included in `GCDM_SBDD_Checkpoints.tar.gz`.

### QuickVina 2
For docking, download QuickVina 2 and copy it to your Conda environment's binary (`bin`) directory:
```bash
wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1
chmod +x qvina2.1
mv qvina2.1 $HOME/mambaforge/envs/GCDM-SBDD/bin
```

We need MGLTools for preparing the receptor for docking (pdb -> pdbqt) but it can mess up your Mamba environment, so I recommend making a new one:
```bash
mamba create -n mgltools -c bioconda mgltools
```

### Binding MOAD
#### Data preparation
Download the dataset
```bash
wget https://zenodo.org/record/13375913/files/every_part_a.zip
wget https://zenodo.org/record/13375913/files/every_part_b.zip
wget https://zenodo.org/record/13375913/files/every.csv

unzip every_part_a.zip
unzip every_part_b.zip
```
Process the raw data using
``` bash
python process_bindingmoad.py <bindingmoad_dir>
```
or, to suppress warnings,
```bash
python -W ignore process_bindingmoad.py <bindingmoad_dir>
```

### CrossDocked Benchmark

#### Data preparation
Download and extract the dataset as described by the authors of Pocket2Mol: https://github.com/pengxingang/Pocket2Mol/tree/main/data

Process the raw data using
```bash
python process_crossdock.py <crossdocked_dir> --no_H
```

## Tutorials

We provide a two-part tutorial series of Jupyter notebooks to provide users with a real-world example of how to use `GCDM-SBDD` for pocket-based molecule generation and filtering, as outlined below.

1. [Generating molecules in target protein pockets](https://github.com/BioinfoMachineLearning/GCDM-SBDD/blob/main/notebooks/pocket_based_molecule_generation.ipynb)
2. [Filtering generated molecules using PoseBusters](https://github.com/BioinfoMachineLearning/GCDM-SBDD/blob/main/notebooks/molecule_filtering_with_posebusters.ipynb)

## Demo

### Sample molecules for a given pocket
To sample small molecules for a given pocket with a trained model use the following command:
```bash
python generate_ligands.py <checkpoint>.ckpt --pdbfile <pdb_file>.pdb --outdir <output_dir> --resi_list <list_of_pocket_residue_ids>
```
For example:
```bash
python generate_ligands.py last.ckpt --pdbfile 1abc.pdb --outdir results/ --resi_list A:1 A:2 A:3 A:4 A:5 A:6 A:7 
```
Alternatively, the binding pocket can also be specified based on a reference ligand in the same PDB file:
```bash 
python generate_ligands.py <checkpoint>.ckpt --pdbfile <pdb_file>.pdb --outdir <output_dir> --ref_ligand <chain>:<resi>
```

Optional flags:
| Flag | Description |
|------|-------------|
| `--n_samples` | Number of sampled molecules |
| `--all_frags` | Keep all disconnected fragments |
| `--sanitize` | Sanitize molecules (invalid molecules will be removed if this flag is present) |
| `--relax` | Relax generated structure in force field |
| `--resamplings` | Inpainting parameter (doesn't apply if conditional model is used) |
| `--jump_length` | Inpainting parameter (doesn't apply if conditional model is used) |

## Instructions for use

### Training
Starting a new training run:
```bash
python -u train.py config=<config>.yml
```

Resuming a previous run:
```bash
python -u train.py config=<config>.yml resume=<checkpoint>.ckpt
```

### Reproduce paper results
`test.py` can be used to sample molecules for the entire testing set:
```bash
python test.py <checkpoint>.ckpt --test_dir <bindingmoad_dir>/processed_noH/test/ --outdir <output_dir> --fix_n_nodes
```
Using the optional `--fix_n_nodes` flag lets the model produce ligands with the same number of nodes as the original molecule. Other optional flags are identical to `generate_ligands.py`. 

#### Compute sample metrics
For assessing basic molecular properties create an instance of the `MoleculeProperties` class and run its `evaluate` method:
```python
from analysis.metrics import MoleculeProperties
mol_metrics = MoleculeProperties()
all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = \
    mol_metrics.evaluate(pocket_mols)
```
`evaluate()` expects a list of lists where the inner list contains all RDKit molecules generated for one pocket.

For computing docking scores, run QuickVina as described below.

#### Run QuickVina2
First, convert all protein PDB files to PDBQT files using MGLTools
```bash
conda activate mgltools
cd analysis
python2 docking_py27.py <bindingmoad_dir>/processed_noH/test/ <output_dir> bindingmoad
cd ..
conda deactivate
```
Then, compute QuickVina scores:
```bash
conda activate GCDM-SBDD
python3 analysis/docking.py --pdbqt_dir <docking_py27_outdir> --sdf_dir <test_outdir> --out_dir <qvina_outdir> --write_csv --write_dict --dataset moad
```

**NOTE**: One can reference `analysis/inference_analysis.py` and `analysis/molecule_analysis.py` to analyze the generated molecules.

## Docker

To build this project in a Docker container, you can use the following commands:

```bash
## Build the image
docker build -t gcdm-sbdd .

## Run the container (with GPUs and mounting the current directory)
docker run -it --gpus all -v .:/mnt --name gcdm-sbdd gcdm-sbdd
```

This Docker image is also available on Docker Hub at [`cford38/gcdm-sbdd`](https://hub.docker.com/r/cford38/gcdm-sbdd), which can be run with the following command:

```bash
# docker pull cford38/gcdm-sbdd

docker run -it --gpus all -v .:/mnt --name gcdm-sbdd cford38/gcdm-sbdd
```
(Note: This image includes the checkpoints in the main working directory `/software/GCDM-SBDD/checkpoints/`.)



## Acknowledgements

GCDM-SBDD builds upon the source code and data from the following projects:

* [Bio-Diffusion](https://github.com/BioinfoMachineLearning/Bio-Diffusion)
* [DiffSBDD](https://github.com/arneschneuing/DiffSBDD)
* [GCPNet](https://github.com/BioinfoMachineLearning/GCPNet)
* [PoseBusters](https://github.com/maabuu/posebusters)

We thank all their contributors and maintainers!

## License
This project is covered under the **MIT License**.

## Citation

If you use the code or data associated with this package or otherwise find this work useful, please cite:

```bibtex
@article{morehead2024geometry,
  title={Geometry-complete diffusion for 3D molecule generation and optimization},
  author={Morehead, Alex and Cheng, Jianlin},
  journal={Communications Chemistry},
  volume={7},
  number={1},
  pages={150},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
