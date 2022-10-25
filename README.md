# LEARNING ON LARGE-SCALE TEXT-ATTRIBUTED GRAPHS VIA VARIATIONAL INFERENCE
This repository is the official implementation of **GLEM**.

## Overview
The proposed GLEM framework trains GNN and LM separately in a variational EM
framework: In E-step, an LM is trained towards predicting both the gold and GNN predicted pseudolabels; In M-step, a GNN is trained by predicting LM-inferenced pseudo-labels using the embeddings and labels predicted by LM.
  
<img src="Framework.jpg" width="80%" height="80%">

## Requirements
We use the miniconda to manage the python environment. We provide the environment.yaml to implement the environment
```
conda env create -f environment.yml
```

## Training
Please look the details in Readme.md of each dataset inside the OGB folder.

## Node Classification Results:

Performance on **ogbn-arxiv**(10 runs):
| Methods   | Validation accuracy  | Test accuracy  |
|  ----  | ----  |  ---- |
| GLEM+RevGAT|0.7749 ± 0.0017 | 0.7697± 0.0019 |
| GLEM+GAMLP |0.7695 ± 0.0014 | 0.7562± 0.0023 |
| GLEM+GCN   |0.7686 ± 0.0019 | 0.7593± 0.0019 |

Performance on **ogbn-products**(10 runs):
| Methods   | Validation accuracy  | Test accuracy  |
|  ----  | ----  |  ---- |
| GLEM+GIANT-XRT+SAGN+SCR| 0.9400 ± 0.0003 | 0.8736 ± 0.0007 |
| GLEM+GAMLP| 0.9419 ± 0.0001 | 0.8509 ± 0.0021 |


## Citation
Our paper:
```
```
The different models used for different datasets are referenced in the corresponding OGB subfolders
