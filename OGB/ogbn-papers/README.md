# LEARNING ON LARGE-SCALE TEXT-ATTRIBUTED GRAPHS VIA VARIATIONAL INFERENCE
This repository is the official implementation of GLEM.

## Environments
Implementing environment: Tesla V100 32GB (GPU)
  
## Requirements
- python=3.8
- ogb=1.3.3
- numpy>=1.19.5
- dgl>=0.8.0
- pytorch=1.10.2
- pyg=2.0.3


## Training

### GLEM+GAMLP
For **ogbn-papers100M**
For the seed=0:
``` 
python src/models/GLEM/trainGLEM.py 
```

## Node Classification Results:
Performance on **ogbn-papers100M**(3 runs):
| Methods   | Validation accuracy  | Test accuracy  |
|  ----  | ----  |  ---- |
| GLEM+GIANT+GAMLP+RLU |0.7354 ± 0.0001 | 0.7037 ± 0.0002 |

## Citation

GAMLP paper:
```
@article{zhang2021graph,
  title={Graph attention multi-layer perceptron},
  author={Zhang, Wentao and Yin, Ziqi and Sheng, Zeang and Ouyang, Wen and Li, Xiaosen and Tao, Yangyu and Yang, Zhi and Cui, Bin},
  journal={arXiv preprint arXiv:2108.10097},
  year={2021}
}
```
