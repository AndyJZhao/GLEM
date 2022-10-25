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
For **ogbn-products**
We first run the `GIANT+SAGN+SCR` model and save the prediction from it.
Please follow the instruction in [SCR](https://github.com/THUDM/SCR/tree/main/ogbn-products) to get the preds.
Note that we just need `--num-runs 1`. Then we get the preds in the path1.
The PROJ_DIR is defined as follows: 
```python 
import os.path as osp
PROJ_DIR = osp.abspath(osp.dirname(__file__)).split('src')[0]
```
Then we can save the preds from the `GIANT+SAGN+SCR`
```
cd src/utils/function/
python save_preds --out_put  PROJ_DIR/temp/prt_gnn/products_TC/SAGN/GIANT_SAGN_SCR --pred_path path1
```

After that, we can run the command as follows:
``` 
python src/models/GraphVF/trainGVF.py 
```

## Node Classification Results:
Performance on **ogbn-products**(10 runs):
| Methods   | Validation accuracy  | Test accuracy  |
|  ----  | ----  |  ---- |
| GLEM+GIANT+SAGN+SCR |0.9400 ± 0.0003 | 0.8736 ± 0.0007 |

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
