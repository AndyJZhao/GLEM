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
In the learning process of the LM model, we turn on fp16 during the training and eval process in order to improve the learning speed; 
In order to obtain better results, we can turn off fp16 and adjust the corresponding batchsize during the pre-training process of LM.

### GLEM+GIANT+SAGN+SCR
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
python src/models/GLEM/trainGLEM.py 
```

## Node Classification Results:
Performance on **ogbn-products**(10 runs):
| Methods   | Validation accuracy  | Test accuracy  |
|  ----  | ----  |  ---- |
| GLEM+GIANT+SAGN+SCR |0.9400 ± 0.0003 | 0.8737 ± 0.0006 |

## Citation

SCR paper:
```
@misc{zhang2021improving,
      title={Improving the Training of Graph Neural Networks with Consistency Regularization}, 
      author={Chenhui Zhang and Yufei He and Yukuo Cen and Zhenyu Hou and Jie Tang},
      year={2021},
      eprint={2112.04319},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}
```
GIANT paper:
```
@article{chien2021node,
  title={Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction},
  author={Eli Chien and Wei-Cheng Chang and Cho-Jui Hsieh and Hsiang-Fu Yu and Jiong Zhang and Olgica Milenkovic and Inderjit S Dhillon},
  journal={arXiv preprint arXiv:2111.00064},
  year={2021}
}
```
SAGN paper:
```
@article{sun2021scalable,
  title={Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training},
  author={Sun, Chuxiong and Wu, Guoshi},
  journal={arXiv preprint arXiv:2104.09376},
  year={2021}
}
```
