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

For the `GLEM+EnGCN`, you will need to implement some special library in the [EnGCN](https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking)

## Training
In the learning process of the LM model, we turn on fp16 during the training and eval process in order to improve the learning speed; 
In order to obtain better results, we can turn off fp16 and adjust the corresponding batchsize during the pre-training process of LM.

### GLEM+EnGCN
For **ogbn-products**

We will later incorporate EnGCN into the `src` directory, for now we will provide the commands for step-by-step implementation:
#### PreTrain Phase
If you have the `Deberta.emb` pretrained from our framework in the `GLEM/temp/prt_lm/products_TC/Deberta/Ftv1/Deberta.emb`, you can run the command as follows to get the `preds` from the EnGCN:

```python
python main.py --type_model EnGCN --dataset ogbn-products --cuda_num 0 --lr 0.01 --weight_decay 0 --dropout 0.2 --epochs 70 --dim_hidden 512 --num_layers 8 --use_batch_norm True --batch_size 10000 --SLE_threshold 0.8 --N_exp 1 --tosparse  --LM_emb_path 'GLEM/temp/prt_lm/products_TC/Deberta/Ftv1/Deberta.emb'
```
The preds will be saved in the `GLEM/OGB/ogbn-products/output/ogbn-products/` and the name is composed of the uuex format(you can rename it as EnGCN.pt)

Then you can move to the 'GLEM/src/utils/function/', and run the command as follows to save the `preds` from the EnGCN:
```python
python save_preds.py --out_put 'GLEM/temp/prt_gnn/products_TC/EnGCN/EnGCN/' --pred_path 'GLEM/OGB/ogbn-products/output/ogbn-products/EnGCN.pt'
```
#### EM Phase:
We can run the command to tune the LM model learning from the EnGCN:
```python
python src/models/GLEM/trainGLEM.py --dataset=products_TC --em_order=LM-first --gnn_ce_reduction=mean --gnn_ckpt=EnGCN  --gnn_model=EnGCN  --inf_n_epochs=1 --inf_tr_n_nodes=200000 --lm_ce_reduction=mean --lm_cla_dropout=0.4 --lm_epochs=1 --lm_eq_batch_size=120 --lm_eval_patience=65308 --lm_init_ckpt=PrevEM --lm_label_smoothing_factor=0 --lm_load_best_model_at_end=T --lm_lr=3e-05 --lm_model=Deberta --lm_pl_ratio=1 --lm_pl_weight=0.05 --pl_filter=0.9 --pseudo_temp=0.2 --seed=0 --gpus=0
```
Then we can get the 'Deberta.emb' generated from the Inference stage, we can move the emb file to the 'GLEM/OGB/ogbn-products/lm_emb/' and we can run the command to get the accuracy of the `GLEM+EnGCN`:
```python
python main.py --type_model EnGCN --dataset ogbn-products --cuda_num 0 --lr 0.01 --weight_decay 0 --dropout 0.2 --epochs 70 --dim_hidden 512 --num_layers 8 --use_batch_norm True --batch_size 10000 --SLE_threshold 0.8 --N_exp 1 --tosparse --exp_name 'GLEM_EnGCN_seed0' --seed 0 --LM_emb_path 'GLEM/OGB/ogbn-products/lm_emb/seed0/Deberta.emb'
```
For the different seed we need to save the `Deberta.emb` in the different place.
The different 
|Seed | seed0   | seed1    | seed2 | seed3   | seed4   | seed5|seed6  | seed7   | seed8 | seed9 | Mean | Std|
|----|  ----  | ----  |  ---- | ----  | ----  |  ---- | ----  | ----  |  ---- | ---- | ---- | ---- |
|val_acc| 93.6449 | 93.6856| 93.6933| 93.7085| 93.6805| 93.7619| 93.7365 | 93.7314| 93.6500| 93.6983| 93.69909 | 0.035193| 
|test_acc | 90.0622 |90.1152| 90.2959|90.1042 | 90.0136| 90.2023|90.2997 | 90.2663| 89.9220| 90.1255| 90.14069| 0.118869|

We do not change  any setting of the `EnGCN.sh` in the [EnGCN Github](https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking/blob/main/scripts/ogbn-products/EnGCN.sh) 
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
| GLEM+EnGCN|0.9370 ± 0.0004 | 0.9014 ± 0.0012 |
| GLEM+GIANT+SAGN+SCR |0.9400 ± 0.0003 | 0.8737 ± 0.0006 |

## Citation
EnGCN paper:
```
@article{duan2022comprehensive,
  title={A Comprehensive Study on Large-Scale Graph Training: Benchmarking and Rethinking},
  author={Duan, Keyu and Liu, Zirui and Wang, Peihao and Zheng, Wenqing and Zhou, Kaixiong and Chen, Tianlong and Hu, Xia and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2210.07494},
  year={2022}
}
```

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
