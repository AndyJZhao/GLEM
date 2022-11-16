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

### GLEM+GIANT+GAMLP
For **ogbn-papers100M**
We first run the `GIANT+GAMLP+RLU` model and save the prediction from it. Note that the `GIANT+GAMLP+RLU` model do not use 
the all nodes' predictions, but our model need all nodes' predictions. Considering the memory limit, we use the 2 hops features to train the `GIANT+GAMLP+RLU` model
and get the all node's predictions.


Then we use the saved prediction to train the GLEM model. We do not use the `RLU` trick in the `GAMLP`, and we follow the default `GAMLP` settings.
``` 
python src/models/GLEM/trainGLEM.py --dataset=paper_TA --em_order=LM-first --gnn_act=sigmoid --gnn_att_dropout=0 --gnn_aug_batch_size=50000 --gnn_average=T0 --gnn_ce_reduction=mean --gnn_ckpt=GIANT-GAMLP --gnn_early_stop=120 --gnn_epochs=150 --gnn_input_dropout=0 --gnn_input_norm=F --gnn_label_input=F --gnn_model=GAMLP --gnn_n_layers=6 --gnn_num_hops=6 --gnn_pl_ratio=0.1 --gnn_pl_weight=0.05 --gnn_prt_batch_size=40000 --inf_n_epochs=1 --inf_tr_n_nodes=1210000 --lm_ce_reduction=mean --lm_cla_dropout=0.3 --lm_epochs=6 --lm_eq_batch_size=480 --lm_eval_patience=400000 --lm_feat_shrink=100 --lm_init_ckpt=None --lm_label_smoothing_factor=0.2 --lm_load_best_model_at_end=T --lm_lr=2e-05 --lm_model=Deberta --lm_pl_ratio=0.5 --lm_pl_weight=0.4 --lm_warmup_epochs=1 --pl_filter=0.85 --pseudo_temp=0.2  --gpus=0
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

GIANT paper:
```
@article{chien2021node,
  title={Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction},
  author={Eli Chien and Wei-Cheng Chang and Cho-Jui Hsieh and Hsiang-Fu Yu and Jiong Zhang and Olgica Milenkovic and Inderjit S Dhillon},
  journal={arXiv preprint arXiv:2111.00064},
  year={2021}
}
```