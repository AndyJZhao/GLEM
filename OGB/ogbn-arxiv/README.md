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
### GLEM+RevGAT
For **ogbn-arxiv**
```
python src/models/GLEM/trainGLEM.py --dataset=arxiv_TA --em_order=LM-first --gnn_ckpt=RevGAT --gnn_early_stop=300 --gnn_epochs=2000 --gnn_input_norm=T --gnn_label_input=F --gnn_model=RevGAT --gnn_pl_ratio=1 --gnn_pl_weight=0.05 --inf_n_epochs=2 --inf_tr_n_nodes=100000 --lm_ce_reduction=mean --lm_cla_dropout=0.4 --lm_epochs=3 --lm_eq_batch_size=30 --lm_eval_patience=30460 --lm_init_ckpt=None --lm_label_smoothing_factor=0 --lm_load_best_model_at_end=T --lm_lr=2e-05 --lm_model=Deberta --lm_pl_ratio=1 --lm_pl_weight=0.8 --pseudo_temp=0.2  --gpus=0 
```
The `inf_n_epochs` controls the number of iterations of the `EM-Step`, which on ogbn-arxiv generally converges at iteration=1, with gnn_val accuracy reaching its maximum.

### GLEM+EnGCN
For the `GLEM+EnGCN`, you will need to implement some special library in the [EnGCN](https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking)
For **ogbn-arxiv**
#### PreTrain Phase
If you have the `Deberta.emb` pretrained from our framework in the `GLEM/temp/prt_lm/arxiv_TA/Deberta/Ftv1/Deberta.emb`, you can run the command as follows to get the `preds` from the EnGCN:
```python
python main.py --type_model EnGCN --dataset ogbn-arxiv --cuda_num 0 --lr 0.001 --weight_decay 0.0001 --dropout 0.1 --epochs 100 --dim_hidden 512 --num_layers 8 --use_batch_norm False --batch_size 10000 --SLE_threshold 0.5 --N_exp 1 --tosparse  --LM_emb_path 'GLEM/temp/prt_lm/arxiv_TA/Deberta/Ftv1/Deberta.emb'
```
The preds will be saved in the `GLEM/OGB/ogbn-arxiv/output/ogbn-arxiv/` and the name is composed of the uuex format(you can rename it as EnGCN.pt)
Then you can move to the 'GLEM/src/utils/function/', and run the command as follows to save the `preds` from the EnGCN:
```python
python save_preds.py --out_put 'GLEM/temp/prt_gnn/arxiv_TA/EnGCN/EnGCN/' --pred_path 'GLEM/OGB/ogbn-arxiv/output/ogbn-arxiv/EnGCN.pt'
```
#### EM Phase:
We can run the command to tune the LM model learning from the EnGCN:
```python
python src/models/GLEM/trainGLEM.py --dataset=arxiv_TA --em_order=LM-first --gnn_ckpt=EnGCN --gnn_epochs=400 --gnn_model=EnGCN --inf_n_epochs=1 --inf_tr_n_nodes=200000 --lm_ce_reduction=mean --lm_cla_dropout=0.4 --lm_epochs=1 --lm_eq_batch_size=60 --lm_eval_patience=65308 --lm_init_ckpt=PrevEM --lm_label_smoothing_factor=0 --lm_load_best_model_at_end=T --lm_lr=2e-07 --lm_model=Deberta --lm_pl_ratio=0.1 --lm_pl_weight=0.1 --pseudo_temp=0.5 --gpus=0
```
Then we can get the 'Deberta.emb' generated from the Inference stage, we can move the emb file to the 'GLEM/OGB/ogbn-arxiv/lm_emb/' and we can run the command to get the accuracy of the `GLEM+EnGCN`:
```python
python main.py --type_model EnGCN --dataset ogbn-arxiv --cuda_num 0 --lr 0.001 --weight_decay 0.0001 --dropout 0.1 --epochs 100 --dim_hidden 512 --num_layers 8 --use_batch_norm False --batch_size 10000 --SLE_threshold 0.5 --N_exp 10 --tosparse --LM_emb_path 'GLEM/OGB/ogbn-arxiv/lm_emb/Deberta.emb'
```
Note that we follow the same experiment's config in 'EnGCN'.

### GLEM+GAMLP
For **ogbn-arxiv**
For the seed:0:
``` 
python src/models/GLEM/trainGLEM.py 
```
### GLEM+GCN
For **ogbn-arxiv**
For the seed:0:
```
python src/models/GLEM/trainGLEM.py --dataset=arxiv_TA --em_order=GNN-first --gnn_early_stop=300 --gnn_epochs=500 --gnn_input_norm=F --gnn_label_input=T --gnn_model=GCN --gnn_pl_ratio=0.2 --gnn_pl_weight=0.7 --inf_n_epochs=2 --inf_tr_n_nodes=100000 --lm_ce_reduction=mean --lm_cla_dropout=0.4 --lm_epochs=1 --lm_eq_batch_size=30 --lm_eval_patience=30460 --lm_init_ckpt=PrevEM --lm_label_smoothing_factor=0 --lm_load_best_model_at_end=T --lm_lr=3e-05 --lm_model=Deberta --lm_pl_ratio=0.1 --lm_pl_weight=0.5 --pl_filter=0.8 --pseudo_temp=0.2 --seed=0--gpus=0
```
## Node Classification Results:
Performance on **ogbn-arxiv**(10 runs):
| Methods   | Validation accuracy  | Test accuracy  |
|  ----  | ----  |  ---- |
| GLEM+EnGCN |0.8017 ± 0.0007 | 0.7966 ± 0.0006 |
| GLEM+RevGAT|0.7749 ± 0.0017 | 0.7697 ± 0.0019 |
| GLEM+GAMLP |0.7695 ± 0.0014 | 0.7562 ± 0.0023 |
| GLEM+GCN   |0.7686 ± 0.0019 | 0.7593 ± 0.0019 |

## Citation
Our paper:
```

```
EnGCN paper:
```
@article{duan2022comprehensive,
  title={A Comprehensive Study on Large-Scale Graph Training: Benchmarking and Rethinking},
  author={Duan, Keyu and Liu, Zirui and Wang, Peihao and Zheng, Wenqing and Zhou, Kaixiong and Chen, Tianlong and Hu, Xia and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2210.07494},
  year={2022}
}
```

RevGAT paper:
```
@InProceedings{li2021gnn1000,
    title={Training Graph Neural Networks with 1000 layers},
    author={Guohao Li and Matthias Müller and Bernard Ghanem and Vladlen Koltun},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2021}
}
```

GAMLP paper:
```
@article{zhang2021graph,
  title={Graph attention multi-layer perceptron},
  author={Zhang, Wentao and Yin, Ziqi and Sheng, Zeang and Ouyang, Wen and Li, Xiaosen and Tao, Yangyu and Yang, Zhi and Cui, Bin},
  journal={arXiv preprint arXiv:2108.10097},
  year={2021}
}
```


GCN paper:
```
@article{kipf2016semi,
  title={Semi-supervised classification with graph convolutional networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
