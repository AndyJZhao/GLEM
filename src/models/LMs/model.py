import torch as th
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from utils.function.os_utils import init_random_state
from models.GLEM.GLEM_utils import compute_loss
import numpy as np
import torch.nn.functional as F


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, pseudo_label_weight=1, dropout=0.0, seed=0, cla_bias=True, is_augmented=False, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)
        self.pl_weight = pseudo_label_weight
        self.is_augmented = is_augmented

    def forward(self, **input):
        # Extract outputs from the model
        labels, is_gold = [input.pop(_) for _ in ["labels", 'is_gold']]
        outputs = self.bert_encoder(**input, output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        # print(f'{sum(is_gold)} gold, {sum(~is_gold)} pseudo')
        loss = compute_loss(logits, labels, self.loss_func, is_gold=is_gold, pl_weight=self.pl_weight, is_augmented=self.is_augmented)
        return TokenClassifierOutput(loss=loss, logits=logits)


class BertEmbInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.bert_encoder = model

    @th.no_grad()
    def forward(self, **input):
        # Extract outputs from the model
        outputs = self.bert_encoder(**input, output_hidden_states=True)
        emb = outputs['hidden_states'][-1]  # Last layer
        # Use CLS Emb as sentence emb.

        node_cls_emb = emb.permute(1, 0, 2)[0]
        return TokenClassifierOutput(logits=node_cls_emb)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink

    @th.no_grad()
    def forward(self, **input):
        # Extract outputs from the model
        batch_nodes = input.pop('node_id').cpu().numpy()
        bert_outputs = self.bert_classifier.bert_encoder(**input, output_hidden_states=True)
        emb = bert_outputs['hidden_states'][-1]  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)
        # Save prediction and embeddings to disk (memmap)
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)
        # Output empty to fit the Huggingface trainer pipeline
        empty = th.cuda.BoolTensor((len(batch_nodes), 1))
        return TokenClassifierOutput(logits=empty)
