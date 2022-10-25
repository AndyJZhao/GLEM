from models.GNNs.SAGN.model import *
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from .load_dataset import load_dataset
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import uuid
import random


def gen_model(args, in_size, num_classes):
    print("R_GAMLP")
    return R_GAMLP(in_size, args.hidden, num_classes, args.num_hops + 1,
                   args.dropout, args.input_drop, args.att_drop, args.alpha, args.n_layers_1, args.n_layers_2, args.act,
                   args.pre_process, args.residual, args.pre_dropout, args.bns)


def gen_model_rlu(args, in_size, num_classes):
    print("R_GAMLP_RLU")
    return R_GAMLP_RLU(in_size, args.hidden, num_classes, args.num_hops + 1,
                       args.dropout, args.input_drop, args.att_drop, args.label_drop, args.alpha, args.n_layers_1,
                       args.n_layers_2, args.n_layers_3, args.act, args.pre_process, args.residual, args.pre_dropout,
                       args.bns)


def gen_model_sagn(args, in_feats, label_in_feats, n_classes):
    num_hops = args.num_hops + 1
    base_model = SAGN(in_feats, args.hidden, n_classes, num_hops,
                      args.mlp_layer, args.num_heads,
                      weight_style=args.weight_style,
                      dropout=args.dropout,
                      input_drop=args.input_drop,
                      attn_drop=args.att_drop,
                      zero_inits=args.zero_inits,
                      position_emb=args.position_emb,
                      focal=args.focal)
    label_model = GroupMLP(label_in_feats,
                           args.hidden,
                           n_classes,
                           args.num_heads,
                           args.label_mlp_layer,
                           args.label_drop,
                           residual=args.label_residual, )
    model = SLEModel(base_model, label_model)
    return model


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def consis_loss(logps, temp, lam):
    ps = [torch.exp(p) for p in logps]
    ps = torch.stack(ps, dim=2)
    avg_p = torch.mean(ps, dim=2)
    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    sharp_p = sharp_p.unsqueeze(2)

    mse = torch.mean(torch.sum(torch.pow(ps - sharp_p, 2), dim=1, keepdim=True))
    # kl = torch.mean(torch.sum(ps * (torch.log(ps+1e-8) - torch.log(sharp_p+1e-8)), dim=1, keepdim=True))
    mse = lam * mse
    return mse


def consis_loss_mean_teacher(p_t, p_s, temp, lam):
    p_t = F.log_softmax(p_t, dim=-1)
    p_t = torch.exp(p_t)
    sharp_p_t = (torch.pow(p_t, 1. / temp) / torch.sum(torch.pow(p_t, 1. / temp), dim=1, keepdim=True)).detach()
    p_s = F.softmax(p_s, dim=1)

    # mse = F.mse_loss(sharp_p_t,p_s, reduction = 'mean')

    log_sharp_p_t = torch.log(sharp_p_t + 1e-8)

    loss = torch.mean(torch.sum(torch.pow(p_s - sharp_p_t, 2), dim=1, keepdim=True))
    kl = torch.mean(torch.sum(p_s * (torch.log(p_s + 1e-8) - log_sharp_p_t), dim=1, keepdim=True))
    # kldiv = F.kl_div(log_sharp_p_t,p_s,reduction = 'mean')

    loss = lam * loss
    return loss, kl


def train_rlu_consis(model, train_loader, enhance_loader, optimizer, evaluator, device, xs, labels, label_emb,
                     predict_prob, args, enhance_loader_cons):
    model.train()
    loss_fcn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []

    total_loss = 0
    iter_num = 0
    for idx_1, idx_2, idx_3 in zip(train_loader, enhance_loader, enhance_loader_cons):
        logits = []
        idx = torch.cat((idx_1, idx_2), dim=0)
        feat_list = [x[idx].to(device) for x in xs]

        batch_feats = [x[idx_3].to(device) for x in xs]

        y = labels[idx_1].to(torch.long).to(device)
        optimizer.zero_grad()
        output_att = model(feat_list, label_emb[idx].to(device))

        output_att_f = model(batch_feats, label_emb[idx_3].to(device))

        output_att_cons = model(batch_feats, label_emb[idx_3].to(device))

        logits.append(torch.log_softmax(output_att_cons, dim=-1))
        logits.append(torch.log_softmax(output_att_f, dim=-1))
        loss_consis = consis_loss(logits, args.tem, args.lam)

        L1 = loss_fcn(output_att[:len(idx_1)], y) * (len(idx_1) * 1.0 / (len(idx_1) + len(idx_2)))
        teacher_soft = predict_prob[idx_2].to(device)
        teacher_prob = torch.max(teacher_soft, dim=1, keepdim=True)[0]
        L3 = (teacher_prob * (teacher_soft * (
                    torch.log(teacher_soft + 1e-8) - torch.log_softmax(output_att[len(idx_1):], dim=1)))).sum(
            1).mean() * (len(idx_2) * 1.0 / (len(idx_1) + len(idx_2)))

        loss = L1 + L3 * args.gama + loss_consis
        loss.backward()
        optimizer.step()
        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss
        iter_num += 1

    loss = total_loss / iter_num
    approx_acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, approx_acc


def train_rlu(model, train_loader, enhance_loader, optimizer, evaluator, device, xs, labels, label_emb, predict_prob,
              gama):
    model.train()
    loss_fcn = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss = 0
    iter_num = 0
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        feat_list = [x[idx].to(device) for x in xs]
        y = labels[idx_1].to(torch.long).to(device)
        optimizer.zero_grad()
        output_att = model(feat_list, label_emb[idx].to(device))
        L1 = loss_fcn(output_att[:len(idx_1)], y) * (len(idx_1) * 1.0 / (len(idx_1) + len(idx_2)))
        teacher_soft = predict_prob[idx_2].to(device)
        teacher_prob = torch.max(teacher_soft, dim=1, keepdim=True)[0]
        L3 = (teacher_prob * (teacher_soft * (
                    torch.log(teacher_soft + 1e-8) - torch.log_softmax(output_att[len(idx_1):], dim=1)))).sum(
            1).mean() * (len(idx_2) * 1.0 / (len(idx_1) + len(idx_2)))
        loss = L1 + L3 * gama
        loss.backward()
        optimizer.step()
        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss
        iter_num += 1

    loss = total_loss / iter_num
    approx_acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, approx_acc


def train(model, feats, labels, device, loss_fcn, optimizer, train_loader, label_emb, evaluator, args, ema=None):
    model.train()
    total_loss = 0
    iter_num = 0
    y_true = []
    y_pred = []
    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        if args.method == "SAGN":
            output_att, _ = model(batch_feats, label_emb[batch].to(device))
        else:
            output_att = model(batch_feats, label_emb[batch].to(device))
        y_true.append(labels[batch].to(torch.long))
        y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        L1 = loss_fcn(output_att, labels[batch].to(device))
        loss_train = L1
        total_loss = loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        if ema != None:
            ema.update()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc


def train_mean_teacher(model, teacher_model, feats, labels, device, loss_fcn, optimizer, train_loader, enhance_loader,
                       label_emb, evaluator, args, global_step, ema=None):
    model.train()
    # teacher_model.train()
    total_loss = 0
    total_loss_consis = 0
    total_loss_kl = 0
    total_loss_supervised = 0
    iter_num = 0
    y_true = []
    y_pred = []
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        feat_list = [x[idx].to(device) for x in feats]
        feat_list_teacher = [x[idx_2].to(device) for x in feats]

        if args.method == "SAGN":
            output_att, _ = model(feat_list, label_emb[idx].to(device))
        else:
            output_att = model(feat_list, label_emb[idx].to(device))

        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        # L1 = loss_fcn(output_att[:len(idx_1)], labels[idx_1])*(len(idx_1)*1.0/(len(idx_1)+len(idx_2)))
        L1 = loss_fcn(output_att[:len(idx_1)], labels[idx_1].to(device))

        with torch.no_grad():
            if args.method == "SAGN":
                mean_t_output, _ = teacher_model(feat_list_teacher, label_emb[idx_2].to(device))
            else:
                mean_t_output = teacher_model(feat_list_teacher, label_emb[idx_2].to(device))

        student_output = output_att[len(idx_1):]

        p_t = mean_t_output
        p_s = student_output

        loss_consis, kl_loss = consis_loss_mean_teacher(p_t, p_s, args.tem, args.lam)
        loss_supervised = args.sup_lam * L1
        kl_loss = args.kl_lam * kl_loss
        if args.kl:
            loss_train = loss_supervised + kl_loss
        else:
            loss_train = loss_supervised + loss_consis

        total_loss += loss_train
        total_loss_consis += loss_consis
        total_loss_kl += kl_loss
        total_loss_supervised += loss_supervised
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        if ema != None:
            ema.update()

        if args.adap == True:
            alpha = min(1 - 1 / (global_step + 1), args.ema_decay)
        else:
            alpha = args.ema_decay
        for mean_param, param in zip(teacher_model.parameters(), model.parameters()):
            mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

        iter_num += 1
    loss = total_loss / iter_num
    loss_cons = total_loss_consis / iter_num
    loss_kl = total_loss_kl / iter_num
    loss_sup = total_loss_supervised / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc


@torch.no_grad()
def test(model, feats, labels, device, test_loader, evaluator, label_emb, args, ema=None):
    if ema != None:
        ema.apply_shadow()
    model.eval()
    preds = []
    true = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        if args.method == "SAGN":
            output, _ = model(batch_feats, label_emb[batch].to(device))
        else:
            output = model(batch_feats, label_emb[batch].to(device))
        preds.append(torch.argmax(output, dim=-1))
        true.append(labels[batch].to(device))
    true = torch.cat(true)
    preds = torch.cat(preds, dim=0)
    res = evaluator(preds, true)
    if ema != None:
        ema.restore()
    return res

def train_sagn(device, model, feats, label_emb, labels, loss_fcn, optimizer, train_loader, args, ema=None):
    model.train()
    for batch in train_loader:
        if len(batch) == 1:
            continue
        batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
        if label_emb is not None:
            batch_label_emb = label_emb[batch].to(device)
        else:
            batch_label_emb = None

        out, _ = model(batch_feats, batch_label_emb)
        loss = loss_fcn(out, labels[batch].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema != None:
            ema.update()


def train_mean_teacher_sagn(device, model, teacher_model, feats, label_emb, labels, loss_fcn, optimizer, train_loader,
                            enhance_loader_cons, args, ema=None):
    model.train()
    total_loss = 0
    total_loss_mse = 0
    total_loss_kl = 0
    total_loss_supervised = 0
    iter_num = 0
    for idx1, idx2 in zip(train_loader, enhance_loader_cons):
        if len(idx1) == 1:
            continue
        batch_feats = [x[idx1].to(device) for x in feats] if isinstance(feats, list) else feats[idx1].to(device)
        batch_feats_cons = [x[idx2].to(device) for x in feats] if isinstance(feats, list) else feats[idx2].to(device)

        if label_emb is not None:
            batch_label_emb = label_emb[idx1].to(device)
            batch_label_emb_cons = label_emb[idx2].to(device)
        else:
            batch_label_emb = None
            batch_label_emb_cons = None

        out, _ = model(batch_feats, batch_label_emb)
        out_s, _ = model(batch_feats_cons, batch_label_emb_cons)
        out_t, _ = teacher_model(batch_feats_cons, batch_label_emb_cons)

        mse, kl = consis_loss_mean_teacher(out_t, out_s, args.tem, args.lam)
        kl = kl * args.kl_lam

        L1 = loss_fcn(out, labels[idx1].to(device))
        if args.kl == False:
            loss = L1 + mse
        else:
            loss = L1 + kl

        total_loss += loss
        total_loss_mse += mse
        total_loss_kl += kl
        total_loss_supervised += L1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema != None:
            ema.update()

        alpha = args.ema_decay
        for mean_param, param in zip(teacher_model.parameters(), model.parameters()):
            mean_param.data.mul_(alpha).add_(1 - alpha, param.data)
        iter_num += 1
    loss = total_loss / iter_num
    loss_mse = total_loss_mse / iter_num
    loss_sup = total_loss_supervised / iter_num
    loss_kl = total_loss_kl / iter_num


def test_sagn(device, model, feats, label_emb, labels, loss_fcn, val_loader, test_loader, evaluator,
              train_nid, val_nid, test_nid, args, ema):
    if ema != None:
        ema.apply_shadow()
    model.eval()
    num_nodes = labels.shape[0]
    loss_list = []
    count_list = []
    preds = []
    for batch in val_loader:
        batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
        batch_label_emb = label_emb[batch].to(device)
        # We can get attention scores from SAGN
        out, _ = model(batch_feats, batch_label_emb)
        loss_list.append(loss_fcn(out, labels[batch].to(device)).cpu().item())
        count_list.append(len(batch))
    loss_list = np.array(loss_list)
    count_list = np.array(count_list)
    val_loss = (loss_list * count_list).sum() / count_list.sum()
    start = time.time()
    for batch in test_loader:
        batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
        batch_label_emb = label_emb[batch].to(device)
        out, _ = model(batch_feats, batch_label_emb)
        preds.append(torch.argmax(out, dim=-1))

    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    end = time.time()
    train_res = evaluator(preds[:len(train_nid)], labels[train_nid])
    val_res = evaluator(preds[len(train_nid):(len(train_nid) + len(val_nid))], labels[val_nid])
    test_res = evaluator(preds[(len(train_nid) + len(val_nid)):], labels[test_nid])

    if ema != None:
        ema.restore()
    return train_res, val_res, test_res, val_loss, end - start


@torch.no_grad()
def gen_output_torch(model, feats, test_loader, device, label_emb, ema=None):
    if ema != None:
        ema.apply_shadow()
    model.eval()
    preds = []
    for batch in test_loader:
        batch_feats = [torch.from_numpy(np.array(x[batch])).to(torch.float32).to(device) for x in feats]
        output, _ = model(batch_feats, label_emb[batch].to(device))
        preds.append(output.cpu())
    preds = torch.cat(preds, dim=0)
    if ema != None:
        ema.restore()
    return preds
