"""
Code originates from original mTAND code on GitHub repository https://github.com/reml-lab/mTAN.

Original paper:
Shukla, Satya Narayan, and Benjamin Marlin. "Multi-Time Attention Networks for Irregularly Sampled Time Series."
International Conference on Learning Representations. 2020.
"""

#pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from random import SystemRandom
import models
import utils
import os

import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_enc')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=128) # 128
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', default=True, action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--classify-pertp', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--num-heads', type=int, default=1)
parser.add_argument('--freq', type=float, default=10.)

parser.add_argument('--dataset', type=str, default='P19', choices=['P12', 'P19', 'eICU', 'PAM'])
parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0')
parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
parser.add_argument('--reverse', default=False, help='if True,use female, older for tarining; if False, use female or younger for training')
parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                    help='use this only when splittype==random; otherwise, set as no_removal')
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                    help='use this only with P12 dataset (mortality or length of stay)')

parser.add_argument('--distillation', type=int, default=0)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--use_feat_range', type=int, default=8) # all: 34, vital: 8 #student(rec) 만 영향 받음

parser.add_argument('--self_distillation', type=int, default=1)
parser.add_argument('--self_distillation_lambda', type=int, default=0)
parser.add_argument('--self_distillation_teacher_lambda', type=int, default=0)
parser.add_argument('--GPU', type=str, default='1')

parser.add_argument('--exp_name', type=str, default='')



args = parser.parse_args()

def softmax_with_temperature(logits, temperature):
    return F.softmax(logits / temperature, dim=-1)

if __name__ == '__main__':
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    """"0 means no missing (full observations); 1.0 means no observation, all missed"""
    if args.withmissingratio:
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        missing_ratios = [0]
    for missing_ratio in missing_ratios:
        acc_all = []
        auc_all = []
        aupr_all = []
        precision_all = []
        recall_all = []
        F1_all = []
        upsampling_batch = True

        split_type = args.splittype  # possible values: 'random', 'age', 'gender' ('age' not possible for dataset 'eICU')
        reverse_ = args.reverse  # False, True
        feature_removal_level = args.feature_removal_level  # 'sample', 'set'
        num_runs = 5

        
        if args.distillation:
            # ----train teacher----
            experiment_id = int(SystemRandom().random() * 100000)
            seed = args.seed
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            print('we are using: {}'.format(device))

            args.classif = True
            dataset = args.dataset  # possible values: 'P12', 'P19', 'eICU', 'PAM'
            print('Dataset used: ', dataset)

            data_obj = utils.get_data(args, dataset, device, args.quantization, upsampling_batch, split_type,
                                      feature_removal_level, missing_ratio, reverse=reverse_, predictive_label=args.predictive_label)

            train_loader = data_obj["train_dataloader"]
            test_loader = data_obj["test_dataloader"]
            val_loader = data_obj["val_dataloader"]
            dim = data_obj["input_dim"]

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                n_classes = 2
            elif dataset == 'PAM':
                n_classes = 8

            teacher = models.enc_mtan_classif(
                        dim, torch.linspace(0, 1., 128), args.rec_hidden, args.embed_time, args.num_heads,
                        args.learn_emb, args.freq, device=device, n_classes=n_classes).to(device)

            params = (list(teacher.parameters()))

            optimizer = optim.Adam(params, lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            if args.fname is not None:
                checkpoint = torch.load(args.fname)
                teacher.load_state_dict(checkpoint['teacher_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('loading saved weights', checkpoint['epoch'])

            best_val_loss = float('inf')
            total_time = 0.
            best_aupr_val = 0
            teacher_model_save_path = f'./model_save_path/{args.exp_name}_teacher.pt'

            teacher.train()
            
            print('\n------------------\nRUN teacher: Training started\n------------------' )
            for itr in range(1, args.niters + 1):
                train_loss = 0
                train_n = 0
                train_acc = 0
                start_time = time.time()
                for train_batch, label in train_loader:
                    train_batch, label = train_batch.to(device), label.to(device)
                    batch_len = train_batch.shape[0]

                    observed_data_t, observed_mask_t, observed_tp_t \
                        = train_batch[:, :, :dim], train_batch[:, :, dim:dim*2], train_batch[:, :, -1]
                    out = teacher(torch.cat((observed_data_t, observed_mask_t), 2), observed_tp_t)
                    if args.classify_pertp:
                        N = label.size(-1)
                        out = out.view(-1, N)
                        label = label.view(-1, N)
                        _, label = label.max(-1)
                        loss = criterion(out, label.long())
                    else:
                        loss = criterion(out, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * batch_len
                    train_acc += torch.mean((out.argmax(1) == label).float()).item() * batch_len
                    train_n += batch_len

                total_time += time.time() - start_time

                # validation set
                val_loss, val_acc, val_auc, val_aupr, val_precision, val_recall, val_F1 = \
                    utils.evaluate_classifier(teacher, val_loader, args=args, dim=dim, dataset=dataset, is_teacher=True)
                best_val_loss = min(best_val_loss, val_loss)

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    print(
                        'VALIDATION: Iter: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.2f}, val_AUROC: {:.2f}, '
                        'val_AUPRC: {:.2f}'
                        .format(itr, train_loss / train_n, train_acc / train_n, val_loss, val_acc * 100, val_auc * 100,
                                val_aupr * 100))
                elif dataset == 'PAM':
                    print(
                        'VALIDATION: Iter: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.2f}, val_AUROC: {:.2f}, '
                        'val_AUPRC: {:.2f}, val_precision: {:.2f},val_recall: {:.2f},val_F1: {:.2f},'
                        .format(itr, train_loss / train_n, train_acc / train_n, val_loss, val_acc * 100, val_auc * 100,
                                val_aupr * 100, val_precision * 100, val_recall * 100, val_F1 * 100))
                # save the best model based on 'aupr'
                if val_aupr > best_aupr_val:
                    best_aupr_val = val_aupr
                    torch.save(teacher, teacher_model_save_path)

            print('\n------------------\nRUN teacher : Training finished\n------------------')
            
            # test set
            teacher = torch.load(teacher_model_save_path)
            test_loss, test_acc, test_auc, test_aupr, test_precision, test_recall, test_F1 = \
                utils.evaluate_classifier(teacher, test_loader, args=args, dim=dim, dataset=dataset, is_teacher=True)
            
            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                print("TEST: test_acc: %.2f, aupr_test: %.2f, auc_test: %.2f" % (
                test_acc * 100, test_aupr * 100, test_auc * 100))
            elif dataset == 'PAM':
                print("TEST: test_acc: %.2f, aupr_test: %.2f, auc_test: %.2f, auc_precision: %.2f, auc_recall: %.2f, auc_F1: %.2f\n" % (
                test_acc * 100, test_aupr * 100, test_auc * 100, test_precision * 100, test_recall * 100, test_F1 * 100))

            acc_all.append(test_acc * 100)
            auc_all.append(test_auc * 100)
            aupr_all.append(test_aupr * 100)
            if dataset == 'PAM':
                precision_all.append(test_precision * 100)
                recall_all.append(test_recall * 100)
                F1_all.append(test_F1 * 100)


            teacher.eval()
        # ---- real train -----
        for r in range(num_runs):
            experiment_id = int(SystemRandom().random() * 100000)
            if r == 0:
                print(args, experiment_id)
            seed = args.seed
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            print('we are using: {}'.format(device))

            args.classif = True
            dataset = args.dataset  # possible values: 'P12', 'P19', 'eICU', 'PAM'
            print('Dataset used: ', dataset)

            data_obj = utils.get_data(args, dataset, device, args.quantization, upsampling_batch, split_type,
                                      feature_removal_level, missing_ratio, reverse=reverse_, predictive_label=args.predictive_label)

            train_loader = data_obj["train_dataloader"]
            test_loader = data_obj["test_dataloader"]
            val_loader = data_obj["val_dataloader"]
            dim = data_obj["input_dim"]

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                n_classes = 2
            elif dataset == 'PAM':
                n_classes = 8

            # model
            if args.enc == 'mtan_enc':
                if args.self_distillation:
                    rec = models.enc_mtan_classif(
                        dim, torch.linspace(0, 1., 128), args.rec_hidden, args.embed_time, args.num_heads,
                        args.learn_emb, args.freq, device=device, n_classes=n_classes).to(device)
                else:
                    rec = models.enc_mtan_classif(
                        args.use_feat_range, torch.linspace(0, 1., 128), args.rec_hidden, args.embed_time, args.num_heads,
                        args.learn_emb, args.freq, device=device, n_classes=n_classes).to(device)
            elif args.enc == 'mtan_enc_activity':
                rec = models.enc_mtan_classif_activity(
                    args.use_feat_range, args.rec_hidden, args.embed_time,
                    args.num_heads, args.learn_emb, args.freq, device=device).to(device)

            params = (list(rec.parameters()))
            if r == 0:
                print('parameters:', utils.count_parameters(rec))
            optimizer = optim.Adam(params, lr=args.lr)

            if args.distillation:
                criterion = nn.KLDivLoss(reduction='batchmean')
            elif args.self_distillation:
                criterion_distill = nn.KLDivLoss(reduction='batchmean')
                criterion_sup = nn.CrossEntropyLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            

            if args.fname is not None:
                checkpoint = torch.load(args.fname)
                rec.load_state_dict(checkpoint['rec_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('loading saved weights', checkpoint['epoch'])

            best_val_loss = float('inf')
            total_time = 0.
            best_aupr_val = 0
            student_model_save_path = f'./model_save_path/{args.exp_name}_student.pt'
            print('\n------------------\nRUN %d: Training started\n------------------' % r)
            
            for itr in range(1, args.niters + 1):
                rec.train()
                train_loss = 0
                train_loss_sup_t = 0
                train_loss_sup_s = 0
                train_loss_distill = 0

                train_n = 0
                train_acc = 0
                start_time = time.time()
                for train_batch, label in train_loader:
                    train_batch, label = train_batch.to(device), label.to(device)
                    batch_len = train_batch.shape[0]

                    if args.self_distillation:
                        observed_data_t, observed_mask_t, observed_tp_t \
                                = train_batch[:, :, :dim].clone(), train_batch[:, :, dim:dim*2].clone(), train_batch[:, :, -1].clone()
                        observed_data, observed_mask, observed_tp \
                                = train_batch[:, :, :dim].clone(), train_batch[:, :, dim:dim*2].clone(), train_batch[:, :, -1].clone()
                        observed_mask[:,:,args.use_feat_range:]=0 # lab은 모두 mask
                        observed_data[:,:,args.use_feat_range:]=0 # 혹시 모르니 값도 0으로

                        out= rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
                        out_t = rec(torch.cat((observed_data_t, observed_mask_t), 2), observed_tp_t)

                    else:
                        observed_data, observed_mask, observed_tp \
                            = train_batch[:, :, :args.use_feat_range], train_batch[:, :, dim:dim+args.use_feat_range], train_batch[:, :, -1]
                        out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
                        if args.distillation:
                            observed_data_t, observed_mask_t, observed_tp_t \
                                = train_batch[:, :, :dim], train_batch[:, :, dim:dim*2], train_batch[:, :, -1]
                            out_t = teacher(torch.cat((observed_data_t, observed_mask_t), 2), observed_tp_t)

                    
                    if args.classify_pertp:
                        N = label.size(-1)
                        out = out.view(-1, N)
                        label = label.view(-1, N)
                        _, label = label.max(-1)
                        loss = criterion(out, label.long())
                    else:
                        ###loss = criterion(out, label)
                        if args.distillation:
                            out_with_temp = softmax_with_temperature(out, args.temperature)
                            softlabel_with_temp = softmax_with_temperature(out_t, args.temperature)
                            # KLDivLoss는 타겟으로 확률 분포를 받으므로 로그 확률로 변환해야 함
                            log_probs = torch.log(out_with_temp + 1e-10)  # 로그 0 방지를 위해 작은 상수를 더함
                            # 손실 계산
                            loss = criterion(log_probs, softlabel_with_temp.detach())
                        elif args.self_distillation:
                            out_with_temp = softmax_with_temperature(out, args.temperature)
                            softlabel_with_temp = softmax_with_temperature(out_t, args.temperature)
                            # KLDivLoss는 타겟으로 확률 분포를 받으므로 로그 확률로 변환해야 함
                            log_probs = torch.log(out_with_temp + 1e-10)  # 로그 0 방지를 위해 작은 상수를 더함
                            # 손실 계산
                            loss_distill = criterion_distill(log_probs, softlabel_with_temp.detach())
                            loss_sup_student = criterion_sup(out, label)
                            loss_sup_teacher = criterion_sup(out_t, label)

                            loss= loss_distill*args.self_distillation_lambda+loss_sup_teacher*args.self_distillation_teacher_lambda+loss_sup_student
                        else:
                            loss = criterion(out, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * batch_len
                    train_acc += torch.mean((out.argmax(1) == label).float()).item() * batch_len
                    train_n += batch_len

                    if args.self_distillation:
                        train_loss_sup_t += loss_sup_teacher.item() * batch_len
                        train_loss_sup_s += loss_sup_student.item() * batch_len
                        train_loss_distill += loss_distill.item() * batch_len

                total_time += time.time() - start_time

                # validation set
                rec.eval()
                if args.self_distillation:
                    val_loss, val_acc, val_auc, val_aupr, val_precision, val_recall, val_F1 = \
                        utils.evaluate_classifier(rec, val_loader, args=args, dim=dim, dataset=dataset, is_teacher=False, self_distillation=True)
                else:
                    val_loss, val_acc, val_auc, val_aupr, val_precision, val_recall, val_F1 = \
                        utils.evaluate_classifier(rec, val_loader, args=args, dim=dim, dataset=dataset, is_teacher=False)
                best_val_loss = min(best_val_loss, val_loss)

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    if args.self_distillation:
                        print(
                            'VALIDATION: Iter: {}, loss: {:.4f} (sup_t: {:.4f}, sup_s: {:.4f}, distill: {:.4f} ), acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.2f}, val_AUROC: {:.2f}, '
                            'val_AUPRC: {:.2f}'
                            .format(itr, train_loss / train_n, train_loss_sup_t / train_n, train_loss_sup_s / train_n, train_loss_distill / train_n, train_acc / train_n, val_loss, val_acc * 100, val_auc * 100,
                                    val_aupr * 100))
                    else:
                        print(
                            'VALIDATION: Iter: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.2f}, val_AUROC: {:.2f}, '
                            'val_AUPRC: {:.2f}'
                            .format(itr, train_loss / train_n, train_acc / train_n, val_loss, val_acc * 100, val_auc * 100,
                                    val_aupr * 100))
                elif dataset == 'PAM':
                    print(
                        'VALIDATION: Iter: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.2f}, val_AUROC: {:.2f}, '
                        'val_AUPRC: {:.2f}, val_precision: {:.2f},val_recall: {:.2f},val_F1: {:.2f},'
                        .format(itr, train_loss / train_n, train_acc / train_n, val_loss, val_acc * 100, val_auc * 100,
                                val_aupr * 100, val_precision * 100, val_recall * 100, val_F1 * 100))
                # save the best model based on 'aupr'
                if val_aupr > best_aupr_val:
                    best_aupr_val = val_aupr
                    torch.save(rec, student_model_save_path)

            print('\n------------------\nRUN %d: Training finished\n------------------' % r)

            # test set
            rec.eval()
            rec = torch.load(student_model_save_path)
            if args.self_distillation:
                test_loss, test_acc, test_auc, test_aupr, test_precision, test_recall, test_F1 = \
                    utils.evaluate_classifier(rec, test_loader, args=args, dim=dim, dataset=dataset, is_teacher=False, self_distillation=True)
            else:
                test_loss, test_acc, test_auc, test_aupr, test_precision, test_recall, test_F1 = \
                    utils.evaluate_classifier(rec, test_loader, args=args, dim=dim, dataset=dataset, is_teacher=False)
            
            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                print("TEST: test_acc: %.2f, aupr_test: %.2f, auc_test: %.2f" % (
                test_acc * 100, test_aupr * 100, test_auc * 100))
            elif dataset == 'PAM':
                print("TEST: test_acc: %.2f, aupr_test: %.2f, auc_test: %.2f, auc_precision: %.2f, auc_recall: %.2f, auc_F1: %.2f\n" % (
                test_acc * 100, test_aupr * 100, test_auc * 100, test_precision * 100, test_recall * 100, test_F1 * 100))

            acc_all.append(test_acc * 100)
            auc_all.append(test_auc * 100)
            aupr_all.append(test_aupr * 100)
            if dataset == 'PAM':
                precision_all.append(test_precision * 100)
                recall_all.append(test_recall * 100)
                F1_all.append(test_F1 * 100)

        # print mean and std of all metrics
        acc_all, auc_all, aupr_all = np.array(acc_all), np.array(auc_all), np.array(aupr_all)
        mean_acc, std_acc = np.mean(acc_all), np.std(acc_all)
        mean_auc, std_auc = np.mean(auc_all), np.std(auc_all)
        mean_aupr, std_aupr = np.mean(aupr_all), np.std(aupr_all)
        print('------------------------------------------')
        print("split:{}, set/sample-level: {}, missing ratio:{}".format(split_type, feature_removal_level, missing_ratio))
        print('args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level',
              args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level)
        print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
        print('AUROC    = %.1f +/- %.1f' % (mean_auc, std_auc))
        print('AUPRC    = %.1f +/- %.1f' % (mean_aupr, std_aupr))
        if dataset == 'PAM':
            precision_all, recall_all, F1_all = np.array(precision_all), np.array(recall_all), np.array(F1_all)
            mean_precision, std_precision = np.mean(precision_all), np.std(precision_all)
            mean_recall, std_recall = np.mean(recall_all), np.std(recall_all)
            mean_F1, std_F1 = np.mean(F1_all), np.std(F1_all)
            print('Precision = %.1f +/- %.1f' % (mean_precision, std_precision))
            print('Recall    = %.1f +/- %.1f' % (mean_recall, std_recall))
            print('F1        = %.1f +/- %.1f' % (mean_F1, std_F1))
