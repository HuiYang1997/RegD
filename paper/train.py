#!/usr/bin/env python
# coding: utf-8
import math
import os
import torch
import numpy as np
from data_utils.data_handler import DataHandler
from data_utils.relations import Relations
import argparse
import time
from BoxDistBoundary import depthBox
import torch.optim as optim
from best_threshold import *
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type('torch.DoubleTensor')
# torch.autograd.set_detect_anomaly(True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, model, model_eval, optimizer, train_loader, val_pos_loader, val_neg_loader, flag_best_threshold=0):
    loss_func = {'vanilla': model.loss, 'contrastive': model.loss_cross}[args.loss_type]
    best_metric = {'precision': -1.0, 'recall': -1.0, 'f1': -1.0, 'threshold': args.margin}
    epoch_best = -1.0
    for epoch in tqdm(range(args.epoch)):
        time1 = time.time()
        model.train()
        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            data = data.to(device)
            chunk_size = data.size(-1) // 2  # can be a hyper-parameter
            optimizer.zero_grad()
            energy0 = model(data[..., :chunk_size], reverse=False)
            energy1 = model(data[..., chunk_size:], reverse=True)
            loss = loss_func(energy0) + loss_func(energy1)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= (len(train_loader.dataset) / (2. * args.neg_size))
        if (epoch + 1) == args.epoch or (epoch + 1) % args.eval_freq == 0:
            if args.debug:
                print(f"epoch {epoch} training time: {'%.2f' % (time.time() - time1)}")
            time2 = time.time()
            if not flag_best_threshold:
                val_triplet = evaluate(model, val_pos_loader, val_neg_loader, method=args.eval_method,
                                       threshold=args.margin)
                val_triplet['threshold'] = args.margin
            else:
                val_triplet = best_threshold(model, val_pos_loader, val_neg_loader, method=args.eval_method)

            if val_triplet['f1'] > best_metric['f1']:
                best_metric = val_triplet
                epoch_best = epoch
                model_eval.emb.weight.data.copy_(model.emb.weight.data)

                # Iterate through each layer in 'denominator' (which is a Sequential module)
                for src_layer, tgt_layer in zip(model.denominator, model_eval.denominator):
                    # Check if the layer is a Linear layer (i.e., it has 'weight' and 'bias')
                    if isinstance(src_layer, torch.nn.Linear):
                        tgt_layer.weight.data.copy_(src_layer.weight.data)
                        tgt_layer.bias.data.copy_(src_layer.bias.data)

            if args.debug:
                print(f"epoch {epoch} evaluate time: {'%.2f' % (time.time() - time2)}")
                print(f"loss: {'%.2f' % epoch_loss}, val: {best_metric}")

    best_metric['epoch_best'] = epoch_best
    print(f"best val: {best_metric}")
    return model_eval, best_metric['threshold']


def best_threshold(model, pos_loader, neg_loader, method="partial"):
    scores = []
    labels = []
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(pos_loader, 0):
            parents, children = data[0], data[1]
            parents_emb = model.emb(parents.to(device))
            children_emb = model.emb(children.to(device))
            if method == "partial":
                scores.append(model.partial(parents_emb, children_emb))
            else:
                scores.append(model.energy(parents_emb, children_emb))
            labels.append(torch.ones(len(parents)))

        for i, data in enumerate(neg_loader, 0):
            parents, children = data[0], data[1]
            parents_emb = model.emb(parents.to(device))
            children_emb = model.emb(children.to(device))
            if method == "partial":
                scores.append(model.partial(parents_emb, children_emb))
            else:
                scores.append(model.energy(parents_emb, children_emb))
            labels.append(torch.zeros(len(parents)))

    scores = torch.cat(scores).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    best_threshold, best_f1, precision, recall = find_best_threshold_with_scipy(scores, labels)
    return {"threshold": float('%.3f' % best_threshold),
            'precision': float('%.2f' % precision), 'recall': float('%.2f' % recall), 'f1': float('%.2f' % best_f1)}


def evaluate(model, pos_loader, neg_loader, method="partial", threshold=0., ratio=1):
    tp = 0.
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(pos_loader, 0):
            parents, children = data[0], data[1]
            parents_emb = model.emb(parents.to(device))
            children_emb = model.emb(children.to(device))
            if method == "partial":
                tp += (model.partial(parents_emb, children_emb) <= threshold).sum()
            else:
                tp += (model.energy(parents_emb, children_emb) <= threshold).sum()
        fn = len(pos_loader.dataset) - tp
        fp = 0.
        for i, data in enumerate(neg_loader, 0):
            parents, children = data[0], data[1]
            parents_emb = model.emb(parents.to(device))
            children_emb = model.emb(children.to(device))
            if method == "partial":
                fp += (model.partial(parents_emb, children_emb) <= threshold).sum()
            else:
                fp += (model.energy(parents_emb, children_emb) <= threshold).sum()
    precision = 100 * tp / (tp + fp + 1e-6)
    recall = 100 * tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {'precision': float('%.1f' % precision), 'recall': float('%.1f' % recall), 'f1': float('%.1f' % f1)}


def argument_parser():
    parser = argparse.ArgumentParser(description='RegD')
    ############# Dataset configurations
    parser.add_argument('-dataset', type=str, choices=['mammal', 'noun', 'MCG', 'hearst'], default='mammal',
                        help='dataset (mammal | verb | noun)')
    parser.add_argument('-train_non_basic_percent', type=int, choices=[0, 1, 2, 5, 10, 25, 50, 90],
                        default=0, help="percentage of non basic edges in training set: (0 | 10 | 25 | 50 | 90)")
    ############# model configurations
    parser.add_argument('-dim', type=int, default=2, help="dimension of hyperbolic embeddings")
    parser.add_argument('-margin', type=float, default=0.001,
                        help="margin in energy function, measures how far to push negatives")
    parser.add_argument('-depth_type', type=str, choices=['norm', 'inner'], default='norm',
                        help='depth  type in depthBox model: (norm | inner)')
    ############# train/eval configurations
    parser.add_argument('-loss_type', type=str, choices=['vanilla', 'contrastive'], default='contrastive',
                        help='loss type for training cone model: (vanilla | contrastive)')
    parser.add_argument('-lr', type=float, default=0.01, help="learning rate for optimization")
    parser.add_argument('-weight_decay', type=float, default=0.0, help="weight_decay for optimization")
    parser.add_argument('-epoch', type=int, default=300, help="training epochs")
    parser.add_argument('-neg_size', type=int, default=10, help="negative sampling size")
    parser.add_argument('-batch_size', type=int, default=16, help="training batch_size")
    parser.add_argument('-neg_sampl_strategy', type=str, choices=['true_neg', 'all'], default='all',
                        help='all or non-connected nodes used for negative sampling')
    parser.add_argument('-where_not_to_sample', type=str, choices=['ancestors', 'children', 'both'], default='children',
                        help='where_not_to_sample')
    parser.add_argument('-eval_method', type=str, choices=['partial', 'energy'], default='partial',
                        help='eval_method method: (partial | energy)')
    parser.add_argument('-eval_freq', type=int, default=20, help="eval_freq during training")
    parser.add_argument('-save_model', type=bool, default=False, help="whether to save model")
    ############# debug configurations
    parser.add_argument('-seed', type=int, default=43, help="random seed for reproducing results")
    parser.add_argument('-debug', type=int, default=1, help="debug mode")
    parser.add_argument('-grid_search', type=int, default=0, help="debug mode")
    parser.add_argument('-mode', type=str, default='ball', choices=['box', 'ball'], help="geo_obj_mode")
    parser.add_argument('-add_denominator', type=str, default='non', choices=['in', 'out', 'non'],
                        help="adding the denominators")
    parser.add_argument("-use_threshold", type=int, default=1, help="whether to use threshold")
    parser.add_argument("-hybrid", type=int, default=0, help="whether to use hybrid boundary distance")
    parser.add_argument("-lamda", type=float, default=1.0, help="weight of depth distance")
    parser.add_argument("-p", type=int, default=2, help="p of Lp norm")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    set_seed(args.seed)
    # args.debug = False if args.debug == 0 else True
    if args.dataset in ['MCG', 'hearst']:
        args.data_dir = f'data_utils/data/{args.dataset}/{args.dataset}_closure.tsv'
    else:
        args.data_dir = f'data_utils/data/maxn/{args.dataset}_closure.tsv'
    ### Load data
    train_data_dir = args.data_dir + f".train_{args.train_non_basic_percent}percent"
    val_pos_data_dir = args.data_dir + ".valid"
    val_neg_data_dir = args.data_dir + ".valid_neg"
    test_pos_data_dir = args.data_dir + ".test"
    test_neg_data_dir = args.data_dir + ".test_neg"
    train_data = Relations(train_data_dir, reverse=False)
    val_data = (Relations(val_pos_data_dir, reverse=False), Relations(val_neg_data_dir, reverse=False))
    test_data = (Relations(test_pos_data_dir, reverse=False), Relations(test_neg_data_dir, reverse=False))
    data_handler = DataHandler(train_data=train_data, val_data=val_data, test_data=test_data,
                               num_negative=args.neg_size,
                               batch_size=args.batch_size, neg_sampl_strategy=args.neg_sampl_strategy,
                               where_not_to_sample=args.where_not_to_sample, num_workers=0
                               )

    ###########################
    ### Initialize model
    ###########################
    size = len(data_handler.indices_set)

    # load data
    data_handler.prepare_train_data()
    train_loader = data_handler.prepare_train_loader()
    val_pos_loader, val_neg_loader, test_pos_loader, test_neg_loader = data_handler.prepare_val_test_loader()

    if args.debug:
        print("number of nodes", size)
    print("model dim", args.dim)
    print("non basic percentage", args.train_non_basic_percent)

    if args.grid_search:
        log_file = f"results_ours/{args.dataset}_train_{args.train_non_basic_percent}percent_{args.loss_type}_{args.eval_method}_{args.mode}.csv"
    else:
        log_file = f"normal.log"

    depthbox = depthBox(num_entities=size, dim=args.dim, margin=args.margin, depth_type=args.depth_type,
                        device=device, mode=args.mode, add_denominator=args.add_denominator, hybrid=args.hybrid,
                        lamda=args.lamda, p=args.p)

    print(f"dim: {depthbox.dim}, margin: {depthbox.margin}")
    for name, param in depthbox.emb.named_parameters():
        print(name, param.size())

    depthbox_eval = depthBox(num_entities=size, dim=args.dim, margin=args.margin, depth_type=args.depth_type,
                             device=device, mode=args.mode, add_denominator=args.add_denominator, hybrid=args.hybrid,
                             lamda=args.lamda, p=args.p)

    print(args)
    optimizer = optim.Adam(depthbox.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    depthbox_eval, threshold = train(args, depthbox, depthbox_eval, optimizer, train_loader, val_pos_loader,
                                     val_neg_loader, flag_best_threshold=args.use_threshold)
    test_triplet = evaluate(depthbox_eval, test_pos_loader, test_neg_loader, method=args.eval_method,
                            threshold=threshold)

    print({'test precision': test_triplet['precision'], 'recall': test_triplet['recall'],
           'f1': test_triplet['f1']})
    with open(log_file, 'a') as f:
        f.write(f"test precision: {test_triplet['precision']}, "
                f"recall: {test_triplet['recall']}, "
                f"f1: {test_triplet['f1']}, "
                f"{args} \n ")

    # save model
    if args.save_model:
        if not os.path.exists("trained_models"):
            os.makedirs("trained_models")
        torch.save({"vocab": data_handler.kv.index2word, "model_weight": hyp_cone_eval.state_dict()},
                   f"trained_models/mammal_with_vocab_epoch_{args.epoch}_model_{args.model}.pth")
