from collections import Counter

import logging
import numpy as np
import torch
import torch.nn.functional as F
import math
import datetime
from tqdm import trange

from ranking_result import RankingResult
from utils.utils import get_device
from utils.data_loader import DataLoader
from model.loaded_models import LoadedModel

from torch.nn.functional import relu, softplus

logging.basicConfig(level=logging.INFO)
int_approx = None
use_bias = None
num_components = 1
par_lambda = 1.0


def main():
    evaluate_volume('GALEN', 'prediction', model_name='boxsqel', embedding_size=200, best=True)


def evaluate_volume(dataset, task, model_name, embedding_size, parameters, best=True, split='test', others={},
                    threshold=0.0):
    global int_approx, use_bias, num_components, par_lambda
    device = get_device()
    print(f'Evaluating {model_name} on {dataset} {task} with embedding size {embedding_size}')

    model = LoadedModel.from_name(model_name, f'data/{dataset}/{task}/{model_name}', embedding_size, device, best)
    num_classes = model.class_embeds.shape[0] if model_name != 'boxel' else model.min_embedding.shape[0]

    print('Loading data')
    data_loader = DataLoader.from_task(task)
    _, classes, relations = data_loader.load_data(dataset)
    assert (len(classes) == num_classes)
    if split == 'test':
        test_data = data_loader.load_test_data(dataset, classes)
    elif split == 'val':
        test_data = data_loader.load_val_data(dataset, classes)
    else:
        raise ValueError('Unknown split.')

    if task == 'prediction':
        nf = 'nf3'
    elif task == 'inferences':
        nf = 'nf1'
    else:
        raise ValueError('Unknown task.')

    ranking, best_threshold, best_f1 = compute_ranks(model, test_data, num_classes, nf, device, use_tqdm=True, threshold = threshold, depth_type = others.get('depth_type'))

    output = f'test on inferred subsumptions \n=========\n{ranking}\n'
    print(output)

    file_name = f'results/output(RegD_EL)_{model_name}_{dataset}_{task}_PlugintBox_tmp1.txt'
    with open(file_name, 'a') as f:
        csv_output = f"{dataset},{task},{model_name},{parameters},\t,"
        if others:
            csv_output += ','.join([f'{k}:{v}' for k, v in others.items()]) + ',\t'
        csv_output += ranking.to_csv() + ",\t"
        csv_output += f'{best_threshold},\t{best_f1}' + ",\t"
        now = datetime.datetime.now()
        csv_output += f',\t{now.strftime("%d-%m-%Y %H:%M:%S")}'
        csv_output += '\n'
        f.write(csv_output)

    return ranking, best_threshold, best_f1


def dists_to_ranks(dists, targets):
    index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
    return torch.take_along_dim(index, targets.reshape(-1, 1), dim=1).flatten()


def compute_rank_roc(ranks, num_classes):
    sorted_ranks = sorted(list(ranks.keys()))
    tprs = [0]
    fprs = [0]
    tpr = 0
    num_triples = sum(ranks.values())
    num_negatives = (num_classes - 1) * num_triples
    for x in sorted_ranks:
        tpr += ranks[x]
        tprs.append(tpr / num_triples)
        fp = sum([(x - 1) * v if k <= x else x * v for k, v in ranks.items()])
        fprs.append(fp / num_negatives)

    tprs.append(1)
    fprs.append(1)
    auc = np.trapz(tprs, fprs)
    return auc


def compute_ranks(model, eval_data, num_classes, nf, device, batch_size=10, use_tqdm=False, threshold=0.0, depth_type=None):
    # start_index is used to exclude the concept classes in the ranking of link prediction tasks (i.e., r(a, b))
    if nf not in eval_data:
        raise ValueError('Tried to evaluate model on normal form not present in the evaluation data')
    eval_data = eval_data[nf]
    eval_data = eval_data.to(device)

    top1, top10, top100 = 0, 0, 0
    ranks, pos_scores, neg_scores = [], [], []
    n = len(eval_data)
    num_batches = math.ceil(n / batch_size)

    range_fun = trange if use_tqdm else range
    for i in range_fun(num_batches):
        start = i * batch_size
        current_batch_size = min(batch_size, n - start)
        batch_data = eval_data[start:start + current_batch_size, :]

        if nf == 'nf1':
            batch_ranks, batch_pos_scores, batch_neg_scores = compute_nf1_ranks(model, batch_data, current_batch_size, depth_type)
        else:
            batch_ranks, batch_pos_scores, batch_neg_scores = compute_nf3_ranks(model, batch_data, current_batch_size, depth_type)

        top1 += (batch_ranks <= 1).sum()
        top10 += (batch_ranks <= 10).sum()
        top100 += (batch_ranks <= 100).sum()
        ranks += batch_ranks.tolist()
        pos_scores += batch_pos_scores.tolist()
        neg_scores += batch_neg_scores.tolist()

    ranks_dict = Counter(ranks)
    auc = compute_rank_roc(ranks_dict, num_classes)
    best_threshold, best_precision, best_recall, best_f1 = compute_best_threshold(np.array(pos_scores),
                                                                                  np.array(neg_scores), threshold)
    print("Best threshold: ", best_threshold, "Best precision: ", best_precision, "Best recall: ", best_recall,
          "Best F1: ", best_f1)
    return RankingResult(top1.item(), top10.item(), top100.item(), ranks, auc), best_threshold, best_f1


def compute_nf1_ranks(model, batch_data, batch_size, depth_type=None):
    global par_lambda
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    offsets = class_boxes.offsets

    batch_centers = centers[batch_data[:, 0]]
    batch_offsets = offsets[batch_data[:, 0]]

    diff_center = batch_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    diff_offsets = batch_offsets[:, None, :] - torch.tile(offsets, (batch_size, 1, 1))

    if offsets.shape[1] == 1:
        offsets = softplus(offsets)

        # elem or elempp
        dist_boundary = torch.norm(diff_center, dim=-1) + diff_offsets.squeeze(-1)
        if depth_type == 'norm':
            offsets_norm = offsets.abs().squeeze(1)
            batch_offsets_norm = batch_offsets.abs().squeeze(1)

            factor = (diff_center.square() + diff_offsets.square()).sum(dim=-1)
            denominator = batch_offsets_norm[:, None] * torch.tile(offsets_norm, (batch_size, 1))
            dist_depth = factor / denominator

            dists = dist_boundary + par_lambda * dist_depth
        else:
            dists = relu(dist_boundary)
    else:
        # box embeddings
        diff = diff_center.abs() + diff_offsets
        dist_1 = torch.max(diff, dim=-1)[0]
        dist_2 = torch.norm(relu(diff), dim=-1)
        dist_boundary = torch.where(dist_1 < 0, dist_1, dist_2)

        if depth_type == 'norm':
            batch_offsets_norm = torch.norm(batch_offsets, dim=1)
            offsets_norm = torch.norm(offsets, dim=1)

            factor = (diff_center.abs() + diff_offsets.abs()).sum(dim=-1)
            denominator = batch_offsets_norm[:, None] * torch.tile(offsets_norm, (batch_size, 1))
            dist_depth = factor / denominator

            dists = dist_boundary + par_lambda * dist_depth
        else:
            dists = relu(dist_boundary)


    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c <= c

    pos_dist = dists[torch.arange(batch_size), batch_data[:, 1]]
    # pick the 10 randomly negative distances excluding the positive distances
    neg_dists = dists[torch.arange(batch_size).unsqueeze(1), torch.randint(0, dists.size(1), (batch_size, 10))]

    return dists_to_ranks(dists, batch_data[:, 1]), pos_dist.reshape(-1), neg_dists.reshape(-1)


def compute_nf3_ranks(model, batch_data, batch_size, depth_type=None):
    global par_lambda
    batch_relations = model.relation_embeds[batch_data[:, 1]]

    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    offsets = class_boxes.offsets

    batch_centers = centers[batch_data[:, 2]] - batch_relations
    batch_offsets = offsets[batch_data[:, 2]]

    diff_center = batch_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    diff_offsets = batch_offsets[:, None, :] - torch.tile(offsets, (batch_size, 1, 1))

    if offsets.shape[1] == 1:
        # offsets = softplus(offsets)

        # elem or elempp
        dist_boundary = torch.norm(diff_center, dim=-1) + diff_offsets.squeeze(-1)
        if depth_type == 'norm':
            offsets_norm = offsets.abs().squeeze(1)
            batch_offsets_norm = batch_offsets.abs().squeeze(1)

            factor = (diff_center.square() + diff_offsets.square()).sum(dim=-1)
            denominator = batch_offsets_norm[:, None] * torch.tile(offsets_norm, (batch_size, 1))
            dist_depth = factor / denominator

            dists = dist_boundary + par_lambda * dist_depth
        else:
            dists = relu(dist_boundary)
    else:
        # box embeddings
        diff = diff_center.abs() + diff_offsets
        dist_1 = torch.max(diff, dim=-1)[0]
        dist_2 = torch.norm(relu(diff), dim=-1)
        dist_boundary = torch.where(dist_1 < 0, dist_1, dist_2)

        if depth_type == 'norm':
            batch_offsets_norm = torch.norm(batch_offsets, dim=1)
            offsets_norm = torch.norm(offsets, dim=1)

            factor = (diff_center.abs() + diff_offsets.abs()).sum(dim=-1)
            denominator = batch_offsets_norm[:, None] * torch.tile(offsets_norm, (batch_size, 1))
            dist_depth = factor / denominator

            dists = dist_boundary + par_lambda * dist_depth
        else:
            dists = relu(dist_boundary)


    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c <= c

    pos_dist = dists[torch.arange(batch_size), batch_data[:, 1]]
    # pick the 10 randomly negative distances excluding the positive distances
    neg_dists = dists[torch.arange(batch_size).unsqueeze(1), torch.randint(0, dists.size(1), (batch_size, 10))]

    return dists_to_ranks(dists, batch_data[:, 0]), pos_dist.reshape(-1), neg_dists.reshape(-1)


def compute_best_threshold(pos_scores, neg_scores, threshold=None):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    print(sorted_scores)

    if isinstance(threshold, float):
        # compute f1 score for the given threshold
        tpr = np.sum(sorted_labels[sorted_scores <= threshold])
        fpr = np.sum(sorted_labels[sorted_scores <= threshold] == 0)
        precision = tpr / (tpr + fpr) if (tpr + fpr) > 0 else 0
        recall = tpr / len(pos_scores)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return threshold, precision, recall, f1

    tprs = [0]
    fprs = [0]
    tpr = 0
    fp = 0
    num_positives = len(pos_scores)
    num_negatives = len(neg_scores)
    best_threshold = None
    best_f1 = 0
    best_precision = 0
    best_recall = 0

    for i, score in enumerate(sorted_scores):
        if sorted_labels[i] == 1:
            tpr += 1
        else:
            fp += 1

        tprs.append(tpr / num_positives)
        fprs.append(fp / num_negatives)

        precision = tpr / (tpr + fp) if (tpr + fp) > 0 else 0
        recall = tpr / num_positives
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = score
            best_precision = precision
            best_recall = recall

    return best_threshold, best_precision, best_recall, best_f1


if __name__ == '__main__':
    main()
