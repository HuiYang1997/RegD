import torch
import numpy as np
import torch.nn.functional as F


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


def volume_based_distance(diff_center, diff_offsets, batch_offsets, mask=None):
    dists = F.relu(torch.abs(diff_center) + diff_offsets)
    if mask != None:
        dists = dists * mask[:, None, :]

    # normalize by the sum of the offsets
    dists = dists / batch_offsets
    # cut the element greater than 1
    dists = torch.clamp(dists, max=1)

    dists = dists.sum(dim=-1)

    return dists


def compute_nf3_ranks_transrolebox(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    offsets = class_boxes.offsets

    d_centers = centers[batch_data[:, 2]]
    d_offsets = offsets[batch_data[:, 2]]

    relation_boxes = model.get_boxes(model.relation_embeds)

    batch_relations_centers = relation_boxes.centers[batch_data[:, 1]]
    batch_relations_offsets = relation_boxes.offsets[batch_data[:, 1]]

    translated_centers = d_centers - batch_relations_centers
    translated_offsets = d_offsets + batch_relations_offsets

    diff_center = torch.tile(centers, (batch_size, 1, 1)) - translated_centers[:, None, :]
    diff_offset = torch.tile(offsets, (batch_size, 1, 1)) - translated_offsets[:, None, :]

    dists = volume_based_distance(diff_center, diff_offset, offsets[None, :, :])

    # dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 0])


def compute_nf4_ranks_transrolebox(model, batch_data, batch_size, start_index=0):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    offsets = class_boxes.offsets

    c_centers = centers[batch_data[:, 1]]
    c_offsets = offsets[batch_data[:, 1]]

    relation_boxes = model.get_boxes(model.relation_embeds)
    batch_relations_centers = relation_boxes.centers[batch_data[:, 0]]
    batch_relations_offsets = relation_boxes.offsets[batch_data[:, 0]]

    translated_centers = c_centers - batch_relations_centers
    translated_offsets = c_offsets + batch_relations_offsets

    diff_center = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    diff_offset = translated_offsets[:, None, :] - torch.tile(offsets, (batch_size, 1, 1))

    dists = volume_based_distance(diff_center, diff_offset, translated_offsets[:, None, :])

    # dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 2])
