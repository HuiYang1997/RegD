#!/usr/bin/env python
import json

import numpy as np
import torch
import torch.optim as optim

from model.Elem import Elem
from model.Elbe import Elbe
from utils.data_loader import DataLoader
import logging
from tqdm import trange
import wandb
from evaluate import evaluate_volume, compute_ranks

from utils.utils import get_device
import sys

logging.basicConfig(level=logging.INFO)


def main():
    torch.manual_seed(42)
    np.random.seed(12)
    run()


def run():
    num_epochs = 5000
    wandb.init()

    dataset = wandb.config.dataset
    task = wandb.config.task
    embedding_dim = wandb.config.embedding_dim
    num_neg = wandb.config.num_neg if 'num_neg' in wandb.config else 1
    margin = wandb.config.margin if 'margin' in wandb.config else 0

    device = get_device()
    data_loader = DataLoader.from_task(task)
    train_data, classes, relations = data_loader.load_data(dataset)
    val_data = data_loader.load_val_data(dataset, classes)
    val_data['nf1'] = val_data['nf1'][:1000]
    if task == 'prediction':
        val_data['nf3'] = val_data['nf3'][:1000]
    print('Loaded data.')

    model_dict = {'elem': Elem, 'elbe': Elbe}
    model_name = wandb.config.model
    parameters = ''
    others = {}
    depth_type = wandb.config.plugin_type
    others['depth_type'] = depth_type

    # initialize model
    model = model_dict[model_name](device, classes, len(relations), embedding_dim, margin=margin,
                                       depth_type=depth_type, task=task)
    parameters += f'lr:{wandb.config.lr}, embedding_dim:{embedding_dim}, margin:{margin}'


    out_folder = f'data/{dataset}/{task}/{model.name}'

    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)
    scheduler = None
    model = model.to(device)

    # if not model.negative_sampling and task != 'old':
    sample_negatives(train_data, num_neg)

    threshold = train(model, train_data, val_data, task, len(classes), optimizer, scheduler, out_folder,
                      num_epochs=num_epochs,
                      val_freq=100, depth_type=depth_type)

    print('Computing test scores...')
    scores = evaluate_volume(dataset, task, model.name, embedding_size=model.embedding_dim,
                             parameters=parameters, best=True, split='test', others=others, threshold=threshold)
    # print('Test scores:', scores)
    wandb.finish()


def train(model, data, val_data, task, num_classes, optimizer, scheduler, out_folder, num_epochs=2000, val_freq=100,
          depth_type=None):
    model.train()
    wandb.watch(model)

    best_f1 = -1
    best_epoch = 0
    best_threshold = 0

    if task == 'inferences':
        nf = 'nf1'
    elif task == 'prediction':
        nf = 'nf3'
    else:
        raise ValueError('Unknown task.')

    try:
        for epoch in trange(num_epochs):
            try:
                loss = model(data, epoch)
            except:
                loss = model(data)

            if epoch % val_freq == 0 and val_data is not None:
                ranking, threshold, f1 = compute_ranks(model.to_loaded_model(), val_data, num_classes, nf,
                                                       model.device, threshold=None, depth_type=depth_type)
                wandb.log({'top10': ranking.top10 / len(ranking), 'top100': ranking.top100 / len(ranking),
                           'mean_rank': np.mean(ranking.ranks), 'median_rank': np.median(ranking.ranks)}, commit=False)
                if f1 >= best_f1:
                    best_f1 = f1
                    best_epoch = epoch
                    best_threshold = threshold
                    model.save(out_folder, best=True)
            wandb.log({'loss': loss})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    except KeyboardInterrupt:
        print('Interrupted. Stopping training...')

    print(f'Best epoch: {best_epoch}')
    model.save(out_folder)
    return best_threshold


def sample_negatives(data, num_neg):
    for i in range(num_neg):
        nf3 = data['nf3']
        randoms = np.random.choice(data['class_ids'], size=(nf3.shape[0], 2))
        randoms = torch.from_numpy(randoms)
        new_tails = torch.cat([nf3[:, [0, 1]], randoms[:, 0].reshape(-1, 1)], dim=1)
        new_heads = torch.cat([randoms[:, 1].reshape(-1, 1), nf3[:, [1, 2]]], dim=1)
        new_neg = torch.cat([new_tails, new_heads], dim=0)
        data[f'nf3_neg{i}'] = new_neg


if __name__ == '__main__':
    main()
