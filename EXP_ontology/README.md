Here is the code for integrating RegD into ontology embeddings.

This repository is the official implementation of the paper [Dual Box Embeddings for the Description Logic EL++](https://arxiv.org/abs/2301.11118).

## Requirements

This implementation requires a working installation of PyTorch 1.12.0 and following Python packages: `joblib==1.1.0`, `numpy==1.22.3`, `tqdm==4.64.0`, `wandb==0.13.9`.

## Data

To obtain the data, unzip `data.zip`:
```sh
unzip data.zip
```

## Training
run the following command:
```sh
python run_all_sweeps.py
```
