# RegD: Hierarchical Data Embedding Framework

## About

RegD is a framework for embedding hierarchical data as regions in Euclidean spaces. 
This implementation is based on the [ShadowCones](https://github.com/ydtydr/ShadowCones) project.

## Installation

### Requirements
- Python >= 3.9
- PyTorch >= 1.12.1

### Setup
1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Dataset
To download the dataset, run:
```bash
pip install gdown
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1WH2LIk2EsTe_lQ03AjCaxZ3o8fSkNt1f?usp=sharing
```

## Usage

To train the model, use the following command:
```bash
python train.py -dataset mammal -dim 2 -epoch 400 -lr 0.01 -mode box -hybrid 0 -lamda 1.0
```

### Parameters
- `mode`: Type of region ("box" or "ball")
- `hybrid`: Boundary distance type (0: our boundary distance, 1: specific boundary distance for boxes/balls)
- `lamda`: Weight for the depth distance in the energy function
- `dim`: Dimension of the embedding space
- `epoch`: Number of training epochs
- `lr`: Learning rate

For recommended hyperparameter settings, refer to `run.sh`.
