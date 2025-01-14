# RegD: Modeling Hierarchies by Distances of Regions

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

### Dataset Preparation
1. First, unzip the `data_utils.zip` file in the project root directory.
2. Download the MCG and Hearst datasets using the following commands:
```bash
pip install gdown
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1WH2LIk2EsTe_lQ03AjCaxZ3o8fSkNt1f?usp=sharing
```
3. Move the downloaded datasets into the `data_utils` directory.

## Usage

### Training
To train the model, use the following command:
```bash
python train.py -dataset mammal -dim 2 -epoch 400 -lr 0.01 -mode box -hybrid 0 -lamda 1.0
```

### Parameters
- `dataset`: Name of the dataset to use (e.g., "mammal", "MCG", "hearst")
- `mode`: Type of region to use for embedding:
  - "box": Axis-aligned boxes
  - "ball": Spherical regions
- `hybrid`: Boundary distance calculation method:
  - 0: Use our boundary distance
  - 1: Use specific boundary distance for boxes/balls
- `lamda`: Weight for the depth distance in the energy function (default: 1.0)
- `dim`: Dimension of the embedding space (default: 2)
- `epoch`: Number of training epochs (default: 400)
- `lr`: Learning rate (default: 0.01)

For recommended hyperparameter settings and example commands, refer to `run.sh`.
