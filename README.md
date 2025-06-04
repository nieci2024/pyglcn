# Semi-supervised Learning with Graph Learning-Convolutional Networks in PyTorch

This repository provides a PyTorch implementation of Graph Learning-Convolutional Networks (GLCN) for the task of semi-supervised node classification. This work is based on the paper:

Bo Jiang, Ziyan Zhang, Doudou Lin, Jin Tang, Bin Luo, [Semi-supervised Learning with Graph Learning-Convolutional Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) (CVPR 2019).

The graph convolution component used in this implementation is inspired by the work of Thomas N. Kipf and Max Welling: [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017).

This PyTorch version aims to replicate the model and experiments presented in the original TensorFlow implementation.

## Introduction

This repository contains the PyTorch code for the GLCN model, demonstrated on the Cora and Citeseer datasets. An implementation of a standard Graph Convolutional Network (GCN) is also included for comparison purposes.

The GLCN model distinctively learns a graph structure tailored to the specific semi-supervised task while simultaneously performing graph convolutions for node classification. This adaptive graph learning can be beneficial when the provided graph structure is noisy, incomplete, or not optimal for the task at hand.

## Requirements

The codebase is implemented in Python. The primary requirements are:

* PyTorch (version 1.8+ recommended)
* SciPy
* NumPy

You can install the necessary packages using pip:
```bash
pip install torch scipy numpy
```

You might also want to create a `requirements.txt` file based on your specific environment.

## Directory Structure

The project is structured as follows:

```
pyglcn/
│  README.md
│
├─data/
│  ├─citeseer/
│  │      adj.mat
│  │      feature.mat
│  │      label.mat
│  │      test.mat
│  │
│  └─cora/
│          adj.mat
│          feature.mat
│          label.mat
│
└─glcn/  # Main PyTorch code for GLCN/GCN
    │  inits.py         # Weight initialization helper 
    │  layers.py        # SparseGraphLearn and 
    │  metrics.py       # Masked loss and accuracy 
    │  models.py        # SGLCN and GCN model definitions
    │  run_citeseer.py  # Script to run experiments on 
    │  run_cora.py      # Script to run experiments on Cora
    │  train.py         # Main training script
    │  utils.py         # Data loading and preprocessing 
    │  __init__.py      # Package initializer
```

## Running the Demo

To run the GLCN model on the Cora dataset using PyTorch (assuming you are in the `pyglcn` directory):

```bash
python run_cora.py
```

Similarly, for the Citeseer dataset:
```bash
python run_citeseer.py
```

You can switch between `sglcn` and `gcn` models and adjust hyperparameters by modifying the respective run scripts (`glcn/run_cora.py` or `glcn/run_citeseer.py`). These scripts typically set flags or parameters that are passed to the `glcn/train.py` script. For example, you might find sections in these run scripts to configure:

```python
# Example of how model selection might be handled in glcn/run_cora.py or glcn/run_citeseer.py

# --- To run SGLCN ---
# flags.model = "sglcn"
# # ... other SGLCN specific flags ...

# --- OR: To run GCN ---
# flags.model = "gcn"
# flags.lr2 = 0.01         # GCN learning rate
# flags.hidden_gcn = 16    # GCN hidden layer units
# flags.weight_decay = 5e-4  # GCN L2 regularization coefficient
# flags.epochs = 200       # GCN typically converges faster
```

(Please adapt the example above to how model selection and hyperparameter settings are actually implemented in your `run_*.py` scripts.)

## Data

The input data for the models primarily consists of three parts, usually stored in `.mat` files within the `data/cora` or `data/citeseer` subdirectories:

* **Feature matrix (`feature.mat`)**: An $N \times P$ sparse matrix, where $N$ is the number of nodes and $P$ is the feature dimension of each node.
* **Adjacency matrix (`adj.mat`)**: An $N \times N$ sparse matrix, representing the graph structure.
* **Label matrix (`label.mat`)**: An $N \times C$ one-hot encoded matrix, where $C$ is the number of classes.

The Cora and Citeseer datasets are commonly used examples. If you use the Planetoid splits (as in the original GLCN TensorFlow implementation), ensure your `glcn/utils.py` correctly loads and processes them.

To use your own dataset:
1. Format your data into the `.mat` files described above.
2. Place them in a new subdirectory within the `data/` directory (e.g., `data/mydataset/`).
3. Update the data loading functions in `glcn/utils.py` to handle your new dataset, including class count and any specific preprocessing steps.
4. Create a new run script (e.g., `glcn/run_mydataset.py`) or modify an existing one to point to your dataset name and set appropriate hyperparameters.

## Models

This PyTorch implementation includes:

### SGLCN (Semi-supervised Graph Learning-Convolutional Network)
The SGLCN model features:
1. **Sparse Graph Learning Layer**: Learns an explicit graph structure $S$ from node features. It typically computes node representations $H=XW_0$ and then learns edge weights for $S$ based on these representations.
2. **Graph Convolution Layers**: Standard GCN layers are then applied using the original node features $X$ and the learned graph $S$ for node classification.

Optimization involves two loss components:
* `loss1`: Guides the graph learning (e.g., graph smoothing, sparsity terms, L2 on SGL parameters).
* `loss2`: Standard cross-entropy classification loss for the GCN part (with L2 on GCN parameters).

### GCN (Graph Convolutional Network)
A standard GCN model (Kipf & Welling, 2017) is also implemented for comparison. It uses the preprocessed input adjacency matrix directly.

## Cite

If you use this GLCN model or code in your research, please cite the original paper:

```
@inproceedings{jiang2019semi,
  title={Semi-supervised learning with graph learning-convolutional networks},
  author={Jiang, Bo and Zhang, Ziyan and Lin, Doudou and Tang, Jin and Luo, Bin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11313--11320},
  year={2019}
}
```
