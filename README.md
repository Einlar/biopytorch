# BioPyTorch
PyTorch implementation of the learning rule proposed in "Unsupervised learning by competing hidden units", D. Krotov, J. J. Hopfield, 2019, https://www.pnas.org/content/116/16/7723

## Installation
Install with `pip`:
```shell
pip install biopytorch
```

## Usage
The package provides two layers, `BioLinear` and `BioConv2d`, that respectively mirror the features of `nn.Linear` and `nn.Conv2d` from PyTorch, with the added support of training with the alternative rule proposed by Krotov & Hopfield. 

They share all the same parameters of their analogues (except for `BioConv2d`, which currently does not support the use of `bias`). To execute a single update step, call the method `training_step`. 

See the example notebook in `notebooks` for more details. 

## Other files
- `dev/bioconv2d_dev.py` contains an alternative implementation of `BioConv2d` using `F.unfold`. The performance is significantly worse (especially for memory), so it should not be used in practice. However, the algorithm is easier to follow, and can be used to get a better understanding of the Krotov learning rule.
- `slides` contains a few explanatory slides
- `notebooks`: examples

## Benchmark 
TODO: Add table with the results on CIFAR-10








