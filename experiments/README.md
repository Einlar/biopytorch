# Experiments

These 3 Python scripts run a hyperparameter search for BioConv2d layers on the CIFAR-10 dataset. Specifically:

- `01_bioconv_convergence.py`: seeks the best parameters $p$, $k$ and $\Delta$ for the convergence of the weights in a BioConv2d layer, when trained in isolation on CIFAR-10 samples.
- `02_biolayer_cifar10.py`: learns a two-layer classifier, consisting of a convolutional layer trained through Krotov's rule, and a SGD fully-connected layer on top of it. This mirrors the experiment done in [1] with two linear layers, to see if convolutions perform better.
- `03_bioarchitecture_cifar10.py`: generalization of the previous script, supporting a deeper architecture of up to $5$ hebbian layers (which can be specified through the argument `--layers [n]`). The order and size of layers is taken from [2], so that a performance comparison may be done.

All scripts accept a single positional argument specifying the number of trials to be made, e.g.:
```
python 01_bioconv_convergence.py 100
```
for $100$ trials.

To see other available CLI parameters, call a script with `--help`, e.g.:
```
python 01_bioconv_convergence.py --help
```

## Sources

[1] Krotov, Hopfield, "Unsupervised learning by competing hidden units", 2019

[2] Amato et al., "Hebbian Learning Meets Deep Convolutional Neural Networks", 2019