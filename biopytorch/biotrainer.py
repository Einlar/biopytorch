from biopytorch.bioconv2d import BioConv2d
from biopytorch.biolinear import BioLinear

from tqdm.auto import trange, tqdm
from typing import Callable, Union, List, Tuple
import torch
import pytorch_lightning as pl
import numpy as np
import optuna
import wandb

import logging

logger = logging.getLogger("biotrainer")


def bio_train(
    layer: Union[BioConv2d, BioLinear],
    dataset: pl.LightningDataModule,
    prev_layers: List[Callable[[torch.Tensor], torch.Tensor]] = None,
    n_epochs: int = 10,
    n_batches: int = None,
    patience: float = 1.0,
    lr_scheduler: Union[float, Tuple[float, float], Callable[[int], float]] = None,
    convergence_threshold: float = 1e-2,
    epoch_callback: Callable[[torch.Tensor], None] = None,
    trial: optuna.trial.Trial = None,
    run: wandb.sdk.wandb_run.Run = None,
    device: str = "cuda",
) -> float:
    """
    Trains a `layer` (either BioLinear or BioConv) over `dataset`. If the layer is preceded by other layers in the architecture,
    they must be provided as `prev_layers`, and will not be modified by this function.

    Parameters
    ---------
    layer : BioConv2d | BioLinear
        Layer to be trained
    dataset : pl.LightningDataModule
        DataModule containing the dataset. Must have a train_dataloader() method, returning a DataLoader for the training dataset.
    prev_layers : [function(torch.Tensor) -> torch.Tensor]
        A list of layers that are applied sequentially to each batch from the training dataset, to generate the inputs for the current layer.
    n_epochs : int
        Maximum number of epochs for training. If set to a value <= 0, training proceeds until either the `convergence_threshold` is reached,
        or the `convergence` has not decreased for `patience * batches_per_epoch` batches.
    n_batches : int
        Maximum number of batches for training. If set, it overrides n_epochs.
    patience : float
        Terminate training if the `convergence` value has not decreased for `patience * batches_per_epoch` batches.
    lr_scheduler : float | (float, float) | function(int) -> float
        Learning rate scheduler. It can be:
        - None: a default exponential decay is used, starting from 0.01 and convergint to 0.003. Equivalent to (0.01, 0.003, 1).
        - A single number: the learning rate is kept fixed to the passed value.
        - A tuple of numbers (start_lr, end_lr, speed): exponential decay starting from `start_lr` and converging to `end_lr`, according to:
            end_lr - (end_lr - start_lr) * np.exp( -speed * batch_id / batches_per_epoch )
        - A function accepting as only argument the current batch number, and returning a learning rate for that batch.
    convergence_threshold : float
        If the `convergence` value goes under this threshold at the end of an epoch, training is stopped.
    epoch_callback : function(torch.Tensor) -> None
        At the end of each epoch, `epoch_callback` is called, passing as argument the learned weights at that moment:
        `epoch_callback(layer.weight.cpu())`
    trial : optuna.trial.Trial
    run : wandb run for logging in the cloud
    device : str
        Device used for training

    Returns
    -------
    convergence : float
        Final value for the convergence. If it is close to 0 (say, < 1e-2), the layer can be considered converged.
        Otherwise, some filters have stopped updating during training, either due to a "difficult" choice
        of parameters ranking_param, delta, lebesgue_p, or a too steep/high learning rate function.
        This is highly dependent on the weights initialization too: sometimes just re-running training will
        achieve convergence.
    """

    batches_per_epoch = len(dataset.train_dataloader())

    #
    if lr_scheduler is None:  # Standard learning rate schedule
        start_lr = 0.01
        end_lr = 0.003

        lr_scheduler = lambda batch_id: end_lr - (end_lr - start_lr) * np.exp(
            -1.0 * batch_id / batches_per_epoch
        )
    else:
        try:  # Callable?
            lr = lr_scheduler(0)

            # No change needed
        except TypeError:
            try:  # Tuple?
                start_lr, end_lr, speed = (
                    lr_scheduler[0],
                    lr_scheduler[1],
                    lr_scheduler[2],
                )

                lr_scheduler = lambda batch_id: end_lr - (end_lr - start_lr) * np.exp(
                    -speed * batch_id / batches_per_epoch
                )
            except TypeError:
                lr = lr_scheduler
                lr_scheduler = lambda batch_id: lr  # Constant learning_rate

    if n_epochs > 0:
        pbar_epochs = trange(n_epochs)
    else:
        pbar_epochs = tqdm()

    pbar_batches = trange(batches_per_epoch)

    batch_id = 0

    min_convergence = np.inf
    patience_counter = 0

    while patience_counter < patience * batches_per_epoch:
        pbar_batches.reset()

        # Do a pass over the training set
        for x, y in dataset.train_dataloader():
            learning_rate = lr_scheduler(batch_id)

            x = x.to(device)
            if prev_layers is not None:
                for prev in prev_layers:
                    x = prev(x)

            convergence = layer.training_step(x, learning_rate)

            if trial is not None:
                trial.report(convergence, batch_id)

                if trial.should_prune():
                    pbar_epochs.close()
                    pbar_batches.close()
                    raise optuna.TrialPruned()

            if run is not None:
                run.log({"convergence": convergence}, step=batch_id)

            desc = f"conv: {convergence:.3f}, lr: {learning_rate:.3f}"
            pbar_epochs.set_postfix_str(desc)

            batch_id += 1
            pbar_batches.update()

            if convergence <= min_convergence:
                min_convergence = convergence
                patience_counter = 0
            else:
                patience_counter += 1

            if (n_batches is not None) and (batch_id > n_batches):
                break

        pbar_epochs.update()

        # Refresh progress bars
        pbar_batches.refresh()
        pbar_epochs.refresh()

        # Call callback at the end of each epoch
        if epoch_callback is not None:
            epoch_callback(layer.weight.cpu())

        # Do at most n_epochs (if specified)
        if (n_epochs > 0) and (pbar_epochs.n > n_epochs):
            break

        # Terminate if converged
        if np.abs(convergence) < convergence_threshold:
            pbar_epochs.set_postfix_str("CONVERGED")
            break

    pbar_epochs.close()
    pbar_batches.close()

    return convergence
