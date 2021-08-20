# Task: learn a BioConv2d layer
# Objective: minimize `convergence` value (0 = fully converged)
# Parameters (optimized): learning rate
# Arguments  (fixed by user): delta, ranking_param, lebesgue_p, architecture

import sys
import os

import optuna
import wandb
import argparse
import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid
import numpy as np
from functools import partial

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
    filename="logs/01_bioconv_convergence.log",
    filemode="w",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

from biopytorch.bioconv2d import BioConv2d
from biopytorch.biotrainer import bio_train
from data import CIFAR10DataModule


class Objective:
    def __init__(
        self, args: argparse.Namespace, dataset: pl.LightningDataModule, device: str
    ) -> None:
        self.args = args
        self.device = device
        self.dataset = dataset

    def __call__(self, trial: optuna.trial.Trial) -> float:
        lebesgue_p = trial.suggest_int("lebesgue_p", 2, 8)
        ranking_param = trial.suggest_int("ranking_param", 2, 10)
        delta = trial.suggest_float("delta", 0.0, 0.4, step=0.005)

        my_layer = BioConv2d(
            3,
            96,
            kernel_size=5,
            lebesgue_p=lebesgue_p,  # args.p,
            ranking_param=ranking_param,  # args.k,
            delta=delta,  # args.delta,
            device=self.device,
        )
        logger.debug(my_layer)

        start = 0.007  # trial.suggest_float("lr_start", 1e-4, 1e-1, log=True)
        end = (
            0.0001  # start / 20  # trial.suggest_float("lr_end", 1e-5, start, log=True)
        )
        speed = 0.8

        config = {
            "trial_p": lebesgue_p,
            "trial_k": ranking_param,
            "trial_delta": delta,
            "trial_lr_start": start,
            "trial_lr_end": end,
            "trial_lr_speed": speed,
        }

        run = wandb.init(
            project="bioconv2_convergence",
            name=f"trial_{trial.number}",
            group="all_params",
            # job_type="hyperopt",
            config=config,
            reinit=True,
        )

        with run:

            def plot_filters(weights):

                random_idx = np.random.choice(len(weights), size=5, replace=False)
                grid = make_grid(
                    weights[random_idx], nrow=5, normalize=True, scale_each=True
                )
                images = wandb.Image(grid, caption="Random filters")
                wandb.log({"filters": images})

            convergence = bio_train(
                my_layer,
                self.dataset,
                n_epochs=5,
                lr_scheduler=(start, end, speed),
                device=self.device,
                trial=trial,
                run=run,
                epoch_callback=plot_filters,
            )

        return convergence


if __name__ == "__main__":
    logger = logging.getLogger("experiment")

    parser = argparse.ArgumentParser(description="Learn a BioConv2d layer")
    parser.add_argument(
        "n_trials", type=int, nargs="?", default=10, help="Number of trials"
    )

    # parser.add_argument("-p", type=int, nargs="?", help="Lebesgue p", default=3)
    # parser.add_argument("-k", type=int, nargs="?", help="Ranking param", default=2)
    # parser.add_argument(
    #    "-delta", type=float, nargs="?", help="Anti-hebbian learning value", default=0.1
    # )

    args = parser.parse_args()

    logger.info(f"Torch version: {torch.__version__}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info(f"Using {device} for training")

    cifar10 = CIFAR10DataModule()
    cifar10.setup()

    objective = Objective(args=args, dataset=cifar10, device=device)

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    batches_per_epoch = len(cifar10.train_dataloader())

    # Optuna study for hyperparameter optimization
    study_name = "01_bioconv_convergence"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=100, interval_steps=10
        ),
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    # Retrieve results
    trials = study.trials

    # Save on Wandb
    summary = wandb.init(
        project="bioconv2_convergence", name="summary", group="logging"
    )
    for step, trial in enumerate(trials):
        summary.log({"convergence": trial.value}, step=step)

        for k, v in trial.params.items():
            summary.log({k: v}, step=step)

    # Log
    best_params = study.best_params
    logger.info(best_params)

# Add wandb support + visualization [OK]
# Save found hyperparameters in a DataFrame so that they can be easily used
# Then replicate for the other experiments
