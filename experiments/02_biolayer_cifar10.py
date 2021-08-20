# Task: classify CIFAR-10 with a BioConv2d layer + SGD classifier
# Objective: maximize validation accuracy
# Parameters (optimized): learning rate
# Arguments  (fixed by user): delta, ranking_param, lebesgue_p, architecture

import sys
import os

from torch.optim import optimizer

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
)
# Hack to import from parent folder

import optuna
import wandb
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.utils import make_grid
import torchsummaryX
import pytorch_lightning as pl

import numpy as np
from tqdm.auto import tqdm, trange

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
    filename="logs/02_biolayer_cifar10.log",
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

        self.in_channels = 3
        self.out_channels = 96

    def __call__(self, trial: optuna.trial.Trial) -> float:
        lebesgue_p = trial.suggest_int("lebesgue_p", 2, 8)
        ranking_param = trial.suggest_int("ranking_param", 2, 10)
        delta = trial.suggest_float("delta", 0.0, 0.4, step=0.005)
        dropout_p = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)

        my_layer = BioConv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=5,
            lebesgue_p=lebesgue_p,  # args.p,
            ranking_param=ranking_param,  # args.k,
            delta=delta,  # args.delta,
            device=self.device,
        )
        logger.debug(my_layer)

        hebbian_segment = nn.Sequential(
            my_layer, nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(96, affine=False)
        ).to(self.device)

        out_shape = hebbian_segment(
            torch.randn((1, 3, 32, 32), device=self.device)
        ).shape[1:]

        num_hidden = np.prod(out_shape)
        classifier = nn.Sequential(
            hebbian_segment,
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(num_hidden, 10),
        ).to(self.device)

        logger.debug(
            torchsummaryX.summary(
                classifier, torch.zeros((1, 3, 32, 32), device=self.device)
            )
        )
        logger.debug("Checking parameters .requires_grad:")
        for name, param in classifier.named_parameters():
            logger.debug(f"{name}.requires_grad = {param.requires_grad}")

        # ---Training (Unsupervised)--- #
        conv_lr_scheduler = (0.007, 0.0001, 0.8)

        config = {
            "trial_p": lebesgue_p,
            "trial_k": ranking_param,
            "trial_delta": delta,
            "batch_size": self.dataset.batch_size,
        }

        run = wandb.init(
            project="bio-cifar10",
            name=f"trial_unsup_{trial.number}",
            group="one_layer",
            job_type="unsupervised",
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
                n_epochs=-1,
                lr_scheduler=conv_lr_scheduler,
                device=self.device,
                run=run,
                epoch_callback=plot_filters,
            )

        # ---Training (Supervised)--- #
        config_supervised = {
            **config,
            "trial_dropout": dropout_p,
            "unsup_convergence": convergence,
            "optimizer": "SGD(momentum=0.9, nesterov=False)",
            "starting_lr": 0.05,
            "lr_scheduler": "CosineAnnealingWarmRestarts(T_0=10, T_mult=2)",
        }
        n_epochs = 150
        optimizer = optim.SGD(
            classifier.parameters(), lr=0.05, momentum=0.9, nesterov=False
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        project_name = "bio-cifar10"
        group_name = "one_layer"
        trial_name = f"trial_sup_{trial.number}"
        run2 = wandb.init(
            project=project_name,
            name=trial_name,
            group=group_name,
            job_type="supervised",
            config=config_supervised,
            reinit=True,
        )
        save_path = f"SavedModels/{project_name}/{group_name}"
        os.makedirs(save_path, exist_ok=True)

        wandb.watch(classifier, log_freq=batches_per_epoch)

        # optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
        loss_fn = nn.CrossEntropyLoss()

        p_epochs = trange(n_epochs)
        n_batches = len(self.dataset.train_dataloader())
        p_batches = trange(n_batches)

        max_val_acc = 0
        batch_id = 0
        with run2:
            for n in p_epochs:
                p_batches.reset()

                train_loss = 0.0
                train_acc = 0.0

                classifier.train()
                for x, y in self.dataset.train_dataloader():
                    x, y = x.to(self.device), y.to(self.device)

                    optimizer.zero_grad()

                    out = classifier(x)
                    loss = loss_fn(out, y)
                    loss.backward()
                    optimizer.step()

                    acc = (y == out.argmax(1)).sum().item() / out.size(0)

                    train_acc += acc
                    train_loss += loss.cpu().item()
                    p_batches.update()

                    p_batches.set_postfix_str(
                        f"loss = {train_loss / p_batches.n:.3f}, acc = {train_acc / p_batches.n * 100:.3f}"
                    )

                run2.log({"train_acc": train_acc / len(p_batches) * 100, "epoch": n})

                classifier.eval()
                with torch.no_grad():
                    val_acc = 0.0
                    for x, y in self.dataset.val_dataloader():
                        x, y = x.to(self.device), y.to(self.device)

                        out = classifier(x)

                        val_acc += (y == out.argmax(1)).sum().item() / out.size(0)

                val_acc = val_acc / len(self.dataset.val_dataloader()) * 100

                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    run2.log({"max_val_acc": max_val_acc})

                    # Save model checkpoint
                    torch.save(classifier, f"{save_path}/{trial_name}_best_ep{n}.pt")

                # scheduler.step(val_acc)
                scheduler.step()

                run2.log(
                    {"learning_rate": optimizer.param_groups[-1]["lr"], "epoch": n}
                )
                run2.log({"val_acc": val_acc, "epoch": n})
                trial.report(val_acc, n)

                if trial.should_prune():
                    p_epochs.close()
                    p_batches.close()
                    raise optuna.TrialPruned()

                p_epochs.set_postfix_str(
                    f"loss = {train_loss / len(p_batches):.3f}, acc = {train_acc / len(p_batches) * 100:.3f}, val_acc = {val_acc:.3f}"
                )
                p_epochs.refresh()
                p_batches.refresh()

        return val_acc


if __name__ == "__main__":
    logger = logging.getLogger("experiment")

    parser = argparse.ArgumentParser(description="Learn a BioConv2d layer")
    parser.add_argument(
        "n_trials", type=int, nargs="?", default=10, help="Number of trials"
    )
    parser.add_argument(
        "--data_augment",
        action="store_true",
        help="Perform data augmentation (RandomHorizontalFlip + RandomCrop to 24x24 with padding=4 of reflect)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        default=64,
        help="Batch size for supervised training",
    )
    parser.add_argument(
        "--restart_db",
        action="store_true",
        help="Remove results from previous Optuna study",
    )

    # parser.add_argument("-p", type=int, nargs="?", help="Lebesgue p", default=3)
    # parser.add_argument("-k", type=int, nargs="?", help="Ranking param", default=2)
    # parser.add_argument(
    #     "-delta", type=float, nargs="?", help="Anti-hebbian learning value", default=0.1
    # )

    args = parser.parse_args()

    logger.info(f"Torch version: {torch.__version__}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info(f"Using {device} for training")

    data_augmentation_transform = None
    if args.data_augment:
        data_augmentation_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(
                    32, padding=4, padding_mode="reflect"
                ),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    cifar10 = CIFAR10DataModule(
        batch_size=args.batch_size, train_data_transform=data_augmentation_transform
    )
    cifar10.setup()

    logger.debug(f"shape x: {cifar10.train_dataset[0][0].shape}")

    objective = Objective(args=args, dataset=cifar10, device=device)

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    batches_per_epoch = len(cifar10.train_dataloader())

    # Optuna study for hyperparameter optimization
    study_name = "02_biolayer_cifar10"

    if args.restart_db:
        if os.path.exists(f"{study_name}.db"):
            os.remove(f"{study_name}.db")

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    # Log
    best_params = study.best_params
    logger.info(best_params)

# Add wandb support + visualization [OK]
# Save found hyperparameters in a DataFrame so that they can be easily used
# Then replicate for the other experiments
