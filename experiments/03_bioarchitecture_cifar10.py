# Task: classify CIFAR-10 with a BioConv2d layer + SGD classifier
# Objective: maximize validation accuracy
# Parameters (optimized): learning rate
# Arguments  (fixed by user): delta, ranking_param, lebesgue_p, architecture

import sys
import os

import optuna
import wandb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchsummaryX
import pytorch_lightning as pl

from functools import partial

import numpy as np
from tqdm.auto import tqdm, trange

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
    filename="logs/03_bioarchitecture_cifar10.log",
    filemode="w",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

from biopytorch.bioconv2d import BioConv2d
from biopytorch.biolinear import BioLinear
from biopytorch.biotrainer import bio_train
from data import CIFAR10DataModule


def get_exponential_decay_lr(start, end, speed):
    return lambda batch_id: end - (end - start) * np.exp(
        -speed * batch_id / batches_per_epoch
    )


class Objective:
    def __init__(
        self, args: argparse.Namespace, dataset: pl.LightningDataModule, device: str
    ) -> None:
        self.args = args
        self.n_layers = args.layers
        self.random = args.random
        self.device = device
        self.dataset = dataset

    def __call__(self, trial: optuna.trial.Trial) -> float:

        # Trial Parameters
        lebesgue_p = trial.suggest_int("lebesgue_p", 2, 8)
        ranking_param = trial.suggest_int("ranking_param", 2, 10)
        delta = trial.suggest_float("delta", 0.0, 0.4, step=0.005)
        dropout_p = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        train_type = trial.suggest_categorical(
            "train_type", ["sequential", "all_at_once"]
        )
        # train_type = args.train_type

        linear_lebesgue_p = 2
        linear_ranking_param = 2
        linear_delta = 0.4
        if self.n_layers == 5:
            linear_lebesgue_p = trial.suggest_int("lin_lebesgue_p", 2, 8)
            linear_ranking_param = trial.suggest_int("lin_ranking_param", 2, 10)
            linear_delta = trial.suggest_float("lin_delta", 0.0, 0.4, step=0.005)

        trial_BioConv2d = partial(
            BioConv2d, lebesgue_p=lebesgue_p, ranking_param=ranking_param, delta=delta
        )  # Same parameters for all the conv layers

        # ---Architecture--- #
        chs = [3, 96, 128, 192, 256, 300]
        conv_layer1 = trial_BioConv2d(chs[0], chs[1], kernel_size=5)
        hebbian_layer1 = nn.Sequential(conv_layer1, nn.ReLU(), nn.MaxPool2d(2))

        conv_layer2 = trial_BioConv2d(chs[1], chs[2], kernel_size=3)
        hebbian_layer2 = nn.Sequential(conv_layer2, nn.ReLU())

        conv_layer3 = trial_BioConv2d(chs[2], chs[3], kernel_size=3)
        hebbian_layer3 = nn.Sequential(conv_layer3, nn.ReLU(), nn.MaxPool2d(2))

        conv_layer4 = trial_BioConv2d(chs[3], chs[4], kernel_size=3)
        hebbian_layer4 = nn.Sequential(conv_layer4, nn.ReLU())

        conv_out_shape = hebbian_layer4(
            hebbian_layer3(hebbian_layer2(hebbian_layer1(torch.randn((1, 3, 32, 32)))))
        ).shape

        linear_layer5 = BioLinear(
            np.prod(conv_out_shape),
            chs[5],
            delta=linear_delta,
            ranking_param=linear_ranking_param,
            lebesgue_p=linear_lebesgue_p,
        )
        hebbian_layer5 = nn.Sequential(linear_layer5, nn.ReLU())

        full_architecture = nn.Sequential(
            *[
                hebbian_layer1,
                hebbian_layer2,
                hebbian_layer3,
                hebbian_layer4,
                hebbian_layer5,
            ][: self.n_layers]
        )

        out_shape = full_architecture(torch.randn((5, 3, 32, 32))).shape
        num_hidden = np.prod(out_shape[1:])

        # Add a BatchNorm before the classifier, and Flatten if needed
        adapter = [nn.BatchNorm2d(chs[self.n_layers], affine=False), nn.Flatten()]
        if self.n_layers == 5:  # The 5th layer's output is already flattened
            adapter = [nn.BatchNorm1d(chs[5], affine=False)]

        # ---Full network--- #
        classifier = nn.Sequential(
            full_architecture,
            *adapter,
            nn.Dropout(dropout_p),
            nn.Linear(num_hidden, 10),
        ).to(self.device)

        logger.debug(
            torchsummaryX.summary(
                classifier, torch.zeros((5, 3, 32, 32), device=self.device)
            )
        )

        logger.debug("Checking parameters .requires_grad:")
        for name, param in classifier.named_parameters():
            logger.debug(f"{name}.requires_grad = {param.requires_grad}")

        # Learning rate scheduler parameters
        conv_lr_scheduler = (0.007, 0.0001, 0.8)
        lin_lr_scheduler = (0.1, 0.005, 0.1)  # BioLinear needs a higher lr

        config = {
            "trial_p": lebesgue_p,
            "trial_k": ranking_param,
            "trial_delta": delta,
            "trial_classifier_dropout": dropout_p,
            "trial_train_type": train_type,
        }

        if self.n_layers == 5:
            config["trial_linear_p"] = linear_lebesgue_p
            config["trial_linear_k"] = linear_ranking_param
            config["trial_linear_delta"] = linear_delta

        # ---Training (Unsupervised)--- #
        # Train one at a time (first for all epochs, then second for all epochs...)
        if self.random:
            logging.info("Skipping training of unsupervised layers")
        else:
            if train_type == "sequential":
                for layer_id in range(self.n_layers):
                    logger.info(f"Training unsupervised layer #{layer_id+1}...")
                    biolayer = full_architecture[layer_id][0]
                    prev_layers = full_architecture[:layer_id]

                    convergence = bio_train(
                        biolayer,
                        self.dataset,
                        prev_layers=prev_layers,
                        n_epochs=-1,
                        lr_scheduler=conv_lr_scheduler
                        if isinstance(biolayer, BioConv2d)
                        else lin_lr_scheduler,
                        device=self.device,
                    )

                    config[f"converg{layer_id+1}"] = convergence  # Store convergence
            elif train_type == "all_at_once":  # Also SGD should be here! (or not?)
                min_convergences = [
                    np.inf,
                ] * self.n_layers
                convergences = [
                    0,
                ] * self.n_layers

                patience_counter = 0
                patience = 10

                batch_id = 0
                lr_scheduler = get_exponential_decay_lr(*conv_lr_scheduler)
                linear_lr_scheduler = get_exponential_decay_lr(*lin_lr_scheduler)
                lr_batches = [
                    0,
                ] * self.n_layers

                pbar_epochs = tqdm()
                pbar_batches = trange(batches_per_epoch)
                while patience_counter < patience:
                    pbar_batches.reset()
                    for x, y in self.dataset.train_dataloader():
                        x = x.to(self.device)

                        for layer_id in range(self.n_layers):
                            layer = full_architecture[layer_id][0]
                            learning_rate = (
                                lr_scheduler(lr_batches[layer_id])
                                if isinstance(layer, BioConv2d)
                                else linear_lr_scheduler(lr_batches[layer_id])
                            )  # Adaptive learning rate?
                            convergences[layer_id] = layer.training_step(
                                x, learning_rate
                            )
                            if (
                                convergences[layer_id] < 10
                            ):  # stop decreasing the learning rate if the conv is too high
                                lr_batches[layer_id] += 1
                            x = full_architecture[layer_id](
                                x
                            )  # get input for next layer

                        if all(
                            [
                                convergences[layer_id] >= min_convergences[layer_id]
                                for layer_id in range(self.n_layers)
                            ]
                        ):
                            patience_counter += 1
                        else:
                            min_convergences = [
                                min(convergences[layer_id], min_convergences[layer_id])
                                for layer_id in range(self.n_layers)
                            ]
                            patience_counter = 0

                        desc = ", ".join(
                            [
                                f"convg{layer_id+1}: {convergences[layer_id]:.3f}"
                                for layer_id in range(self.n_layers)
                            ]
                        )
                        pbar_batches.set_postfix_str(desc)

                        pbar_batches.update()
                    pbar_epochs.update()
                    pbar_batches.refresh()
                pbar_batches.close()

                for layer_id in range(self.n_layers):
                    config[f"converg{layer_id+1}"] = convergences[layer_id]
            else:
                raise NotImplementedError(
                    f"The train_type '{train_type}' is not implemented."
                )

        # ---Logging--- #
        project_name = "bioarchitectures-cifar10"
        trial_name = f"trial_sup_{trial.number}"
        group_name = f"layers{self.n_layers}" + ("_random" if self.random else "")

        config["optimizer"] = "SGD(momentum=0.9, nesterov=False)"
        config["starting_lr"] = 0.05
        config["lr_scheduler"] = "CosineAnnealingWarmRestarts(T_0=10, T_mult=2)"
        config["batch_size"] = self.dataset.batch_size

        n_epochs = 150

        optimizer = optim.SGD(
            classifier.parameters(), lr=0.05, momentum=0.9, nesterov=False
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        run = wandb.init(
            project=project_name,
            name=trial_name,
            group=group_name,
            config=config,
            reinit=True,
            save_code=True,
        )

        wandb.watch(classifier, log_freq=batches_per_epoch)

        # ---Training (Supervised)--- #

        # optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, "max", patience=3
        # )
        loss_fn = nn.CrossEntropyLoss()

        p_epochs = trange(n_epochs)
        n_batches = len(self.dataset.train_dataloader())
        p_batches = trange(n_batches)

        max_val_acc = 0
        batch_id = 0

        # Save model
        save_path = f"SavedModels/{project_name}/{group_name}"
        os.makedirs(save_path, exist_ok=True)

        with run:
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

                run.log({"train_acc": train_acc / len(p_batches) * 100, "epoch": n})

                classifier.eval()
                with torch.no_grad():
                    val_acc = 0.0
                    for x, y in self.dataset.val_dataloader():
                        x, y = x.to(self.device), y.to(self.device)

                        out = classifier(x)

                        val_acc += (y == out.argmax(1)).sum().item() / out.size(0)

                val_acc = val_acc / len(self.dataset.val_dataloader()) * 100

                scheduler.step()
                # scheduler.step(val_acc)

                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    run.log({"max_val_acc": max_val_acc})
                    torch.save(classifier, f"{save_path}/{trial_name}_best_ep{n}.pt")

                run.log({"learning_rate": optimizer.param_groups[-1]["lr"], "epoch": n})
                run.log({"val_acc": val_acc, "epoch": n})
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
        "--layers",
        type=int,
        choices=[1, 2, 3, 4, 5],
        nargs="?",
        default=5,
        help="Number of layers",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Do not train the unsupervised layers (benchmark)",
    )  # TODO This needs testing
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
    # parser.add_argument(
    #     "--train_type",
    #     type=str,
    #     choices=["sequential", "all_at_once"],
    #     nargs="?",
    #     default="sequential",
    #     help="Type of training. Sequential = each layer is trained separately. All_at_once = all layers are trained at the same time.",
    # )
    args = parser.parse_args()

    logging.debug(vars(args))

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

    objective = Objective(args=args, dataset=cifar10, device=device)

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    batches_per_epoch = len(cifar10.train_dataloader())

    # Optuna study for hyperparameter optimization
    study_name = f"03_bioarchitecture_cifar10_layers{args.layers}" + (
        "_random" if args.random else ""
    )

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

# Add save checkpoints!
