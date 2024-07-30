from __future__ import annotations

import click

from der_die_das.data_processing import process_data
from der_die_das.evaluate import evaluate
from der_die_das.model import DEFAULT_SETTINGS
from der_die_das.train import train
from der_die_das.utils import LANGUAGES


@click.group()
def cli() -> None:
    pass


@cli.command(name="process_data", help="Run the data processing pipeline")
@click.argument("language", type=click.Choice(LANGUAGES))
@click.option("--raw", is_flag=True, help="Whether to rerun the raw to processed step")
@click.option("--split", is_flag=True, help="Whether to rerun the train test split step")
def process_data_command(language: str, *, raw: bool, split: bool) -> None:
    click.echo("Running data processing pipeline")
    process_data(language, rerun_raw_to_processed=raw, rerun_train_test_split=split)


@cli.command(name="train", help="Train the model")
@click.argument("language", type=click.Choice(LANGUAGES))
@click.option(
    "--batch_size",
    type=int,
    help="Batch size for training",
    default=DEFAULT_SETTINGS["batch_size"],
    show_default=True,
)
@click.option(
    "--lr", type=float, help="The learning rate for training", default=DEFAULT_SETTINGS["lr"], show_default=True
)
@click.option(
    "--step_size",
    type=int,
    help="Step size for the learning rate scheduler",
    default=DEFAULT_SETTINGS["step_size"],
    show_default=True,
)
@click.option(
    "--gamma",
    type=float,
    help="Gamma for the learning rate scheduler",
    default=DEFAULT_SETTINGS["gamma"],
    show_default=True,
)
@click.option(
    "--num_epochs",
    type=int,
    help="Number of epochs for training",
    default=DEFAULT_SETTINGS["num_epochs"],
    show_default=True,
)
@click.option(
    "--val_size",
    type=float,
    help="Size of the validation set as a fraction of the training set." " If not provided, no validation set is used",
)
@click.option(
    "--early_stop",
    type=int,
    help="Number of epochs to wait before stopping training if the validation loss does not decrease."
    " If not provided, early stopping is not used",
)
def train_command(
    language: str,
    batch_size: int,
    lr: float,
    step_size: int,
    gamma: float,
    num_epochs: int,
    val_size: float,
    early_stop: int | None,
) -> None:
    click.echo("Training the model")

    settings = {k: v for k, v in locals().items() if v is not None}

    train(settings)


@cli.command(name="evaluate", help="Evaluate the model")
@click.option(
    "--timestamp",
    help="Timestamp of the model to evaluate. If not provided, the latest model will be evaluated",
)
def evaluate_command(timestamp: str | None) -> None:
    click.echo("Evaluating the model")
    evaluate(timestamp)


if __name__ == "__main__":
    cli()
