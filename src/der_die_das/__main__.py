import click

from der_die_das.data_processing import process_data
from der_die_das.train import train


@click.group()
def cli() -> None:
    pass


@cli.command(name="process_data", help="Run the data processing pipeling")
@click.option("--rerun_raw_to_processed", is_flag=True, help="Whether to rerun the raw to processed step")
@click.option("--rerun_train_test_split", is_flag=True, help="Whether to rerun the train test split step")
def process_data_command(*, rerun_raw_to_processed: bool, rerun_train_test_split: bool) -> None:
    click.echo("Running data processing pipeline")
    process_data(rerun_raw_to_processed=rerun_raw_to_processed, rerun_train_test_split=rerun_train_test_split)


@cli.command(name="train", help="Train the model")
def train_command() -> None:
    click.echo("Training the model")
    train()


if __name__ == "__main__":
    cli()
