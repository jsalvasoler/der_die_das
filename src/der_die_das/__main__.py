import click

from der_die_das.data_processing import process_data

@click.group()
def cli() -> None:
    pass

@cli.command(name="process_data", help="Run the data processing pipeling")
@click.option("--rerun_raw_to_processed", is_flag=True, help="Whether to rerun the raw to processed step")
def process_data_command(rerun_raw_to_processed: bool) -> None:
    click.echo("Running data processing pipeline")
    process_data(rerun_raw_to_processed)


if __name__ == "__main__":
    cli()