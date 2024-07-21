import click

from der_die_das.data_processing import process_data

@click.group()
def cli() -> None:
    pass

@cli.command(name="process_data", help="Run the data processing pipeling")
def process_data_command():
    click.echo("Running data processing")
    process_data()


if __name__ == "__main__":
    cli()