import click

from data.folder.__main__ import folder
from data.scrape.__main__ import scrape


@click.group()
def cli():
    """ "Consolidate CLIs"""
    pass


cli.add_command(folder)
cli.add_command(scrape)


if __name__ == "__main__":
    cli()
