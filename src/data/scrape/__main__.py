import click
from loguru import logger

from data.scrape.extract import DocURLs
from data.scrape.scrape_etl import ScrapeETL
from data.store import StoreEnum


def pipeline(docURLs: DocURLs, database: StoreEnum) -> None:
    etl = ScrapeETL(docURLs, database)
    etl.run()


@click.command()
@click.argument(
    "doc_urls",
    type=click.Choice([e.name for e in DocURLs]),
)
@click.option(
    "-d",
    "--database",
    "database",
    type=click.Choice([e.value for e in StoreEnum]),
    default=StoreEnum.neo4j,
)
def scrape(doc_urls: str, database: str) -> None:
    """
    Command-line interface for scraping and processing documentation.

    Args:
        doc_urls (str): The name of the documentation source to scrape.
    """
    try:
        doc_urls_enum = DocURLs[doc_urls]
    except KeyError:
        logger.error(f"{doc_urls} is not supported")
        return
    try:
        db_enum = StoreEnum[database]
    except KeyError:
        logger.error(f"{database} is not supported")
        return

    pipeline(doc_urls_enum, db_enum)


if __name__ == "__main__":
    scrape()
