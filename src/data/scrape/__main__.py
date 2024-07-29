import click
from loguru import logger

from data.scrape.extract import DocURLs
from data.scrape.scrape_etl import ScrapeETL


def pipeline(docURLs: DocURLs) -> None:
    etl = ScrapeETL(docURLs)
    etl.run()


@click.command()
@click.argument(
    "doc_urls",
    type=click.Choice([e.name for e in DocURLs]),
)
def scrape(doc_urls: str) -> None:
    """
    Command-line interface for scraping and processing documentation.

    Args:
        doc_urls (str): The name of the documentation source to scrape.
    """
    try:
        doc_urls_enum = DocURLs[doc_urls]
    except KeyError:
        logger.error(f"Invalid documentation source: {doc_urls}")
        return

    pipeline(doc_urls_enum)


if __name__ == "__main__":
    scrape()
