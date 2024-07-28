from typing import Generator

import click
from langchain_core.documents import Document
from loguru import logger

from data.scrape.extract import DocScraper, DocURLs
from data.scrape.transform import transform_all


class DataLoader:
    """
    A class to handle loading transformed data into the target storage.
    """

    def load(self, transformed_data: Generator[Document, None, None]):
        """
        Loads the transformed data into the target storage.

        Args:
            transformed_data (Generator[Document, None, None]): The generator yielding transformed content.
        """
        for i, document in enumerate(transformed_data):
            logger.info(
                f"document[{i}] :: {document.page_content}, len({document.page_content})"
            )


def pipeline(docURLs: DocURLs) -> None:
    scraper = DocScraper(docURLs)
    loader = DataLoader()
    try:
        raw_data = scraper.scrape()
        transformed_data = transform_all(raw_data)
        loader.load(transformed_data)

    finally:
        scraper.close_driver()


@click.command()
@click.argument(
    "doc_urls",
    type=click.Choice([e.name for e in DocURLs]),
    # help="Chose a supported doc sites to parse",
)
def ingest(doc_urls: str) -> None:
    try:
        doc_urls_enum = DocURLs[doc_urls]
    except KeyError:
        logger.error(f"Invalid documentation source: {doc_urls}")
        return

    pipeline(doc_urls_enum)


if __name__ == "__main__":
    ingest()
