import os

# from datetime import date
from typing import Generator

import click
from langchain.docstore.document import Document

from data.docusaurus import extract, transform
from data.store import get_default_store


def pipeline(pages: Generator[Document, None, None]) -> None:
    db = get_default_store()
    pages_list = list(pages)
    if docs := transform.as_graph_documents(pages_list):
        db.store_graph(docs)


@click.command()
@click.argument(
    "base_url",
    type=str,
    nargs=1,
    envvar="DOCS_BASE_URL",
)
@click.option(
    "-f",
    "--filter",
    "filter_urls",
    type=list[str],
    required=False,
    default=lambda: os.environ.get("DOCS_FILTER", [""]),
)
def ingest(base_url: str, filter_urls: list[str]) -> None:
    """Ingest files from folder as documents."""
    pipeline(extract.load_pages(base_url, filter_urls))


if __name__ == "__main__":
    ingest()
