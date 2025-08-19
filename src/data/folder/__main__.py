import os
from datetime import date
from pathlib import Path
from typing import Generator, Optional

import click
from loguru import logger

from data.folder import extract, transform
from data.graph_transformer_settings import dracula_settings, ms_graphrag_settings
from data.store import get_default_store


def pipeline(paths: Generator[Path, None, None], embed: bool, graph: bool) -> None:
    db = get_default_store()
    for path in paths:
        if graph and embed:
            if docs := transform.as_graph_documents(path, ms_graphrag_settings):
                logger.info(f"Storing {len(docs)} graph documents in db")
                db.store_graph(docs)
                transform.as_vectors_from_graph(db.embeddings)
                logger.info("Embedded Document nodes")
        elif embed:
            transform.as_vectors_from_graph(db.embeddings)
            logger.info("Embedded Document nodes")
        elif graph:
            if docs := transform.as_graph_documents(path):
                logger.info(f"Storing {len(docs)} graph documents in db")
                db.store_graph(docs)
        else:
            # docs = transform.as_graph_documents(path, ms_graphrag_settings)
            raise ValueError("You need to select to create graph, embeddings or both")


@click.command()
@click.argument(
    "directory",
    type=str,
    nargs=1,
    envvar="FOLDER_INGEST_DIR",
)
@click.option(
    "-g",
    "--glob",
    "glob",
    type=str,
    required=False,
    default=lambda: os.environ.get("FOLDER_INGEST_GLOB", "**/*.*"),
)
@click.option(
    "-s",
    "--since",
    "since",
    type=click.DateTime(formats=[extract.DATE_FORMAT]),
    required=False,
    default=lambda: os.environ.get("FOLDER_INGEST_SINCE", "0001-01-01"),
)
@click.option("-e", "--embed", type=bool, required=False, default=False)
@click.option("-G", "--graph", type=bool, required=False, default=False)
def folder(
    directory: str, glob: str, since: Optional[date], embed: bool, graph: bool
) -> None:
    """Ingest files from folder as documents."""
    since_date = since if since else date.min
    pipeline(extract.directory(directory, glob, since_date), embed, graph)


if __name__ == "__main__":
    folder()
