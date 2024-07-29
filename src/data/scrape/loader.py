from typing import Generator

from langchain_core.documents import Document
from loguru import logger


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
            # TODO: load into vector store
