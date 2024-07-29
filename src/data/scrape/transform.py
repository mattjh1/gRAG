import hashlib
import json
from typing import Any, Generator, Union

from langchain_core.documents import Document

from data.processors import split


class Transformer:
    """
    A class to handle the transformation of content into Document instances with metadata.
    """

    @staticmethod
    def get_checksum(obj: object) -> str:
        """
        Return checksum of the specific object (will be encoded to bytes).

        Args:
            obj (object): The object to calculate the checksum for.

        Returns:
            str: The MD5 checksum of the object.
        """
        b: bytes
        if isinstance(obj, bytes):
            b = obj
        elif isinstance(obj, str):
            b = obj.encode()
        else:
            b = json.dumps(obj).encode()
        return hashlib.md5(b).hexdigest()

    def transform(
        self, content: str, metadata: dict
    ) -> Union[list[Document], Document]:
        """
        Transform content and metadata into Document instances.

        Args:
            content (str): The content to be transformed.
            metadata (dict): Metadata associated with the content.

        Returns:
            Union[list[Document], Document]: A single Document or a list of Documents.
        """
        docs = []
        if metadata["length"] > 512:
            _doc = Document(
                page_content=content,
            )
            split_contents = split(_doc)
            for part in split_contents:
                checksum = self.get_checksum(part.page_content)
                doc = Document(
                    page_content=part.page_content,
                    metadata={"content_checksum": checksum, **metadata},
                )
                docs.append(doc)
        else:
            checksum = self.get_checksum(content)
            doc = Document(
                page_content=content,
                metadata={"content_checksum": checksum, **metadata},
            )
            return doc
        return docs

    def transform_all(
        self, data: Generator[dict[str, Any], None, None]
    ) -> Generator[Document, None, None]:
        """
        Transform all content and metadata in the generator into Document instances.

        Args:
            data (Generator[dict[str, Any], None, None]): A generator yielding dictionaries with content and metadata.

        Yields:
            Generator[Document, None, None]: A generator yielding Document instances.
        """
        for item in data:
            content = item["content"]
            metadata = item["metadata"]
            documents = self.transform(content, metadata)
            if isinstance(documents, Document):
                yield documents
            else:
                for document in documents:
                    yield document
