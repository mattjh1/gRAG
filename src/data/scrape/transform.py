import hashlib
import json
from typing import Generator, Tuple

from langchain_core.documents import Document

from data.processors import split


def get_checksum(obj: object) -> str:
    """Return checksum of the specific object (will be encoded to bytes)."""
    b: bytes
    if isinstance(obj, bytes):
        b = obj
    elif isinstance(obj, str):
        b = obj.encode()
    else:
        b = json.dumps(obj).encode()
    return hashlib.md5(b).hexdigest()


def transform(content: str, length: int) -> list[Document] | Document:
    docs = []
    if length > 512:
        _doc = Document(
            page_content=content,
        )
        split_contents = split(_doc)
        for part in split_contents:
            checksum = get_checksum(part.page_content)
            doc = Document(
                page_content=part.page_content,
                metadata={"content_checksum": checksum},
            )
            docs.append(doc)
    else:
        checksum = get_checksum(content)
        doc = Document(
            page_content=content,
            metadata={"content_checksum": checksum},
        )
        return doc
    return docs


def transform_all(
    data: Generator[Tuple[str, int], None, None],
) -> Generator[Document, None, None]:
    for content, length in data:
        documents = transform(content, length)
        if isinstance(documents, Document):
            yield documents
        else:
            for document in documents:
                yield document
