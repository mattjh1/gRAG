from typing import Generator

from langchain_community.document_loaders import DocusaurusLoader
from langchain_core.documents import Document


def get_docusaurus_client(base_url: str, filter_urls: list[str]) -> DocusaurusLoader:
    """
    Returns a DocusaurusLoader configured to scrape documentation from the specified base URL.

    Parameters:
    - base_url (str): The base URL of the Docusaurus website.
    - filter_urls (List[str]): List of URLs to filter and scrape.

    Returns:
    - DocusaurusLoader: Configured loader instance.
    """
    return DocusaurusLoader(url=base_url, filter_urls=filter_urls)


def load_pages(
    base_url: str, filter_urls: list[str]
) -> Generator[Document, None, None]:
    docusaurus_client = get_docusaurus_client(base_url, filter_urls)
    results = docusaurus_client.aload()
    yield from results
