from typing import Any, Generator, Union

from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document

from data.etl_base import ETLBase
from data.scrape.extract import DocScraper, DocURLs
from data.scrape.loader import DataLoader
from data.scrape.transform import Transformer
from data.store import StoreEnum


class ScrapeETL(ETLBase):
    def __init__(self, doc_urls: DocURLs, database: StoreEnum):
        self.scraper = DocScraper(doc_urls)
        self.transformer = Transformer(database)
        self.loader = DataLoader()

    def extract(self) -> Generator[dict[str, Any], None, None]:
        return self.scraper.scrape()

    def transform(
        self, data: Generator[dict[str, Any], None, None]
    ) -> Union[list[GraphDocument], list[Document]]:
        return self.transformer.as_documents(data)

    def load(
        self,
        transformed_data: Union[list[GraphDocument], list[Document]],
    ) -> None:
        self.loader.load(transformed_data)

    def run(self) -> None:
        try:
            super().run()
        finally:
            self.scraper.close_driver()
