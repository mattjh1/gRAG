from typing import Generator, Any
from langchain_core.documents import Document
from data.etl_base import ETLBase
from data.scrape.extract import DocScraper, DocURLs
from data.scrape.transform import Transformer
from data.scrape.loader import DataLoader


class ScrapeETL(ETLBase):
    def __init__(self, doc_urls: DocURLs):
        self.scraper = DocScraper(doc_urls)
        self.transformer = Transformer()
        self.loader = DataLoader()

    def extract(self) -> Generator[dict[str, Any], None, None]:
        return self.scraper.scrape()

    def transform(
        self, data: Generator[dict[str, Any], None, None]
    ) -> Generator[Document, None, None]:
        return self.transformer.transform_all(data)

    def load(self, transformed_data: Generator[Document, None, None]) -> None:
        self.loader.load(transformed_data)

    def run(self) -> None:
        try:
            super().run()
        finally:
            self.scraper.close_driver()
