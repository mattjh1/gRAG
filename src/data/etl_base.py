from typing import Generator, Any
from abc import ABC, abstractmethod
from langchain_core.documents import Document


class ETLBase(ABC):
    @abstractmethod
    def extract(self) -> Generator[dict[str, Any], None, None]:
        pass

    @abstractmethod
    def transform(
        self, data: Generator[dict[str, Any], None, None]
    ) -> Generator[Document, None, None]:
        pass

    @abstractmethod
    def load(self, transformed_data: Generator[Document, None, None]) -> None:
        pass

    def run(self) -> None:
        extracted_data = self.extract()
        transformed_data = self.transform(extracted_data)
        self.load(transformed_data)
