import time
from enum import Enum
from typing import Any, Generator

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from tqdm import tqdm


class DocURLs(Enum):
    angular = [
        "https://angular.dev/api",
        "https://angular.dev/cli",
        "https://angular.dev/errors",
        "https://angular.dev/extended-diagnostics",
        "https://angular.dev/reference/releases",
        "https://angular.dev/reference/versions",
        "https://angular.dev/reference/configs/file-structure",
        "https://angular.dev/reference/configs/workspace-config",
        "https://angular.dev/reference/configs/angular-compiler-options",
        "https://angular.dev/reference/configs/npm-packages",
        "https://angular.dev/reference/concepts",
        "https://angular.dev/guide/ngmodules",
    ]
    # TODO: add more supported doc sites
    react = [""]


class DocScraper:
    """
    A class to scrape documentation from various technology websites using Selenium and BeautifulSoup.

    Attributes:
        doc_urls (DocURLs): An enumeration of documentation URLs to scrape.
        driver (WebDriver): The Selenium WebDriver instance for browsing web pages.

    Methods:
        scrape(): Public method to scrape content from the specified documentation URLs.
        close_driver(): Closes the Selenium WebDriver instance.
    """

    def __init__(self, doc_urls: DocURLs):
        """
        Initializes the DocScraper with the specified documentation URLs.

        Args:
            doc_urls (DocURLs): An enumeration value specifying the documentation URLs to scrape.
        """
        self.doc_urls = doc_urls
        self.driver = self._initialize_driver()

    def _initialize_driver(self) -> WebDriver:
        """Initializes and returns a Selenium WebDriver instance with headless configuration."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        return driver

    def _get_subpage_links(
        self, url: str, subpage_patterns: list[str]
    ) -> list[str | None]:
        """Retrieves subpage links matching the specified patterns from the given URL."""
        self.driver.get(url)
        time.sleep(2)  # Wait for the page to load

        css_selector = ", ".join(
            [f"a[href^='{pattern}']" for pattern in subpage_patterns]
        )
        links = self.driver.find_elements(By.CSS_SELECTOR, css_selector)
        subpage_links = [link.get_attribute("href") for link in links]

        # Remove duplicates
        return list(set(subpage_links))

    def _scrape_page_content(self, url: str) -> dict[str, Any]:
        """Scrapes and returns the text content and metadata of the given URL."""
        self.driver.get(url)
        time.sleep(2)  # Wait for the page to load

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        page_content = soup.get_text()
        metadata = {
            "title": soup.title.string if soup.title else "No title",
            "url": url,
            "length": len(page_content),
        }
        return {
            "content": page_content,
            "metadata": metadata,
        }

    def _scrape_angular(self, urls: list[str]) -> Generator[dict[str, Any], None, None]:
        """Scrapes content specifically from Angular documentation URLs."""
        subpage_patterns = ["/api/", "/cli/", "/errors/", "/extended-diagnostics/"]
        yield from self._scrape(urls, subpage_patterns)

    def _scrape_react(self, urls: list[str]) -> Generator[dict[str, Any], None, None]:
        """Scrapes content specifically from React documentation URLs."""
        subpage_patterns = ["/docs/"]
        yield from self._scrape(urls, subpage_patterns)

    def _scrape(
        self, urls: list[str], subpage_patterns: list[str]
    ) -> Generator[dict[str, Any], None, None]:
        """Scrapes content generically based on the given URLs and subpage patterns."""
        for url in tqdm(urls, desc="Scraping main pages"):
            main_page_content = self._scrape_page_content(url)
            yield main_page_content

            subpage_links = self._get_subpage_links(url, subpage_patterns)
            for subpage_url in tqdm(
                subpage_links, desc="Scraping subpages", leave=False
            ):
                if subpage_url is None:
                    continue

                subpage_content = self._scrape_page_content(subpage_url)
                yield subpage_content

    def scrape(self) -> Generator[dict[str, Any], None, None]:
        """
        Public method to scrape content from the specified documentation URLs.

        Returns:
            list[str]: A list of scraped content from the specified documentation source.

        Raises:
            ValueError: If the documentation source is unsupported.
        """
        if self.doc_urls == DocURLs.angular:
            yield from self._scrape_angular(self.doc_urls.value)
        if self.doc_urls == DocURLs.react:
            yield from self._scrape_react(self.doc_urls.value)
        else:
            supported_sources = ", ".join([e.name for e in DocURLs])
            raise ValueError(
                f"Unsupported documentation source: {self.doc_urls}. Supported sources are: {supported_sources}"
            )

    def close_driver(self):
        """Closes the Selenium WebDriver instance."""
        self.driver.quit()
