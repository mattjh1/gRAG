import time
from enum import Enum

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By


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
    react = [""]


class DocScraper:
    def __init__(self, doc_urls: DocURLs):
        self.doc_urls = doc_urls
        self.driver = self.initialize_driver()

    def initialize_driver(self) -> WebDriver:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        return driver

    def get_subpage_links(
        self, driver: WebDriver, url: str, subpage_patterns: list[str]
    ) -> list[str]:
        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        # Build the CSS selector string dynamically from the subpage patterns
        css_selector = ", ".join([
            f"a[href^='{pattern}']" for pattern in subpage_patterns
        ])
        links = driver.find_elements(By.CSS_SELECTOR, css_selector)
        subpage_links = [link.get_attribute("href") for link in links]
        return list(set(subpage_links))  # Remove duplicates

    def scrape_page_content(self, driver: WebDriver, url: str) -> str:
        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        # Use BeautifulSoup to parse the page content
        soup = BeautifulSoup(driver.page_source, "html.parser")
        return soup.get_text()

    def _scrape_angular(self, urls: list[str]) -> list[str]:
        subpage_patterns = ["/api/", "/cli/", "/errors/", "/extended-diagnostics/"]
        return self._scrape_generic(urls, subpage_patterns)

    def _scrape_react(self, urls: list[str]) -> list[str]:
        subpage_patterns = ["/docs/"]
        return self._scrape_generic(urls, subpage_patterns)

    def _scrape_generic(
        self, urls: list[str], subpage_patterns: list[str]
    ) -> list[str]:
        driver = self.initialize_driver()
        try:
            all_content = []
            for url in urls:
                main_page_content = self.scrape_page_content(driver, url)
                all_content.append(main_page_content)

                subpage_links = self.get_subpage_links(driver, url, subpage_patterns)
                for subpage_url in subpage_links:
                    subpage_content = self.scrape_page_content(driver, subpage_url)
                    all_content.append(subpage_content)
            return all_content
        finally:
            driver.quit()

    def scrape(self, doc_urls: DocURLs) -> list[str]:
        if doc_urls == DocURLs.angular:
            return self.scrape_angular(doc_urls.value)
        if doc_urls == DocURLs.react:
            return self.scrape_angular(doc_urls.value)
        else:
            supported_sources = ", ".join([e.name for e in DocURLs])
            raise ValueError(
                f"Unsupported documentation source: {doc_urls}. Supported sources are: {supported_sources}"
            )
