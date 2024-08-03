# Data ingestion

ETL CLI contain two parts, get data from files or scrape tech docs sites. For starters support scrape `Angular` docs, more to come.

Extracting data from files revolve around parsing data using `tika`, splitting the results with a `semantic chunker`, leveraging embeddings. `LangChain` `LLMGraphTransformer` to transform `Documents` to `GraphDocuments` which is saved to a running `Neo4j` instance.

Scraping use a combination of `Selenium` and `BeautifulSoup`, transforming `Documents` to `GraphDocuments` is done in a similar fashion, there is also an option to save pure vector to a `redis` db instead, which form a typical vector store.

## Examples

Too see all commands see `--help`

```sh
python -m data folder --help
> Usage: python -m data folder [OPTIONS] DIRECTORY

  Ingest files from folder as documents.

Options:
  -g, --glob TEXT
  -s, --since [%Y-%m-%d]
  -e, --embed BOOLEAN
  -g, --graph BOOLEAN
  --help                  Show this message and exit.
```

```sh
python -m data scrape --help
> Usage: python -m data scrape [OPTIONS] {angular|react}

  Command-line interface for scraping and processing documentation.

  Args:     doc_urls (str): The name of the documentation source to scrape.

Options:
  -d, --database [neo4j|redis]
  --help                        Show this message and exit.
```
