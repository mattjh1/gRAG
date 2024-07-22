Data ingestion

This ingestion pipeline revolve around parsing data using `tika`, splitting the results with a `semantic chunker`, leveraging embeddings. `LangChain` `LLMGraphTransformer` to transform `Documents` to `GraphDocuments` which is saved to a running `Neo4j` instance.

```sh
# ingest all files across directory subtree
python -m data.folder ./docs --glob '**/*.*'
```

```python
...
```
