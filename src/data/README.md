Data ingestion

This ingestion pipeline revolve around parsing data using `tika`, splitting the results with a `semantic chunker`, leveraging embeddings. `LangChain` `LLMGraphTransformer` to

```sh
# ingest all files across directory subtree
python -m data.folder ./docs --glob '**/*.*'
```

```python
...
```
