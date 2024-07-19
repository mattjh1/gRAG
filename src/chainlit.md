# Dracula expert chat bot

This is demo showcasing the power of using knowledge graph in conjunction with typical RAG. The center of the application is an ingestion pipeline that process plain text into `Langchain` `Documents` and further `GraphDocuments`. Generating `GraphDocuments` is a experimental feature contributed by `neo4j` staff to `Langchain`, it uses an `llm` and `function calling` to extract information from `Documents` that in turn enables us to save the content as a knowledge graph.

Additionally we create two indices in the graph, one for vector embeddings and one for keyword search. We save the `Documents` in the graph as nodes, we store the vector embeddings for these `Documents` on the nodes.

With this setup we end up with multiple retrievers for the RAG. Vector and keyword (hybrid search) and of course our graph retriever.
The knowledge graph contains the full contents of the novel "Dracula" by Bram Stoker, enjoy!

## Example questions

- **Describe the role of Renfield in the novel.**
- **Analyze the theme of modernity versus antiquity in "Dracula".**
- **Compare and contrast the characters of Jonathan Harker and Dr. John Seward in terms of their roles and character development.**
