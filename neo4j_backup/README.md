# Neo4j Backup and Restore Instructions

This backup contains a graph based on the novel Dracula by Bram Stoker for demonstration purposes. It was generated using the experimental `LLMGraphTransformer` feature of `Langchain`.

## Restore script

For a quick and automated restore, run the provided script. This will stop the Neo4j instance, move the backup file, restore the database, and restart the Neo4j instance.

```sh
./restore_neo4j.sh

```

## Manual restore instructions

Follow these instructions to manually apply backup to your local Neo4j instance

### Restore Instructions

1. **Ensure Neo4j is Stopped**:

   ```sh
   docker compose stop neo4j
   ```

2. **Move the Backup File to the Neo4j Backup Directory**:

   ```sh
   cp ./neo4j_backup/neo4j.dump ./neo4j/backup/
   ```

3. **Restore the Database**:

   ```sh
    docker exec neo4j neo4j-admin database load --from-path=/backup/neo4j.dump --overwrite-destination=true
   ```

4. **Start Neo4j**:
   ```sh
   docker compose up -d neo4j
   ```
