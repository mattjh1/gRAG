#!/bin/bash

# Check for required commands
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Stop the Neo4j container if it is running
echo "Stopping Neo4j..."
docker compose stop neo4j

# Ensure the backup file exists
if [ ! -f ./neo4j.dump ]; then
    echo "Backup file neo4j_backup/neo4j.dump not found. Please ensure the backup file exists and try again."
    exit 1
fi

# Move the backup file to the Neo4j backup directory
echo "Moving backup file to Neo4j backup directory..."
mkdir -p ./neo4j/backup
cp ./neo4j.dump ./neo4j/backup/

# Start the Neo4j container
echo "Starting Neo4j..."
docker compose up -d neo4j

# Wait for Neo4j to start
echo "Waiting for Neo4j to start..."
sleep 30

# Restore the database
echo "Restoring the database..."
docker exec neo4j neo4j-admin database load --from-path=/backup/neo4j.dump --overwrite-destination=true

# Confirm the database was restored
if [ $? -eq 0 ]; then
    echo "Database restored successfully!"
else
    echo "Database restore failed. Please check the logs for more information."
    exit 1
fi

echo "Neo4j backup and restore process completed successfully!"
