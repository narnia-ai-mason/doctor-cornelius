# Database Migrations

This directory contains Neo4j schema migration scripts.

## Naming Convention

```
NNN_description.cypher
```

Example: `001_initial_schema.cypher`

## Workflow

1. Migrations are applied in order on startup
2. Each migration runs in a transaction
3. Applied versions are tracked in Neo4j as `(:SchemaVersion)` nodes

## Backup

Before major migrations:

```bash
neo4j-admin database dump neo4j --to-path=/backup/
```
