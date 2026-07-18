# Supercomputer Analytics Agent — Hybrid Router with Persistent Chats

This revision uses two evidence routes:

- **Structured route:** validated, read-only MySQL queries for aggregation,
  filtering, ranking, comparison, and statistics.
- **Semantic route:** Qdrant vector search for descriptive and knowledge-based
  questions.

The router and SQL generator are combined into one LLM call. A second LLM call
creates the grounded final answer. SQL repair adds one extra call only when the
first SQL query fails validation or execution.

The web application now uses FastAPI with the QueryMind-style interface. Chat
conversations and messages are stored in MySQL, so follow-up questions use
saved history and chats remain available after the server restarts.

## 1. Add the datasets

Put these files inside `data/`:

- `green500_systems.csv`
- `top500_systems.csv`
- `system_details.csv`
- `top500_stat.csv`
- `country_stats.csv`

## 2. Configure the environment

```powershell
Copy-Item .env.example .env
```

Update the MySQL and Qdrant credentials in `.env`. For production, create a
separate `MYSQL_QUERY_USER` with SELECT-only access to the five data tables.

## 3. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen3:8b
```

## 4. Import MySQL data and build Qdrant embeddings

Make sure MySQL, Qdrant, and Ollama are running, then execute:

```powershell
python -m src.bootstrap
```

The bootstrap uses a SHA-256 fingerprint. It skips MySQL import and Qdrant
embedding when none of the five CSV files changed. Force a rebuild with:

```powershell
python -m src.bootstrap --force
```

The bootstrap also creates `chat_conversations` and `chat_messages`. Deleting a
chat removes its messages through a MySQL foreign key with `ON DELETE CASCADE`.

## 5. Create restricted application users

After bootstrap has created the data and chat tables, create the SELECT-only
analytics user and the chat-history CRUD user shown in the **MySQL users and
permissions** section below. Set the matching passwords in `.env`.

For local development only, you can set `MYSQL_QUERY_USER` and
`MYSQL_CHAT_USER` to the ingestion account instead.

## 6. Run the application

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Query flow

```text
User question
    -> Router + SQL planner (one LLM call)
        -> Structured: SQL safety -> read-only MySQL execution
        -> Semantic: Qdrant similarity search
    -> Evidence-grounded answer (one LLM call)
```

Greetings are answered without an LLM call. Invalid SQL is repaired once and
then validated again; unsafe SQL is never executed.

## Conversation and follow-up flow

1. A browser receives a persistent random client ID in local storage.
2. New chats create a row in `chat_conversations`.
3. User and assistant messages are stored in `chat_messages`.
4. The last messages are loaded and passed to the hybrid router for follow-up
   questions. Semantic follow-ups are rewritten into a self-contained Qdrant
   retrieval query inside the existing router call.
5. Opening a sidebar chat reloads its messages and executed SQL.
6. The trash button permanently deletes the selected chat and all its messages.

## Files replaced from the current project

- Replace `chroma_retriever.py` with `qdrant_store.py`.
- Replace the Pandas `code_executor.py` route with `mysql_store.py`.
- Replace Python AST `code_safety.py` with MySQL AST validation in
  `sql_safety.py`.
- Replace the old planner/code-generator graph with the smaller `graph.py`.
- Replace `prompts.py`, `schema_context.py`, `data_loader.py`, `llm.py`, and
  `app.py` with the revised versions.
- Add `api_schemas.py`, `chat_store.py`, `templates/index.html`, and the
  `static/` frontend files.

The old `chroma_db/` folder is no longer used.

## MySQL users and permissions

Run the following once as a MySQL administrator after changing the password:

```sql
CREATE USER IF NOT EXISTS 'supercomputer_reader'@'%'
IDENTIFIED BY 'replace_with_a_strong_password';

GRANT SELECT ON supercomputer_analytics.green500_systems
TO 'supercomputer_reader'@'%';
GRANT SELECT ON supercomputer_analytics.top500_systems
TO 'supercomputer_reader'@'%';
GRANT SELECT ON supercomputer_analytics.system_details
TO 'supercomputer_reader'@'%';
GRANT SELECT ON supercomputer_analytics.top500_stat
TO 'supercomputer_reader'@'%';
GRANT SELECT ON supercomputer_analytics.country_stats
TO 'supercomputer_reader'@'%';
```

The application also executes every generated query inside a read-only MySQL
transaction and validates its SQL syntax tree before it reaches MySQL.

Create a separate chat user after running bootstrap:

```sql
CREATE USER IF NOT EXISTS 'supercomputer_chat'@'%'
IDENTIFIED BY 'replace_with_a_strong_password';

GRANT SELECT, INSERT, UPDATE, DELETE
ON supercomputer_analytics.chat_conversations
TO 'supercomputer_chat'@'%';

GRANT SELECT, INSERT, UPDATE, DELETE
ON supercomputer_analytics.chat_messages
TO 'supercomputer_chat'@'%';
```

The browser client ID keeps local users' chat lists separate, but it is not a
replacement for authentication. Add your organization's login/session layer
before exposing this application as a public multi-user service.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest
```
