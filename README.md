## Setup with Docker | (Recommended)

1. Start **Docker Desktop** and wait until it says the engine is running (`docker version` must show a *Server*).
2. Create the settings file:
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   ```notepad
   # Copy to .env and fill in. Do not commit .env.
   # JWT_SECRET: at least 32 random characters. Generate with:
   #   python -c "import secrets; print(secrets.token_urlsafe(48))"
   JWT_SECRET=
   # First start only: creates user "admin" (and "viewer") if no users exist yet. Min 10 characters.
   BOOTSTRAP_ADMIN_PASSWORD=
   BOOTSTRAP_VIEWER_PASSWORD=
   # Optional overrides (defaults in config.py):
   # ACCESS_TOKEN_MINUTES=60
   # RATE_LIMIT_REQUESTS=3
   # RATE_LIMIT_WINDOW_SECONDS=600
   # TOKEN_BUDGET=10000
   # RELEVANCE_MIN_SCORE=0.25
   # LOG_LEVEL=INFO

   ```
   
   * `JWT_SECRET`: 32+ random characters (`python -c "import secrets; print(secrets.token_urlsafe(48))"`).
   * `BOOTSTRAP_ADMIN_PASSWORD` and `BOOTSTRAP_VIEWER_PASSWORD`: 10+ characters each. They create the
     accounts **`admin`** and **`viewer`** on the first start. Afterwards delete both lines.
3. Put your PDF/DOCX files in `docs/` (`mkdir docs`), then:
   ```powershell
   docker compose up --build
   ```
   The first start downloads the models (`bge-m3`, `qwen2.5:7b`, several GB) and indexes `docs/`. Chat says
   "still starting up" until http://localhost:8000/ready returns 200. Then open http://localhost:8000.

More users: `docker compose run --rm rag-app python manage_users.py add alice --role viewer`
(also `list`, `passwd`, `remove`).

## Run locally | (Without Docker)

```bash
pip install -r requirements-dev.txt
ollama pull bge-m3 && ollama pull qwen2.5:7b
python manage_users.py add admin --role admin
python main.py        # RAG_ENV=dev: a throw-away JWT secret is generated
```
