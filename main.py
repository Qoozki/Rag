# Local development runner (Docker uses the uvicorn command from the Dockerfile).
import os

import uvicorn

if __name__ == "__main__":
    os.environ.setdefault("RAG_ENV", "dev")   # allows a throw-away JWT secret locally
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
