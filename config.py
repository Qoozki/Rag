"""Configuration. Same settings as the original project; everything can be overridden by environment variables."""
import os


def _int(name, default):
    return int(os.getenv(name, default))


def _float(name, default):
    return float(os.getenv(name, default))


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Environment ------------------------------------------------------------------
ENV = os.getenv("RAG_ENV", "production").strip().lower()
IS_DEV = ENV in {"dev", "development", "test"}

# --- Paths / services (original names kept) ----------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_WAIT_SECONDS = _int("OLLAMA_WAIT_SECONDS", 180)   # startup waits this long for Ollama
DATA_PATH = os.getenv("DOCS_DIR", "./docs")              # uploads AND indexing use this one folder
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")
COLLECTION_NAME = "pdf_documents"
USERS_FILE = os.getenv("USERS_FILE", "./data/users.json")
SUPPORTED_EXTENSIONS = (".pdf", ".docx")

# --- Authentication (no secrets in code) -------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "")                 # required in production, >= 32 chars
SECRET_KEY = JWT_SECRET                                  # legacy alias
ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = _int("ACCESS_TOKEN_MINUTES", 60)
BOOTSTRAP_ADMIN_USERNAME = os.getenv("BOOTSTRAP_ADMIN_USERNAME", "admin")
BOOTSTRAP_ADMIN_PASSWORD = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")
BOOTSTRAP_VIEWER_USERNAME = os.getenv("BOOTSTRAP_VIEWER_USERNAME", "viewer")
BOOTSTRAP_VIEWER_PASSWORD = os.getenv("BOOTSTRAP_VIEWER_PASSWORD", "")
MIN_PASSWORD_LENGTH = 10
LOGIN_MAX_FAILURES = _int("LOGIN_MAX_FAILURES", 5)
LOGIN_WINDOW_SECONDS = _int("LOGIN_WINDOW_SECONDS", 300)
WS_AUTH_TIMEOUT = _int("WS_AUTH_TIMEOUT", 10)

# --- Limits (server-side, per user: reconnecting does not reset them) ----------------
RATE_LIMIT_REQUESTS = _int("RATE_LIMIT_REQUESTS", 3)
RATE_LIMIT_WINDOW_SECONDS = _int("RATE_LIMIT_WINDOW_SECONDS", 600)
TOKEN_BUDGET = _int("TOKEN_BUDGET", 10000)
TOKEN_BUDGET_WINDOW_SECONDS = _int("TOKEN_BUDGET_WINDOW_SECONDS", 86400)
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 25)
MAX_QUESTION_CHARS = _int("MAX_QUESTION_CHARS", 2000)

# --- Logging ----------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_JSON = _bool("LOG_JSON", False)

# Multilingual embedding model (Arabic + English, cross-lingual retrieval)
#   ollama pull bge-m3
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# LLM with solid Arabic support (llama3.2 does not officially support Arabic)
#   ollama pull qwen2.5:7b      (alternative: aya-expanse:8b)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_NUM_CTX = 8192  # Arabic uses many tokens; default 4096 would silently truncate the context

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 375

RETRIEVER_K = 4

# --- Chat memory (no extra LLM call) -----------------------------------------------
HISTORY_TURNS = 3          # past Q&A turns remembered per connection
FOLLOWUP_MAX_WORDS = 8     # a question this short is treated as a possible follow-up:
                           # the previous question is added to the search query
HISTORY_ANSWER_CHARS = 600 # old answers are cut to this length inside the prompt
# Cross-lingual similarity scores are lower than same-language ones, so keep this modest
SCORE_THRESHOLD = 0.5
# "Not covered" safety net: if even the best retrieved chunk scores below this, answer
# "I don't know" without calling the LLM. Deliberately low so it only catches clearly unrelated
# questions; set to 0 to disable. Watch the "Best retrieval score" log lines to tune it.
RELEVANCE_MIN_SCORE = _float("RELEVANCE_MIN_SCORE", 0.25)

# Original prompt; the only changes are (1) sources are cited with [S#] tags that the server verifies
# and turns into file/page references, and (2) the line that told the model to "state where it is".
prompt = (
    "You are an assistant for question-answering tasks.\n"
    "Use the following pieces of retrieved context to answer the question.\n"
    "Every piece of context starts with a source tag such as [S1]. After each fact you state, cite the tag "
    "of the context piece it came from, for example [S2]. Use only tags that appear in the context. "
    "Do NOT write file names or page numbers yourself; they are added automatically.\n"
    "Give the full answer to the question.\n"
    "ALWAYS write your answer in the same language as the question "
    "(Arabic question -> Arabic answer, English question -> English answer), DO NOT USE ANY OTHER LANGUAGE OTHER THAN THESE TWO, "
    "If the user asks multiple questions, answer the ones that are supported by the context.\n"
    "For any sub-question or topic NOT supported by the context, state that you dont know the answer.\n"
    "Do not reject the whole request if partial information is available.\n"
    "translating the relevant facts from the context when needed.\n"
    "Earlier messages in the conversation are only there to help you understand references such as \"it\" or \"the second one\"; "
    "always answer from the retrieved context below.\n"
    "IMPORTANT : If you do not know the answer, only say \"I dont know the answer to that question\" nothing extra.\n\n"
    "{context}"
    )
