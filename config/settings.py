import os
from dotenv import load_dotenv

load_dotenv()


def _clean_env_url(value: str | None, default: str) -> str:
    if not value:
        return default
    cleaned = value.strip().strip("`").strip()
    return cleaned or default


def _normalize_google_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/openai"):
        return normalized
    return f"{normalized}/openai"


# ─── Provider Selection ───
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# ─── OpenRouter Configuration ───
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = _clean_env_url(
    os.getenv("OPENROUTER_BASE_URL"), 
    "https://openrouter.ai/api/v1"
)

# ─── Google Configuration ───
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GOOGLE_BASE_URL = _normalize_google_openai_base_url(
    _clean_env_url(
        os.getenv("GOOGLE_BASE_URL"),
        "https://generativelanguage.googleapis.com/v1beta",
    )
)

# ─── Active Configuration ───
if LLM_PROVIDER == "google":
    LLM_API_KEY = GOOGLE_API_KEY
    LLM_BASE_URL = GOOGLE_BASE_URL
else:
    LLM_API_KEY = OPENROUTER_API_KEY
    LLM_BASE_URL = OPENROUTER_BASE_URL

# ─── Model Configuration ───
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "deepseek/deepseek-chat-v3-0324")
LIGHT_MODEL = os.getenv("LIGHT_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
REASONING_MODEL = os.getenv("REASONING_MODEL", "deepseek/deepseek-r1")

# Override with LLM_MODEL if set
PRIMARY_MODEL = os.getenv("LLM_MODEL", PRIMARY_MODEL)

# ─── Embedding Configuration ───
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024

# ─── Qdrant Configuration ───
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "geotech_knowledge"

# ─── Document Processing ───
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 100  # tokens
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1500"))
USE_STRUCTURE_ASSEMBLY = os.getenv("USE_STRUCTURE_ASSEMBLY", "true").lower() == "true"
DOCLING_MAX_PAGES = int(os.getenv("DOCLING_MAX_PAGES", "300"))

# ─── OCR Configuration ───
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng+chi_sim")

# ─── HK-Specific Standards ───
HK_STANDARDS = {
    "cop_foundations": "Code of Practice for Foundations 2017",
    "geoguide_1": "Geoguide 1 - Guide to Retaining Wall Design",
    "geoguide_2": "Geoguide 2 - Guide to Site Investigation",
    "geoguide_3": "Geoguide 3 - Guide to Rock and Soil Descriptions",
    "geoguide_4": "Geoguide 4 - Guide to Cavern Engineering",
    "geoguide_5": "Geoguide 5 - Guide to Slope Maintenance",
    "geoguide_7": "Geoguide 7 - Guide to Soil Nail Design and Construction",
    "geo_manual_slopes": "Geotechnical Manual for Slopes",
    "ec7": "BS EN 1997-1 Eurocode 7",
    "bs8002": "BS 8002 Earth Retaining Structures",
    "bs8004": "BS 8004 Foundations",
}
