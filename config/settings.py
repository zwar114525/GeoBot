import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PRIMARY_MODEL = "openai/gpt-oss-120b:free"
LIGHT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
REASONING_MODEL = "deepseek/deepseek-r1"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "geotech_knowledge"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

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
