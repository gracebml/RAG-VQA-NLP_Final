"""
Configuration file for the RAG-enhanced VQA system
"""
import os
from pathlib import Path

# Detect if running on Kaggle
IS_KAGGLE = Path('/kaggle/working').exists()

# Base paths
if IS_KAGGLE:
    # Kaggle paths
    BASE_DIR = Path('/kaggle/working')
    CODE_DIR = BASE_DIR / "code"
    DATA_DIR = Path('/kaggle/input')
    MODELS_DIR = BASE_DIR / "models"
else:
    # Local paths
    BASE_DIR = Path(__file__).parent.parent.parent # NLP project root
    CODE_DIR = BASE_DIR / "code"
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"

# Model configurations

QWEN2VL_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
QWEN2VL_4BIT = True  # Use 4-bit quantization for T4 GPU

# Vietnamese embedding model for retrieval
VIETNAMESE_EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# RAG configuration
RETRIEVAL_METHOD = "hybrid"  # "bm25", "embedding", or "hybrid"
TOP_K_RETRIEVE = 3
BM25_K1 = 1.5
BM25_B = 0.75

# Offline retrieval threshold (only use KB results if similarity >= this threshold)
# Set to 0.0 to always include KB results, set higher (e.g., 0.6) to be more selective
OFFLINE_SIMILARITY_THRESHOLD = 0.5

# Wikipedia configuration
WIKIPEDIA_LANG = "vi"  
WIKIPEDIA_FALLBACK = True  # Always search Wikipedia (combined with offline)  

# Knowledge base paths
if IS_KAGGLE:
    # Kaggle: adjust dataset name if different
    KB_JSON_PATH = Path('/kaggle/input/knowledge-base-data/knowledge_base.json')
    
    # Vector DB can be from input dataset or working directory
    VECTOR_DB_PATH = Path('/kaggle/input/vector_db') if Path('/kaggle/input/vector_db').exists() else MODELS_DIR / "vector_db"
else:
    # Local paths
    KB_JSON_PATH = DATA_DIR / "knowledge_base.json"
    VECTOR_DB_PATH = MODELS_DIR / "vector_db"

# Generation parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 4


# Device
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    # Fallback if torch not available yet
    DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"

# Logging
LOG_LEVEL = "INFO"

