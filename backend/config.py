import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
WORKSPACE = BASE.parent.parent
# Server-only configuration. Explicit process environment takes precedence.
local_config = BASE / '.env.local'
if local_config.exists():
    for line in local_config.read_text(encoding='utf-8-sig').splitlines():
        key, separator, value = line.partition('=')
        if separator and key.strip() in {'NVIDIA_API_KEY', 'NVIDIA_MODEL', 'NVIDIA_VISION_MODEL', 'LLM_PROVIDER', 'WARDROBE_OLLAMA_MODEL'}:
            os.environ.setdefault(key.strip(), value.strip())
MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_REVISION = 'main'  # Resolved to an immutable commit by build_index.py.
VERSION = 'wardrobe-rag-0.10.0-public-case'
ALPHA = 2 / 3
MIN_COVERAGE = 0.75
RRF_K = 60
OLLAMA_MODEL = os.getenv('WARDROBE_OLLAMA_MODEL', '')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY', '')
NVIDIA_MODEL = os.getenv('NVIDIA_MODEL', 'nvidia/nemotron-3.5-lightning-30b-a3b')
NVIDIA_VISION_MODEL = os.getenv('NVIDIA_VISION_MODEL', 'nvidia/nemotron-3.5-super-vl-120b-a12b')
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'nvidia' if NVIDIA_API_KEY else 'ollama' if OLLAMA_MODEL else 'extractive')
