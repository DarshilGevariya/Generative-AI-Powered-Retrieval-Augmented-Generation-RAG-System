# config.py
import os

# OpenAI API key (set as env var OPENAI_API_KEY) or paste here (not recommended)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding model (OpenAI)
EMBEDDING_MODEL = "text-embedding-3-large"  # pick available model

# Chat model
CHAT_MODEL = "gpt-4o-mini"  # replace with a model you have access to

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# FAISS index path
FAISS_INDEX_PATH = "experiments/faiss_index.index"
METADATA_PATH = "experiments/metadata.pkl"

# Audio / signal params
SAMPLE_RATE = 1_000_000  # 1 MHz typical for ultrasonic; tune as needed
