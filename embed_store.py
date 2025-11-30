# embed_store.py
import os
import pickle
from openai import OpenAI
import numpy as np
import faiss
from config import OPENAI_API_KEY, EMBEDDING_MODEL, FAISS_INDEX_PATH, METADATA_PATH
from tqdm import tqdm

# Simple wrapper to get embeddings from OpenAI
def get_embeddings(texts, batch_size=16):
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vecs = [e.embedding for e in resp.data]
        all_vecs.extend(vecs)
    return np.array(all_vecs, dtype='float32')

def build_faiss_index(chunks, dim=None, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    print("Computing embeddings...")
    embs = get_embeddings(texts)
    if dim is None:
        dim = embs.shape[1]
    # build index (Flat L2 or IndexHNSWFlat)
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors (we'll normalize)
    # normalize
    faiss.normalize_L2(embs)
    index.add(embs)
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadatas, f)
    print(f"Saved index to {index_path} with {index.ntotal} vectors")
    return index

def load_index(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadatas = pickle.load(f)
    return index, metadatas

if __name__ == "__main__":
    import pickle
    with open("experiments/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    build_faiss_index(chunks)
