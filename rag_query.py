# rag_query.py
from openai import OpenAI
import numpy as np
import faiss
import pickle
from config import OPENAI_API_KEY, CHAT_MODEL
from embed_store import load_index, get_embeddings
import textwrap

client = OpenAI(api_key=OPENAI_API_KEY)

def retrieve(query, k=5):
    index, metadatas = load_index()
    q_emb = get_embeddings([query])[0].astype('float32')
    faiss.normalize_L2(q_emb.reshape(1,-1))
    D, I = index.search(np.array([q_emb]), k)
    results = []
    for idx in I[0]:
        meta = metadatas[idx]
        results.append(meta)
    return I[0], results

def build_context_text(indices, chunks_pkl="experiments/chunks.pkl"):
    import pickle
    with open(chunks_pkl, "rb") as f:
        chunks = pickle.load(f)
    ctxs = []
    for i in indices:
        c = chunks[i]
        meta = c.metadata
        url = meta.get("source", meta.get("url", "unknown"))
        ctxs.append({
            "text": c.page_content,
            "source": url
        })
    return ctxs

def generate_answer(query, temperature=0.0, k=5):
    idxs, metas = retrieve(query, k=k)
    contexts = build_context_text(idxs)
    # create a prompt that instructs to only use the context and attribute sources
    prompt = "You are an expert assistant. Use ONLY the provided contexts to answer the question. If information is not present, say 'Not found in the provided documents'. Cite sources inline.\n\n"
    for i, c in enumerate(contexts):
        prompt += f"[Source {i+1}: {c['source']}]\n{c['text']}\n\n"
    prompt += f"User question: {query}\nAnswer concisely and include source references like [Source 1]."

    # call chat model
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":"You are a helpful assistant."},
                  {"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=512
    )
    answer = resp.choices[0].message["content"]
    return answer

if __name__ == "__main__":
    q = "Does LangChain support FAISS?"
    print(generate_answer(q))
