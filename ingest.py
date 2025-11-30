# ingest.py
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import pickle
from tqdm import tqdm
from config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain.schema import Document

def clean_text(text: str) -> str:
    # normalize whitespace and remove weird unicode
    text = re.sub(r'\s+', ' ', text)
    try:
        text = text.encode('ascii', 'ignore').decode()
    except Exception:
        pass
    return text.strip()

def load_from_urls(urls, mode="single", continue_on_failure=True, show_progress=False):
    loader = UnstructuredURLLoader(urls=urls, mode=mode, continue_on_failure=continue_on_failure)
    docs = loader.load()
    # clean
    for d in docs:
        d.page_content = clean_text(d.page_content)
    return docs

def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    return chunks

if __name__ == "__main__":
    # Example usage
    urls = [
        "https://example.com/article1",
    ]
    docs = load_from_urls(urls)
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")
    # Save chunks for later
    with open("experiments/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
