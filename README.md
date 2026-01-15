# Generative AI–Based Retrieval-Augmented Generation (RAG) System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system to answer questions using external documents. The system retrieves relevant content from articles or text files and uses a language model to generate responses grounded in the retrieved data.

## Data Ingestion
- Built an ingestion pipeline to fetch articles from URLs and process uploaded text files  
- Cleaned and preprocessed raw text to make it suitable for downstream indexing  

## Document Processing & Embeddings
- Used LangChain’s UnstructuredURLLoader to extract text from web sources  
- Converted documents into vector representations using OpenAI embeddings  
- Stored embeddings in a vector database for efficient retrieval  

## Retrieval & Generation
- Implemented similarity search using FAISS to retrieve relevant documents  
- Integrated ChatGPT to generate answers based on retrieved context  
- Added basic attribution by linking responses back to source documents  

## Key Learnings
- Importance of clean document ingestion for reliable retrieval  
- Trade-offs between retrieval quality and generation accuracy  
- Practical experience with vector search and prompt grounding  

## Tools & Technologies
- Python  
- LangChain  
- FAISS  
- OpenAI Embeddings  
- ChatGPT

## Notes
The project focuses on building a simple and extensible RAG pipeline rather than optimizing for large-scale production use.
