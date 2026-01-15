# Smartphone Tweet Sentiment Analysis

## Overview
The goal of this project was to analyze public sentiment around popular smartphone brands using Twitter data. The work involved collecting and cleaning tweets, handling multilingual text, extracting useful features, and training models to classify sentiment and study brand perception.

## Data
- Processed **12,000+ tweets** related to smartphones  
- Tweets were **manually labeled** for sentiment  
- Analysis covered **9 major smartphone brands**

## Text Preprocessing
Several preprocessing steps were applied to improve data quality:
- Emoji decoding to retain sentiment information  
- Translation of **1000+ Hindi phrases** to English using Google Translate API  
- Basic text normalization (lowercasing, noise removal, etc.)  
- Brand name extraction using regex patterns  

---

## Feature Engineering
- Created **TF-IDF vectors** for traditional ML models  
- Used **BERT embeddings** to capture contextual meaning in text  

## Modeling
- Trained an **XGBoost classifier** on TF-IDF features  
- Fine-tuned **BERT** on an external dataset for sentiment classification  
- Compared model behavior across different brands  

## Observations
- BERT performed better on context-heavy and ambiguous tweets  
- TF-IDF + XGBoost worked well for shorter, cleaner text  
- The results highlighted sentiment differences across smartphone brands  


## Tools Used
- Python  
- scikit-learn  
- XGBoost  
- Hugging Face Transformers  
- Regex  
- Google Translate API  

## Notes
This project helped in understanding practical challenges in real-world text data, especially multilingual content and noisy social media text.
