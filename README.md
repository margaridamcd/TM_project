# Text Mining — Group Project 2025/2026

Financial Market Sentiment Prediction from Tweets (Bearish/Bullish/Neutral)

### Group 11

| Name | Student ID |
|------|------------|
| Ana Macedo | 20250405 |
| Carlota Pires | 20250383 |
| Francisca Calçoa | [ID] |
| Francisca Martins | [ID] |

---

## Project Overview

The goal of this project is to develop an NLP model capable of predicting market sentiment (Bearish, Bullish, or Neutral) from financial tweets. Multiple feature representations (BoW, TF-IDF, Word2Vec, GloVe, FinBERT, RoBERTa) were combined with different classification models (Random Forest and XGBoost) to identify the best performing approach. Transformer-based models (FinBERT and RoBERTa) were also used directly for classification.

---

## Models Tested

### Traditional ML Models (with embeddings)
- **Random Forest (RF)** — tested with all feature engineering methods
- **XGBoost** — tested with all feature engineering methods

### Transformer Encoders
- **FinBERT** — BERT model pre-trained on financial corpora
- **RoBERTa** — Robustly optimized BERT approach

---

## Feature Engineering Techniques Tested

| Method | Type | Dimensionality |
|--------|------|----------------|
| Bag-of-Words (BoW) | Sparse | 5,000 |
| TF-IDF | Sparse | 5,000 |
| Word2Vec (CBOW) | Dense | 100 |
| Word2Vec (Skip-gram) | Dense | 100 |
| GloVe-100 | Dense | 100 |
| FinBERT CLS | Dense (Transformer) | 768 |
| RoBERTa CLS | Dense (Transformer) | 768 |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `EDA1.ipynb` | Exploratory Data Analysis — class distribution, tweet length analysis, word clouds |
| `RF.ipynb` | Random Forest experiments with all embedding types |

---

## Reproducibility

All experiments use `random_state = 42` for consistent results across runs.

- **Train/Validation Split**: 80/20 stratified split (7,634 train / 1,909 validation)
- **Corpus Size**: 9,543 training tweets, 299 test tweets
- **Test Set Predictions**: Saved as `pred_XX.csv` (ID + predicted label)

---

## Preprocessing Techniques Applied

The following preprocessing techniques were implemented (meeting the 4+ requirement):

1. **Regular Expressions** — removal of URLs, mentions (@user), hashtags, and special characters
2. **Stop Words Removal** — removal of common Portuguese/English stop words
3. **Lowercasing** — standardization of text
4. **Tokenization** — NLTK tokenizer for BoW/Word2Vec
5. **Lemmatization** — applied for BoW and traditional embeddings

---

## Classification Models Summary

| Model Type | Models | Feature Methods Tested |
|------------|--------|------------------------|
| Traditional ML | Random Forest, XGBoost | BoW, TF-IDF, Word2Vec, GloVe |
| Transformer Encoders | FinBERT, RoBERTa | CLS embeddings, Fine-tuning |

---

## Key Results 
| Configuration | Accuracy | Macro F1 | Bearish F1 | Bullish F1 | Neutral F1 |
|---------------|----------|----------|------------|------------|------------|


