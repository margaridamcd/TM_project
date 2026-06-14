# Text Mining — Group Project 2025/2026

Financial Market Sentiment Prediction from Tweets (Bearish/Bullish/Neutral)

### Group 11

| Name | Student ID |
|------|------------|
| Ana Macedo | 20250405 |
| Carlota Pires | 20250383 |
| Francisca Calçoa | 20240266 |
| Francisca Martins | 20250347 |

---

## Project Overview

The goal of this project is to develop an NLP model capable of predicting financial market sentiment (Bearish, Bullish, or Neutral) from financial tweets. The work follows a full NLP pipeline: data exploration, preprocessing, feature engineering, and classification modelling, with macro-averaged F1-score as the primary evaluation metric due to class imbalance.

Multiple feature representations were combined with several classification models to identify the best performing approach. Transformer-based models (FinBERT, RoBERTa, BERTweet) were also used directly as classifiers, both pre-trained and fine-tuned. As extra work, Mistral 7B was evaluated as a zero-shot decoder-based classifier.

---

## Models Tested

### Traditional ML Models (with embeddings)
- **Logistic Regression (LR)** — tested with all feature engineering methods
- **Random Forest (RF)** — tested with all feature engineering methods, including grid search
- **XGBoost** — tested with TF-IDF, GloVe, and RoBERTa embeddings, including grid search

### Transformer Encoders
- **FinBERT** — BERT model pre-trained on financial corpora (used as classifier and as CLS embeddings)
- **RoBERTa** — Robustly optimized BERT approach (used as classifier and as CLS embeddings)
- **BERTweet** — RoBERTa-based model pre-trained on Twitter data (used as classifier, as CLS embeddings, and fine-tuned)

### Decoder Models (extra work)
- **Mistral 7B** — zero-shot classification via prompting

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
|`tm_tests_11.ipynb` | Full pipeline: data exploration, preprocessing, feature engineering, and testing of all classification models |
| `tm_final_11.ipynb` | Final model only: fine-tunes BERTweet on the full training set and generates test set predictions |

---

## Reproducibility

All experiments use `random_state = 42` for consistent results across runs.

- **Train/Validation Split**: 80/20 stratified split (7,634 train / 1,909 validation)
- **Corpus Size**: 9,543 training tweets, 299 test tweets
- **Test Set Predictions**: Saved as `pred_11.csv` (ID + predicted label)

---

## Preprocessing Techniques Applied

1. **Language Detection** — FastText (`lid.176.bin`) to identify non-English tweets
2. **Lowercasing** — standardization of text
3. **Regular Expressions** — removal of URLs; selective removal/retention of punctuation (sentiment-bearing symbols like `!`, `?`, `$`, `%` kept)
4. **Date/Time Normalization** — removal of dates, years, and hours, with retention of historically significant references (e.g. `financial_crisis_2008`, `russell2000`)
5. **Stopword Removal** — optional, used for comparison across models
6. **Lemmatization** — applied to preserve semantic context

---

## Key Results 
 Configuration | Macro F1 | Details |
|---------------|----------|----------|
| **BERTweet fine-tuned** | **0.8447** | Final model, raw data |
| Mistral 7B | 0.7600 | Zero-shot, extra work |
| XGBoost + RoBERTa | 0.7520 | Grid search |
| Random Forest + FinBERT | 0.7500 | Grid search |
| Logistic Regression + RoBERTa / FinBERT | 0.7200 | |
| FinBERT (pre-trained classifier) | 0.6627 | Raw data |
| BERTweet (pre-trained classifier) | 0.5911 | Raw data |
| RoBERTa (pre-trained classifier) | 0.5801 | Raw data |

Full results for all feature/model combinations are available in the report (Appendix B, Table 7).


