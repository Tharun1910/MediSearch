# 🩺 MediSearch: A Health Information Retrieval System

MediSearch is a **health-domain information retrieval system** that combines **classical lexical retrieval** techniques with **neural semantic retrieval** to improve the accuracy and relevance of medical search results. The system is built using the **MedQuAD dataset** curated from trusted National Institutes of Health (NIH) sources and demonstrates the effectiveness of **hybrid retrieval models** in medical information retrieval.

---

## 📌 Project Overview

General-purpose search engines rely heavily on keyword matching and often fail to capture the semantic complexity of medical queries. MediSearch addresses this limitation by integrating:

- **TF-IDF** – baseline lexical retrieval  
- **BM25** – probabilistic keyword-based retrieval  
- **Sentence-BERT (SBERT)** – semantic sentence embeddings  
- **Hybrid Retrieval** – weighted fusion of BM25 and SBERT  

The system supports **end-to-end retrieval**, from preprocessing and indexing to ranking and deployment via a **Streamlit web interface**.

---

## 📊 Dataset

**MedQuAD (Medical Question Answer Dataset)**  
- Curated from multiple NIH websites  
- Contains verified medical question–answer pairs  
- After preprocessing: **16,353 records**

### Dataset Fields
- `question` – Medical question in natural language  
- `answer` – Verified medical answer  
- `source` – NIH website source  
- `focus_area` – Medical topic category  

### Dataset Sources
- Original NIH Repository:  
  https://github.com/abachaa/MedQuAD  
- Kaggle Version Used:  
  https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research  

---

## ⚙️ System Architecture

```text
User Query
   ↓
Text Preprocessing
   ↓
TF-IDF | BM25 | SBERT
   ↓
Hybrid Score Fusion (α = 0.70)
   ↓
Top-k Ranked Medical Answers

```
---

## 📈 Evaluation

The MediSearch system is evaluated using **standard ranking-based information retrieval metrics**. Since the MedQuAD dataset does not provide explicit human-annotated relevance judgments, **pseudo-relevance labels** are constructed by treating documents that share the same `focus_area` as relevant for a given query. This strategy enables consistent and fair comparison across retrieval models.

### Evaluation Metrics (@10)

- **Precision@10** – Measures the proportion of relevant documents among the top 10 retrieved results.
- **Recall@10** – Measures the fraction of relevant documents retrieved within the top 10 results.
- **nDCG@10** – Evaluates ranking quality by assigning higher importance to relevant documents appearing at higher ranks.
- **MRR@10 (Mean Reciprocal Rank)** – Measures how quickly the first relevant result appears in the ranked list.

### Key Findings

- **TF-IDF** serves as a baseline and performs well only when strong keyword overlap exists.
- **BM25** consistently outperforms TF-IDF across all evaluation metrics.
- **SBERT** achieves the highest early relevance, as reflected by the highest MRR@10.
- The **Hybrid model** (BM25 + SBERT with α = 0.70) achieves the best overall performance, yielding the highest nDCG@10, Precision@10, and Recall@10.

These results demonstrate that combining lexical and semantic retrieval signals leads to more robust and effective medical information retrieval.

---

## 🚀 Deployment

MediSearch is deployed as a **lightweight web application** using the **Streamlit** framework, providing an interactive interface for medical information retrieval.

### Deployment Features

- User-friendly medical query input
- Retrieval model selection (TF-IDF, BM25, SBERT, Hybrid)
- Real-time ranked retrieval results
- Display of answers along with their sources and focus areas

### Running the Application Locally

```bash
pip install -r requirements.txt
streamlit run app.py
