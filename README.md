# Resume–Job Matching and Candidate Ranking using Interpretable Machine Learning

## Overview

This project builds an interpretable machine learning pipeline for automated resume–job matching and candidate ranking.

Given a resume and a job description, the system:

* predicts whether a candidate should be selected or rejected
* ranks candidates by predicted relevance
* analyzes model behavior, ranking quality, and benchmark limitations

Unlike many resume-screening projects that mainly focus on deep learning or LLM fine-tuning, this project emphasizes:

* interpretable NLP
* feature engineering
* ranking-aware evaluation
* model behavior analysis
* error analysis
* dataset and benchmark validity analysis

The project investigates how lexical alignment, TF-IDF similarity, and explicit skill matching influence hiring prediction and ranking performance.

---

# Main Research Questions

This project studies several key questions:

1. How effective are lexical NLP methods for resume–job matching?

2. Do engineered alignment features improve prediction quality?

3. Why do ranking metrics become nearly saturated across models?

4. Does the dataset structure artificially simplify the ranking task?

5. What limitations arise when lexical patterns dominate hiring prediction?

---

# Repository Structure

```text
resume-job-matching/
├── .gitignore
├── README.md
├── requirements.txt
├── Report.pdf
│
├── notebooks/
│   ├── eda.ipynb
│   └── model_training_results_analysis.ipynb
│
└── src/
    ├── data.py
    ├── features.py
    ├── models.py
    └── evaluate.py
```

---

# Dataset

Dataset: Resume Screening Dataset

The dataset contains approximately 10k resume–job pairs.

Each sample includes:

* `resume`
* `job_description`
* `decision` (`select` / `reject`)
* `role`
* `reason_for_decision`

The dataset supports both:

* binary classification
* candidate ranking

---

# Methodology

## Text Representations

The project compares multiple lexical representations:

* Bag-of-Words (BoW)
* TF-IDF
* TF-IDF + TruncatedSVD

---

## Engineered Matching Features

The project builds interpretable alignment features between resumes and job descriptions.

### Overlap Ratio

Measures explicit lexical overlap between resume tokens and job-description tokens.

### Cosine Similarity

Measures global TF-IDF similarity between resumes and job descriptions.

### Skill Matching Features

Detects alignment on technical skills such as:

* Python
* SQL
* Java
* AWS
* Docker
* Spark
* Kubernetes
* TensorFlow

and many others.

---

# Models

The project compares multiple interpretable ML baselines:

| Model                                                   |
| ------------------------------------------------------- |
| BoW + Multinomial Naive Bayes                           |
| TF-IDF + Logistic Regression                            |
| TF-IDF + Linear SVM                                     |
| TF-IDF + TruncatedSVD + Logistic Regression             |
| TF-IDF + overlap + Logistic Regression                  |
| TF-IDF + overlap + cosine + Logistic Regression         |
| TF-IDF + overlap + cosine + skill + Logistic Regression |
| Engineered Features + Random Forest                     |
| Engineered Features + XGBoost                           |

---

# Evaluation

## Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

## Ranking Metrics

* Precision@10
* NDCG@10
* Precision@1
* Recall@1
* MRR
* MAP

---

# Main Analyses

A major focus of this project is understanding *why* models behave the way they do.

The analysis includes:

* confusion matrix analysis
* false positive analysis
* false negative analysis
* coefficient interpretation
* feature importance analysis
* prediction score distribution analysis
* ranking saturation analysis
* cross-model ranking correlation analysis
* learning curve diagnostics
* XGBoost logloss diagnostics

---

# Main Findings

## 1. Lexical Features Dominate Performance

Simple lexical models perform surprisingly strongly.

BoW + Naive Bayes achieves the highest F1-score because many hiring decisions in the dataset are associated with strong lexical patterns.

---

## 2. Engineered Features Provide Limited Improvements

Overlap ratio, cosine similarity, and skill matching improve interpretability but only slightly improve classification performance.

This suggests that surface-level lexical signals dominate the benchmark.

---

## 3. Ranking Metrics Are Nearly Saturated

Most models achieve near-perfect:

* Precision@10
* NDCG@10

However, classification performance remains moderate.

Further analysis shows that this occurs largely because many job descriptions contain extremely small candidate pools.

---

## 4. Dataset Structure Simplifies Ranking

The project performs benchmark-validity analysis and finds:

* 905 / 1265 job descriptions contain only one candidate
* 88.1% of jobs contain at most one positive example

This means many ranking tasks reduce to identifying a single obvious positive candidate rather than solving a difficult ranking problem.

---

## 5. Models Learn Annotation-Style Lexical Patterns

Logistic Regression coefficient analysis shows that highly weighted features include terms such as:

### Strong Positive Signals

* excellent
* select
* fluent
* organization
* domain

### Strong Negative Signals

* limited
* gaps
* reject
* lack
* struggled

This suggests that models partially rely on annotation-style wording patterns rather than deeper semantic understanding.

---

# Exploratory Data Analysis (EDA)

The EDA investigates whether resumes and job descriptions are already lexically separable before model training.

The analysis includes:

* vocabulary size analysis
* token frequency analysis
* overlap-ratio analysis
* cosine similarity analysis
* skill frequency analysis
* candidate-pool analysis

The EDA reveals that many lexical statistics substantially overlap between selected and rejected samples, suggesting that simple lexical alignment alone is insufficient for robust hiring prediction.

---

# Ranking Saturation and Benchmark Validity

One major contribution of this project is the investigation of ranking saturation.

Although many models achieve nearly perfect ranking metrics, further analysis shows that the benchmark itself may simplify the ranking task.

The project studies:

* candidate-pool size distribution
* positive candidate count distribution
* ranking score gaps
* ranking sensitivity under stricter metrics
* cross-model ranking correlation

The analysis suggests that dataset structure, candidate-pool size, and lexical separability strongly influence ranking performance.

---

# Main Files

## notebooks/eda.ipynb

Exploratory data analysis notebook containing:

* vocabulary analysis
* overlap ratio analysis
* cosine similarity analysis
* skill frequency analysis
* dataset separability analysis

---

## notebooks/model_training_results_analysis.ipynb

Main modeling and evaluation notebook containing:

* model training
* ranking evaluation
* confusion matrix analysis
* error analysis
* coefficient analysis
* feature importance analysis
* learning curves
* ranking saturation diagnostics

---

## src/data.py

Data loading and preprocessing pipeline.

Main responsibilities:

* dataset loading
* missing value handling
* duplicate removal
* label conversion
* train/test split

---

## src/features.py

Feature engineering utilities.

Main responsibilities:

* overlap ratio computation
* cosine similarity computation
* technical skill matching
* token processing
* vectorizer construction

---

## src/models.py

Machine learning model definitions.

Includes:

* Logistic Regression
* Linear SVM
* Naive Bayes
* Random Forest
* XGBoost

---

## src/evaluate.py

Classification and ranking evaluation utilities.

Includes:

* F1-score
* Precision
* Recall
* Precision@k
* NDCG@k
* confusion matrix generation

---

# Requirements

Main libraries used:

* pandas
* NumPy
* scikit-learn
* XGBoost
* matplotlib
* seaborn
* nltk

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Report

The full research-style report is included in:

```text
Report.pdf
```

The report includes:

* exploratory data analysis
* methodology
* model comparison
* ranking evaluation
* error analysis
* feature analysis
* learning curve analysis
* ethical considerations
* benchmark limitation analysis

---

# Ethical Considerations

AI-assisted hiring systems may introduce or amplify bias.

This project discusses several important risks:

* lexical bias
* annotation artifacts
* keyword-based unfairness
* ranking saturation
* dataset limitations
* over-reliance on automated screening

The system is intended to support recruiters rather than replace human decision-making.

---

# Future Work

Possible future improvements include:

* transformer embeddings
* sentence-transformer ranking models
* LLM-based semantic matching
* fairness-aware evaluation
* retrieval-based ranking pipelines
* real-world candidate-pool evaluation

---

# Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* matplotlib
* seaborn
* nltk
* Jupyter Notebook

---

# Author

Shiyi Li

M.S. Candidate in Data Science  
New York University

