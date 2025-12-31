# Project Roadmap: IR System Evaluation

This document outlines the roadmap for building and evaluating the Information Retrieval system using TF-IDF, BM25, and Rocchio models.

## Phase 1: Foundation (Completed)
- [x] Project structure setup
- [x] Implementation of TF-IDF Model (scikit-learn)
- [x] Implementation of BM25 Model (rank_bm25)
- [x] Implementation of Rocchio Algorithm
- [x] Basic Preprocessing Pipeline (Tokenization, Stemming, Lemmatization)
- [x] Evaluation Metrics (MAP, P@k, NDCG)
- [x] Synthetic Data Experiment (`run_experiment.py`)

## Phase 2: Data Integration & Configuration (Current Focus)
- [ ] **Data Loading**: Implement loaders for real-world datasets (CISI, MS MARCO).
    - Use `ir_datasets` or custom parsers.
- [ ] **Configuration Management**:
    - Populate `configs/` with YAML files for each model.
    - Update `run_experiment.py` to load parameters from configs.
- [ ] **Logging & artifacts**:
    - Improve result saving (JSON/CSV).
    - Add logging to track experiment progress.

## Phase 3: Comprehensive Evaluation
- [ ] **Hyperparameter Tuning**:
    - Grid search for BM25 (k1, b).
    - Tuning Rocchio parameters (alpha, beta, gamma).
- [ ] **Full-Scale Experiments**:
    - Run all models on the full dataset.
    - Compare performance (Time vs. Accuracy).
- [ ] **Analysis**:
    - Analyze query drift in Rocchio.
    - Compare Stemming vs. Lemmatization impact.

## Phase 4: Visualization & Reporting
- [ ] **Visualization**:
    - Plot Precision-Recall curves.
    - Bar charts for MAP/NDCG comparison.
- [ ] **Final Report**:
    - Summarize findings.
    - Document implementation details.

