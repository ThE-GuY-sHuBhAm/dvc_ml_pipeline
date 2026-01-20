# Building Reproducible ML Pipelines with DVC

## Overview

This repository contains a **sample machine learning project** for detecting sentiment from text, specifically **movie reviews**.  
The task involves classifying reviews as positive or negative using a standard NLP-based approach.

The **sentiment classification itself is not the primary focus** of this project.

---

## Primary Objective

The main goal of this project is to serve as a **proof of work** for learning and understanding how to build **production-style machine learning systems**, with a strong emphasis on:

- **DVC (Data Version Control)**
- **End-to-end ML pipeline design**
- **Data, model, and parameter versioning**
- **Reproducibility and experiment tracking**

The ML problem is intentionally kept simple so that the focus remains on **engineering practices and workflow design**, rather than model complexity.

---

## Engineering Practices Followed

To closely resemble real-world ML systems, the project incorporates the following practices:

- **Modular code structure**, separating data ingestion, preprocessing, feature engineering, and model training
- **Centralized configuration management** using `params.yaml`
- **Robust exception handling** to make pipeline stages fault-tolerant
- **Structured logging** for better traceability and debugging
- **Cookiecutter Data Science–style project layout** to enforce consistency and clarity

These choices help ensure that the pipeline is maintainable, scalable, and easy to reason about.

---

## Pipeline Structure

The workflow follows a modular pipeline approach:

- Data ingestion
- Data preprocessing
- Feature engineering
- Model training
- Model versioning with DVC

Each stage is tracked using **DVC**, ensuring that changes in data, parameters, or code automatically trigger the appropriate pipeline re-execution.

---

## Tools & Technologies

- Python
- Scikit-learn
- Pandas / NumPy
- DVC
- Git
- Nltk

---

## Key Takeaway

This project emphasizes **ML engineering fundamentals** over raw model performance.  
It demonstrates how to design, version, and reproduce end-to-end machine learning pipelines in a structured and systematic manner.

---
