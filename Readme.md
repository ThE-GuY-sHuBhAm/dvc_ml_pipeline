# End-to-End MLOps Pipeline: Sentiment Analysis

![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-0194E2)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6)
![Python](https://img.shields.io/badge/Python-3.11-3776AB)

## 📖 Overview

This repository demonstrates a production-grade **Machine Learning Operations (MLOps)** pipeline for sentiment analysis. While the model classifies movie reviews (Positive/Negative), the primary focus is on **engineering robust, reproducible, and automated ML systems.**

The project implements a full lifecycle workflow: from data versioning and experiment tracking to automated testing (CI) and containerized deployment (CD).

---

## 🚀 Key Features

* **Data & Model Versioning:** Utilizes **DVC** to track datasets and model artifacts, ensuring full reproducibility of any historical experiment.
* **Experiment Tracking:** Integrates **MLflow** to log metrics, parameters, and artifacts (models + vectorizers) to a remote tracking server (DagsHub).
* **Model Registry:** Implements a stage-based workflow (Staging $\to$ Production) for model management.
* **CI/CD Automation:** Uses **GitHub Actions** to:
    * Pull data from remote storage.
    * Reproduce the pipeline.
    * Run unit & integration tests on the model.
    * Automatically **promote** valid models to Production.
* **Containerized Inference:** Wraps the prediction API (Flask) in **Docker**, making it portable and ready for cloud deployment.

---

## 🛠️ Tech Stack

| Component | Tool | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Core logic |
| **Versioning** | DVC | Data and pipeline versioning |
| **Tracking** | MLflow | Metrics, parameters, and Model Registry |
| **Storage** | DagsHub / S3 | Remote storage for DVC and MLflow |
| **CI/CD** | GitHub Actions | Automated testing and deployment pipelines |
| **Container** | Docker | Application containerization |
| **API** | Flask | Real-time inference REST API |
| **ML Library** | Scikit-learn | Gradient Boosting & TF-IDF Vectorization |

---

## 🏗️ Pipeline Architecture

The pipeline is modularized into distinct stages defined in `dvc.yaml`:

1.  **Data Ingestion:** Downloads and saves raw data.
2.  **Preprocessing:** Cleans text (Lemmatization, Stopword removal, Noise reduction).
3.  **Feature Engineering:** Transforms text to TF-IDF vectors (and pickles the vectorizer).
4.  **Model Building:** Trains a Gradient Boosting Classifier.
5.  **Model Evaluation:** Calculates metrics (Accuracy, F1) and logs them to MLflow.
6.  **Registration:** Registers the candidate model to the MLflow Model Registry.

---

## 💻 Installation & Setup

### Prerequisites
* Python 3.11+
* Git
* Docker (optional, for containerization)

### 1. Clone the Repository
```bash
git clone [https://github.com/ThE-GuY-sHuBhAm/dvc_ml_pipeline.git](https://github.com/ThE-GuY-sHuBhAm/dvc_ml_pipeline.git)
cd dvc_ml_pipeline