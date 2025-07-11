# Build and Operate Machine Learning Solutions with Azure (Project)

This repository contains the completed project for the **Microsoft Learn** course:  
**[Build and Operate Machine Learning Solutions with Azure](https://learn.microsoft.com/en-us/training/paths/build-operate-machine-learning-solutions-azure/)**

The course teaches how to manage the complete machine learning lifecycle using Azure Machine Learning. It covers key concepts such as training, deployment, batch inference, fairness, and monitoring. This project replicates those workflows locally using Python and the UCI Adult Census Income dataset.

---

## Course Overview

The course is organized into several modules that simulate real-world machine learning workflows in Azure:

### 1. Getting Started with Azure ML
- Set up an Azure Machine Learning workspace
- Run experiments using notebooks, CLI, and SDK
- Train and register models

### 2. Working with Data in Azure ML
- Create and manage datastores and datasets
- Access and preprocess data in the cloud
- Use compute resources for training

### 3. Training Models with Pipelines
- Build, publish, and execute training pipelines
- Register and deploy models with the Azure ML service

### 4. Batch Inference and Hyperparameter Tuning
- Create and run batch inference pipelines
- Perform large-scale hyperparameter sweeps

### 5. Automated ML, Fairness, and Explainability
- Use AutoML to find the best model
- Analyze fairness using Fairlearn
- Apply privacy techniques and explainability methods

### 6. Monitoring and Managing Models
- Monitor deployed models using telemetry
- Detect data drift and maintain performance
- Implement responsible AI practices

---

## Project Structure

This project simulates Azure ML workflows using open-source tools and a local environment.

| File / Folder                                | Description                                                   |
|---------------------------------------------|---------------------------------------------------------------|
| `data/adult.csv`                             | Cleaned UCI Adult dataset                                     |
| `notebook1_data_pipeline_training.ipynb`     | Data preprocessing, pipeline creation, training, and tuning  |
| `notebook2_batch_inference.ipynb`            | Simulated batch scoring with trained model                   |
| `notebook3_model_monitoring_drift_detection.ipynb` | Feature distribution comparison and drift detection       |
| `models/`                                    | Serialized trained model (`.pkl`)                            |
| `metrics/`                                   | Metrics and feature importance plots                         |
| `requirements.txt`                           | Python dependencies                                           |
| `README.md`                                  | Project documentation                                         |

---

## Notebooks Summary

### Notebook 1: Data Pipeline and Training
- Loads and cleans the Adult Income dataset
- Encodes categorical variables and scales features
- Trains a `RandomForestClassifier` using a pipeline
- Performs hyperparameter tuning with `GridSearchCV`
- Saves the model and metrics

### Notebook 2: Batch Inference
- Loads the saved model
- Samples new data and simulates a batch prediction process
- Generates and stores inference results

### Notebook 3: Drift Detection and Monitoring
- Compares training and batch data distributions
- Uses Kolmogorov–Smirnov test for numeric features
- Uses Chi-squared test for categorical features
- Identifies feature drift that may require model retraining

---

## Dataset Information

- **Source**: [UCI Machine Learning Repository – Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Goal**: Predict whether a person earns more than \$50,000 per year
- **Features**: Age, workclass, education, occupation, hours-per-week, etc.
- **Target Variable**: `income` (binary classification)

The dataset is saved locally as `data/adult.csv`.

---

## Setup Instructions

Clone the repository and install dependencies:

```bash
git clone https://github.com/Sara-J-86/build-operate-ml-solutions-azure.git
cd build-operate-ml-solutions-azure
pip install -r requirements.txt
