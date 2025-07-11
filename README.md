# Build and Operate Machine Learning Solutions with Azure (Project)

This repository contains the completed project from the **Microsoft Learn** course:  
**[Build and Operate Machine Learning Solutions with Azure](https://learn.microsoft.com/en-us/training/paths/build-operate-machine-learning-solutions-azure/)**

The course is designed to teach machine learning engineers and data scientists how to build, train, deploy, and monitor ML models in the **Azure Machine Learning (Azure ML)** environment. This local project mirrors that end-to-end lifecycle using Python and the **UCI Adult Census Income dataset**.

---

## Course Overview

The course is structured to simulate a real-world machine learning solution lifecycle in Azure, with each module targeting a specific aspect of operational ML:

### 1. **Getting Started with Azure ML**
- Provisioning and managing an Azure Machine Learning workspace.
- Running code-based experiments with notebooks, CLI, and SDK.
- Training a model and registering it in the workspace.


---

### 2. **Working with Data in Azure ML**
- Creating and managing **Datastores** and **Datasets**.
- Accessing and preprocessing data in the cloud.
- Using cloud compute targets for training at scale.


---

###  3. **Training Models with Pipelines**
- Building, publishing, and running pipelines to train models.
- Model registration and deployment with Azure ML service.



---

###  4. **Batch Inference & Hyperparameter Tuning**
- Creating batch inference pipelines.
- Running **cloud-scale hyperparameter sweeps**.



---

###  5. **Automated ML, Fairness & Explainability**
- Using **AutoML** for model selection.
- Ensuring **data privacy** and using **Fairlearn** for fairness.
- Analyzing model predictions and explainability with SHAP.



---

###  6. **Monitoring & Managing Models**
- Understanding real-world telemetry after deployment.
- Detecting **data drift** and **concept drift**.
- Maintaining model accuracy and fairness in production.


---

##  Project Contents

This repository recreates the Azure ML lifecycle **locally**, using Python and `scikit-learn`. It uses the **UCI Adult Income dataset** to predict whether an individual earns more than $50K/year.

| File / Folder | Description |
|---------------|-------------|
| `data/adult.csv` | Cleaned dataset for training |
| `notebook1_data_pipeline_training.ipynb` | Model training, preprocessing, pipeline, and hyperparameter tuning |
| `notebook2_batch_inference.ipynb` | Simulates batch scoring with the trained model |
| `notebook3_model_monitoring_drift_detection.ipynb` | Detects data drift between training and new data |
| `metrics/` | Metrics file and feature importance plot |
| `models/` | Saved trained model (`.pkl`) |
| `requirements.txt` | List of dependencies |
| `README.md` | This documentation file |

---

## Summary of Notebooks

###  Notebook 1: Data Pipeline & Model Training
- Cleans the dataset and encodes features
- Builds a training pipeline using `Pipeline` and `GridSearchCV`
- Trains a `RandomForestClassifier`
- Saves the model and evaluation metrics

### Notebook 2: Batch Inference Simulation
- Loads trained model
- Samples new batch data from original dataset
- Performs predictions and saves results

### Notebook 3: Monitoring and Drift Detection
- Compares batch and training distributions
- Uses:
  - **Kolmogorov–Smirnov test** for numeric features
  - **Chi-squared test** for categorical features
- Flags any significant data drift

---

## Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Description**: Dataset based on 1994 US Census. Goal is to predict whether income exceeds $50K based on features like age, education, and occupation.
- Already downloaded locally as: `data/adult.csv`

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
