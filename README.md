
# ScaleSage-GPP

Hybrid XGBoost + PyTorch ML system for semiconductor scaling analysis and real-time GPU FPS prediction with SHAP, LIME, and Scaling Integrated Gradients explainability.

---

## Overview

ScaleSage-GPP is a forensic Hybrid Classical-ML Pipeline (HCMLP) designed for semiconductor scaling analysis and real-time graphics performance prediction.

The system integrates:

- XGBoost regression
- PyTorch MLP
- SHAP explainability
- LIME local attribution
- Scaling Integrated Gradients (S-IG)

to deliver transparent and production-ready GPU FPS prediction workflows.

---

## Features

- Hybrid Classical-ML Pipeline (HCMLP)
- XGBoost + PyTorch MLP weighted ensemble
- Semiconductor scaling analysis
- Moore’s Law feature engineering
- SHAP global feature attribution
- LIME local explainability
- Scaling Integrated Gradients (S-IG)
- Real-time GPU FPS prediction
- Streamlit interactive dashboard
- FastAPI inference backend
- Hardware forensic analytics
- Batch CSV audit workflows
- GPU benchmark analysis
- Low-latency inference pipeline
- Dockerized deployment
- Kubernetes manifests

---

## Performance

- R² = 0.9988
- RMSE = 2.11 FPS
- MAE = 1.24 FPS
- Sub-45ms inference latency
- 48,721 merged benchmark records

---

## Tech Stack

### Machine Learning
- XGBoost
- PyTorch
- Scikit-learn

### Explainable AI
- SHAP
- LIME
- Scaling Integrated Gradients (S-IG)

### Backend
- FastAPI

### Frontend
- Streamlit

### Visualization
- Plotly
- Matplotlib

### Deployment
- Docker
- Kubernetes

---

## Run

### Training

python -m training.train

### Backend

uvicorn app.api.server:app --reload

### Dashboard

streamlit run dashboard/dashboard.py
