# Setup Guide

## Prerequisites

* Python 3.10+
* pip
* Git



# Clone Repository


git clone <repository-url>
cd gpu-performance-simulator




# Create Virtual Environment

Windows:


python -m venv venv
venv\Scripts\activate


Linux / macOS:


python3 -m venv venv
source venv/bin/activate




# Install Dependencies


pip install -r requirements.txt




# Project Structure


project/
│
├── datasets/
│
├── preprocessing.py
├── featureengg.py
├── mergebenchmarks.py
├── model_training.py
├── bottleneck_model.py
├── fps_preprocessing.py
├── fps_model.py
├── gpu_simulator.py
│
├── gpu_performance_model.pkl
├── bottleneck_model.pkl
├── fps_model.pkl
│
├── app.py
│
├── SYSTEM_ARCHITECTURE.md
├── setup.md
└── README.md




# Training Pipeline

## Step 1

Preprocess datasets


python preprocessing.py




## Step 2

Generate engineered features


python featureengg.py




## Step 3

Merge benchmark datasets


python mergebenchmarks.py



## Step 4

Train benchmark prediction model


python model_training.py




## Step 5

Train bottleneck classification model


python bottleneck_model.py




## Step 6

Flatten FPS dataset


python fps_preprocessing.py




## Step 7

Train FPS prediction model


python fps_model.py




# Running Application

Launch Streamlit dashboard:


streamlit run app.py


Application will start at:


http://localhost:8501


# Features

* GPU Benchmark Prediction
* FPS Prediction
* Bottleneck Classification
* Future GPU Simulation
* Hardware Scaling Analysis
* NVIDIA / AMD / Intel Support


# Troubleshooting

## Missing Dependencies


pip install -r requirements.txt


## Streamlit Not Found


pip install streamlit

## Model Files Missing

Re-run training scripts:


python model_training.py
python bottleneck_model.py
python fps_model.py

