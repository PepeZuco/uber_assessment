# Uber Trip Duration Prediction Assessment

This repository contains code and data for predicting the duration of Uber rides in São Paulo. The project aims to improve trip duration prediction accuracy through data analysis and machine learning models. The repository is organized to facilitate data cleaning, processing, model building, and analysis.

## Repository Structure

- `data_cleaning/`
  - Contains scripts for cleaning and processing the data.
  - **`data_processing.py`**: Generates cleaned datasets from raw input data.
  - **`run.py`**: Executes the data processing workflow using `data_processing.py`.
  - **`EDA.py`**: A Jupyter notebook that explains the steps in `data_processing.py` and the rationale behind the data cleaning process.

- `models/`
  - Contains scripts for training and testing models.
  - **`model_pipeline.py`**: Handles model training and testing.
  - **`run.py`**: Runs the model pipeline, utilizing `model_pipeline.py`.
  - `Graphs/`: Stores all graphs generated during model training.
  - `trained_model/`: Stores the model trained.

- `PowerPoint/`
  - Contains a Jupyter notebook for generating graphs for presentations and a .pptx file.
  - **`graph_generator.ipynb`**: Generates graphs specifically for presentations.
  - `Graphs/`: Stores all presentation-related graphs.

- `data_extra/`
  - Stores additional datasets.
- `data_final/`
  - Stores cleaned datasets.
- `data_original/`
  - Stores original datasets.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` or `conda` for managing dependencies

### Installation

```bash
pip install -r requirements.txt
