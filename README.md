# Uber Trip Duration Prediction Assessment

This repository contains the code and data for predicting the duration of Uber rides in the city of São Paulo. The project aims to improve the accuracy of trip duration predictions by using data analysis and machine learning models. The repository is organized to facilitate data processing, cleaning, model building, and analysis.

## Repository Structure

- `data_cleaning/`
  - Contains scripts to clean and process the data.
  - **`data_processing.py`**: Script responsible for generating cleaned datasets from raw input.
  - **`run.py`**: Script to execute the data processing workflow, utilizing `data_processing.py`.
  - **`data_management.py`**: (Currently incomplete) This script will eventually contain an explanation of the `data_processing.py` steps and the rationale behind the data cleaning process.

- `models/`
  - Contains scripts and notebooks used to train and evaluate different machine learning models for predicting trip duration.

- `notebooks/`
  - Jupyter notebooks used for exploratory data analysis (EDA) and model experimentation.

- `data/`
  - Directory for storing raw and processed datasets.

- `README.md`
  - Project documentation, including instructions on how to run the code and understand the project structure.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` or `conda` for managing dependencies

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/PepeZuco/uber_assessment.git
   cd uber_assessment
