# Telecommunications Customer Segmentation - KNN Classifier

## Overview
This repository contains a machine learning project focused on customer segmentation in the telecommunications industry using the K-Nearest Neighbors (KNN) classifier. The project aims to classify customers into different categories based on demographic and usage data, providing insights for targeted marketing strategies.

## Dataset
The dataset used (`teleCust1000t.csv`) includes information on customer attributes and their assigned customer category (`custcat`):
- `region`, `tenure`, `age`, `marital`, `address`, `income`, `ed`, `employ`, `retire`, `gender`, `reside`
- `custcat`: Customer category (1, 2, 3, 4)

## Project Structure
1. **Data Exploration**: Analyzed the dataset to understand distributions and correlations.
2. **Data Preprocessing**: Standardized the data to prepare for model training.
3. **Model Training**: Utilized the KNN classifier to predict customer categories.
4. **Model Evaluation**: Assessed model performance using accuracy metrics and visualizations.

## Setup Instructions
To run the notebook locally, ensure Python and the following libraries are installed:
```bash
pip install numpy pandas matplotlib scikit-learn
