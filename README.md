# Breast Cancer Prediction Project

## Problem Statement

Breast cancer is one of the most common cancers among women worldwide. Early detection significantly increases treatment success rates and patient survival. However, manual classification of breast masses through visual examination of cell samples is time-consuming and can be subject to human error.

This project aims to use machine learning to assist in the accurate diagnosis of breast cancer using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The dataset is available from the UCI Machine Learning Repository and can be downloaded [here](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## Objectives

1. Develop a machine learning model that can accurately classify breast masses as benign or malignant
2. Identify the most important features for breast cancer diagnosis
3. Evaluate model performance using metrics relevant to medical diagnostics (accuracy, precision, recall, etc.)
4. Create a reliable tool that could potentially assist medical professionals in diagnosis

## Dataset

This project uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, which contains features computed from digitized images of fine needle aspirates (FNAs) of breast masses. The dataset includes:

- 569 instances (357 benign, 212 malignant)
- 30 features describing characteristics of cell nuclei present in the image
- Features include measurements like radius, texture, perimeter, area, smoothness, etc.

## Project Structure

- `/notebooks/`: Jupyter notebooks for exploration and analysis
- `/src/`: Source code for the project
- `/models/`: Saved model files
- `/reports/`: Generated reports and visualizations
