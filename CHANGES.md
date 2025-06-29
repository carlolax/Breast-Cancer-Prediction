# Breast Cancer Prediction Project - Changelog

## Version 0.0.2 (2025-06-30)

### Added
- Logging system implementation:
  - Created `src/logger` with setup_logger functionality
  - Configurable log levels (INFO, DEBUG, etc.)
  - Console and file output handlers
  - Custom formatted log messages with timestamps
  - Log directory structure with separate log files per module
  - Automatic log directory creation if not existing

### Changed
- Updated project structure to include logs directory
- Added logs/ to .gitignore file to prevent log files from being tracked

## Version 0.0.1 (2025-06-29)

### Added
- Project initialization with standard data science directory structure
- README.md with project description, objectives and structure
- Data preprocessing pipeline in src/data_preprocessing.py:
  - Dataset loading with appropriate column names
  - Basic data exploration functions
  - Feature scaling using StandardScaler
  - Train-test split functionality
  - Data visualization capabilities for class distribution and feature correlation
  - Functions to save processed datasets
- Model training pipeline in src/model_training.py:
  - Implementation of multiple classification algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, KNN)
  - Model evaluation functions with metrics (accuracy, precision, recall, F1-score)
  - Visualization of model performance comparisons
  - Confusion matrix visualization for each model
  - ROC curve generation
  - Feature importance analysis for tree-based models
  - Best model selection and persistence
- Environment setup:
  - requirements.txt with necessary Python dependencies
  - environment.yaml for conda environment configuration
- Jupyter notebook implementation with integrated pipeline components
