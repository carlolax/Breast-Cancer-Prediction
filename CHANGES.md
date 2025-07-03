# Breast Cancer Prediction Project - Changelog

## Version 0.0.4 (2025-07-03)

### Added
- Data validation functionality:
  - Created `validate_dataset()` function in `src/data_preprocessing.py`
  - Checks for expected columns, data types, and valid diagnosis values
  - Returns validation status and detailed error messages
  - Integrated with logging system

- Missing data handling:
  - Added `handle_missing_data()` function in `src/data_preprocessing.py`
  - Drops rows with missing target values
  - Drops columns with >30% missing values
  - Performs median imputation for remaining missing numerical values
  - Logs detailed statistics on missing data

- Data versioning system:
  - Implemented `create_dataset_version()` in `src/data_preprocessing.py`
  - Generates hash-based version IDs for dataset snapshots
  - Records comprehensive metadata including shape, types, and class distribution
  - Saves versioned datasets as CSV with JSON metadata
  - Ensures JSON serializability by converting NumPy/pandas types

- Interactive visualizations:
  - Added Plotly-based interactive visualization capabilities
  - Created feature distribution plots with color-coding by class
  - Added interactive scatter matrix for feature relationships
  - Implemented 3D scatter plots for multi-dimensional analysis
  - Created ROC curves with confidence intervals using bootstrapping
  - Added descriptive captions to all visualizations

### Changed
- Enhanced standard visualizations with explanatory captions:
  - Updated `plot_standard_visualizations()` function
  - Added informative text explaining how to interpret each visualization
  - Improved color schemes and layout for better readability

- Updated notebook implementation:
  - Restructured `notebooks/notebook.py` to use new functionality
  - Added interactive visualization section with browser support
  - Implemented buttons for opening visualizations directly from notebook
  - Added detailed markdown documentation for each section

## Version 0.0.3 (2025-07-01)

### Added
- Advanced data visualization module:
  - Created [src/visualization.py](cci:7://file:///Users/carlo/Developer/notebooks/Breast-Cancer-Prediction/src/visualization.py:0:0-0:0) with specialized visualization functions
  - Standardized plots (histograms, boxplots, correlation matrices)
  - Feature distribution by class visualizations
  - ROC curves with confidence intervals using bootstrap sampling
  - Consistent styling and output formatting
  - Integration with the logging system

### Changed
- Enhanced data preprocessing module to use the new visualization capabilities
- Improved code modularity with better separation of concerns

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
