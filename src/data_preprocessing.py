import os
import numpy as np
import pandas as pd
import hashlib
import json
import datetime

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.logger import setup_logger
from src.visualization import (
    plot_standard_visualizations,
    plot_feature_distribution_by_class
)

logger = setup_logger('data-preprocessing')

def load_dataset(data_path):
    column_names = ['id', 'diagnosis'] + [
        f'{feature}_{stat}' for feature in 
        ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
         'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'] 
        for stat in ['mean', 'se', 'worst']
    ]

    try:
        df = pd.read_csv(data_path, header=None, names=column_names)
        # Convert diagnosis to binary: Malignant (1) vs Benign (0)
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        logger.info(f"Data loaded successfully with shape: {df.shape}")
        
        if not validate_dataset(df):
            logger.warning("Dataset validation failed. Proceed with caution.")
        
        raw_version_id = create_dataset_version(df, "raw_data")
        logger.info(f"Raw dataset versioned with ID: {raw_version_id}")
        
        return df
    except Exception as exception:
        logger.error(f"Error loading data: {exception}")
        return None

def validate_dataset(df):
    logger.info("Validating dataset structure and features.")
    
    if df is None or df.empty:
        logger.error("Dataset is empty or None")
        return False
    
    expected_features = ['id', 'diagnosis'] + [
        f'{feature}_{stat}' for feature in 
        ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
         'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'] 
        for stat in ['mean', 'se', 'worst']
    ]
    
    missing_cols = set(expected_features) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing expected columns: {missing_cols}")
        return False
    
    if not set(df['diagnosis'].unique()).issubset({'M', 'B', 0, 1}):
        logger.error(f"Invalid diagnosis values found: {df['diagnosis'].unique()}")
        return False
    
    numeric_cols = [col for col in df.columns if col != 'id' and col != 'diagnosis']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"Column {col} should be numeric but is {df[col].dtype}")
            return False
    
    logger.info("Dataset validation passed successfully.")
    return True

def handle_missing_data(df):
    logger.info("Handling missing data.")
    
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0].index.tolist()
    
    if not missing_cols:
        logger.info("No missing values found in the dataset.")
        return df
    
    logger.info(f"Found {len(missing_cols)} columns with missing values: {missing_cols}")
    
    df_clean = df.copy()
    
    for col in missing_cols:
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        if col == 'diagnosis':
            df_clean = df_clean.dropna(subset=[col])
            logger.info(f"Dropped {missing_count} rows with missing diagnosis values")
        elif missing_percent > 30:
            df_clean = df_clean.drop(columns=[col])
            logger.info(f"Dropped column {col} with {missing_percent:.2f}% missing values")
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_value = df[col].median()
                df_clean[col] = df_clean[col].fillna(median_value)
                logger.info(f"Imputed {missing_count} missing values in {col} with median ({median_value:.4f})")
    
    logger.info(f"Missing data handling completed. Rows before: {len(df)}, after: {len(df_clean)}")
    return df_clean

def create_dataset_version(df, dataset_name, output_dir='../data/versions'):
    os.makedirs(output_dir, exist_ok=True)
    
    df_string = df.to_csv(index=False)
    hash_object = hashlib.md5(df_string.encode())
    content_hash = hash_object.hexdigest()
    
    version_info = {
        "dataset_name": dataset_name,
        "version_id": content_hash[:10],
        "timestamp": datetime.datetime.now().isoformat(),
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "has_missing_values": bool(df.isnull().any().any()),
        "class_distribution": {str(k): int(v) for k, v in df['diagnosis'].value_counts().items()} if 'diagnosis' in df.columns else {}
    }
    
    version_file = f"{output_dir}/{dataset_name}_v{content_hash[:10]}.json"
    with open(version_file, 'w') as f:
        json.dump(version_info, f, indent=2)
    
    versioned_data_file = f"{output_dir}/{dataset_name}_v{content_hash[:10]}.csv"
    df.to_csv(versioned_data_file, index=False)
    
    logger.info(f"Created dataset version {content_hash[:10]} saved to {version_file}")
    return content_hash[:10]

def explore_dataset(df):
    print("\nBasic Information:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nClass Distribution:")
    print(df['diagnosis'].value_counts())

    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found in the dataset.")
    
def preprocess_data(df, test_size=0.2, random_state=42):
    df = handle_missing_data(df)
    
    df = df.drop('id', axis=1)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling applied.\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def visualize_data(df):    
    feature_cols = [col for col in df.columns if col not in ['id', 'diagnosis']]

    plot_standard_visualizations(df, feature_cols)
    
    plot_feature_distribution_by_class(df, feature_cols)

def save_processed_dataset(X_train, X_test, y_train, y_test, scaler, output_dir='../data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/X_train.npy", X_train)
    np.save(f"{output_dir}/X_test.npy", X_test)
    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/y_test.npy", y_test)
    dump(scaler, f"{output_dir}/scaler.joblib")
    
    processed_df = pd.DataFrame(X_train)
    processed_df['target'] = y_train
    processed_version_id = create_dataset_version(processed_df, "processed_train_data")
    logger.info(f"Processed training dataset versioned with ID: {processed_version_id}")

    logger.info("Preprocessed data saved in '../data/processed' directory.")
