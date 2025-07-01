import os
import numpy as np
import pandas as pd

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
        return df
    except Exception as exception:
        logger.error(f"Error loading data: {exception}")
        return None

def explore_dataset(df):
    print("\Basic Information:")
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

    logger.info("Preprocessed data saved in '../data/processed' directory.")
