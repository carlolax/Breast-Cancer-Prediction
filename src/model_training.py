import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

def load_processed_data(data_dir='../data/processed'):
    X_train = np.load(f"{data_dir}/X_train.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")

    print("Preprocessed data loaded successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name} model.")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} model trained completed.\n")

    return trained_models

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nEvaluation results for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred
    }

def visualize_results(models_results, X_test, y_test):
    os.makedirs('../reports/figures', exist_ok=True)

    model_names = [result['model_name'] for result in models_results]
    accuracies = [result['accuracy'] for result in models_results]
    precisions = [result['precision'] for result in models_results]
    recalls = [result['recall'] for result in models_results]
    f1_scores = [result['f1'] for result in models_results]

    plt.figure(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.2

    plt.bar(x - 1.5*width, accuracies, width, label='Accuracy')
    plt.bar(x - 0.5*width, precisions, width, label='Precision')
    plt.bar(x + 0.5*width, recalls, width, label='Recall')
    plt.bar(x + 1.5*width, f1_scores, width, label='F1 Score')

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../reports/figures/model_comparison.png')

    for result in models_results:
        model_name = result['model_name']
        y_pred = result['y_pred']

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'../reports/figures/confusion_matrix_{model_name.replace(" ", "_")}.png')

    plt.figure(figsize=(10, 8))

    for i, result in enumerate(models_results):
        model = result['model']
        model_name = result['model_name']

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('../reports/figures/roc_curves.png')

    print("Visualizations saved in '../reports/figures' directory.")

def save_best_model(models_results, output_dir='../models'):
    os.makedirs(output_dir, exist_ok=True)

    best_model_result = max(models_results, key=lambda x: x['f1'])
    best_model = best_model_result['model']
    best_model_name = best_model_result['model_name'].replace(' ', '_').lower()

    model_path = f"{output_dir}/{best_model_name}.joblib"
    dump(best_model, model_path)

    model_info = {
        'name': best_model_result['model_name'],
        'accuracy': best_model_result['accuracy'],
        'precision': best_model_result['precision'],
        'recall': best_model_result['recall'],
        'f1': best_model_result['f1']        
    }

    pd.DataFrame([model_info]).to_csv(f"{output_dir}/{best_model_name}_metrics.csv", index=False)

    print(f"Best model ({best_model_result['model_name']}) saved to {model_path}")
    print("Model metrics:")
    for metric, value in model_info.items():
        if metric != 'name':
            print(f"  {metric}: {value:.4f}")

def feature_importance(models_results, X_train, feature_names=None):
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    if len(feature_names) != X_train.shape[1]:
        print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of features ({X_train.shape[1]})")
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    tree_models = [result for result in models_results if 
                  result['model_name'] in ['Random Forest', 'Decision Tree']]
    
    if tree_models:
        os.makedirs('../reports/figures', exist_ok=True)
        
        for result in tree_models:
            model = result['model']
            model_name = result['model_name']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(12, 8))
                plt.title(f'Feature Importance - {model_name}')
                plt.bar(range(len(indices)), importances[indices], align='center')
                plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig(f'../reports/figures/feature_importance_{model_name.replace(" ", "_")}.png')
                
                print(f"\nTop 10 important features for {model_name}:")
                for i in range(min(10, len(feature_names))):
                    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    else:
        print("No tree-based models found in results. Feature importance is only available for Random Forest and Decision Tree models.")
