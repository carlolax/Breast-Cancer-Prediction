import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
from src.logger import setup_logger

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, make_scorer
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

logger = setup_logger('model-training')

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
    logger.info("Starting model training.")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name} model.")
        print(f"Training {name} model.")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            logger.info(f"{name} model trained completed.")
        except Exception as exception:
            logger.error(f"Error training {name} model: {str(exception)}")
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

    logger.info("Visualizations saved in '../reports/figures' directory.")

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

def implement_ensemble_methods(X_train, y_train, X_test, y_test):
    os.makedirs('../reports/ensemble_methods', exist_ok=True)

    base_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    print("Training base models.")
    base_results = []

    for name, model in base_models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        base_results.append({
            'Model': name,
            'Type': 'Base Model',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    print("\nImplementing Voting Classifiers.")

    voting_hard = VotingClassifier(
        estimators=[
            ('lr', base_models['Logistic Regression']),
            ('dt', base_models['Decision Tree']),
            ('svm', base_models['SVM']),
            ('knn', base_models['KNN'])
        ],
        voting='hard'
    )

    voting_soft = VotingClassifier(
        estimators=[
            ('lr', base_models['Logistic Regression']),
            ('dt', base_models['Decision Tree']),
            ('svm', base_models['SVM']),
            ('knn', base_models['KNN'])
        ],
        voting='soft'
    )

    voting_models = {
        'Voting (Hard)': voting_hard,
        'Voting (Soft)': voting_soft
    }

    for name, model in voting_models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        base_results.append({
            'Model': name,
            'Type': 'Ensemble - Voting',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    print("\nImplementing Bagging.")

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=False,
        random_state=42
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    bagging_models = {
        'Bagging (Decision Trees)': bagging,
        'Random Forest': random_forest
    }

    for name, model in bagging_models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        base_results.append({
            'Model': name,
            'Type': 'Ensemble - Bagging',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    print("\nImplementing Boosting.")

    adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    boosting_models = {
        'AdaBoost': adaboost,
        'Gradient Boosting': gradient_boosting
    }

    for name, model in boosting_models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        base_results.append({
            'Model': name,
            'Type': 'Ensemble - Boosting',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    print("\nImplementing Stacking.")

    stacking = StackingClassifier(
        estimators=[
            ('lr', base_models['Logistic Regression']),
            ('dt', base_models['Decision Tree']),
            ('svm', base_models['SVM']),
            ('knn', base_models['KNN'])
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )

    stacking.fit(X_train, y_train)

    y_pred = stacking.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    base_results.append({
        'Model': 'Stacking',
        'Type': 'Ensemble - Stacking',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    print(f"Stacking - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    results_df = pd.DataFrame(base_results)

    visualize_ensemble_results(results_df)

    plot_roc_curves(
        {**base_models, **voting_models, **bagging_models, **boosting_models, 'Stacking': stacking},
        X_test, y_test
    )

    results_df.to_csv('../reports/ensemble_methods/ensemble_comparison.csv', index=False)

    all_models = {
        **base_models,
        **voting_models,
        **bagging_models,
        **boosting_models,
        'Stacking': stacking
    }

    return all_models

def visualize_ensemble_results(results_df):

    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)

        sns.barplot(
            x='Model', 
            y=metric, 
            hue='Type',
            data=results_df,
            palette='viridis'
        )

        plt.title(f'Comparison of {metric}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model Type')
        plt.tight_layout()

    plt.suptitle('Ensemble Methods vs Base Models', fontsize=16, y=1.02)
    plt.savefig('../reports/ensemble_methods/ensemble_comparison.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))

    heatmap_data = results_df.set_index('Model')[metrics]

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Performance Metrics Across All Models')
    plt.tight_layout()
    plt.savefig('../reports/ensemble_methods/metrics_heatmap.png')
    plt.close()

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(12, 10))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, color=color, 
                     label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('../reports/ensemble_methods/roc_curves.png')
    plt.close()

    print("ROC curves saved to '../reports/ensemble_methods/roc_curves.png'")

def perform_cross_validation(X, y, models, cv_folds=5, stratified=True):
    os.makedirs('../reports/cross_validation', exist_ok=True)

    if stratified:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_type = 'Stratified'
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_type = 'Standard'

    print(f"Performing {cv_type} {cv_folds}-fold cross-validation")

    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    cv_results = []

    visualize_cv_splits(X, y, cv)

    for name, model in models.items():
        print(f"\nCross-validating {name}.")

        scores = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )

        result = {metric: {} for metric in scoring.keys()}
        print(f"Results for {name}:")

        for metric in scoring.keys():
            train_scores = scores[f'train_{metric}']
            test_scores = scores[f'test_{metric}']

            train_mean = np.mean(train_scores)
            train_std = np.std(train_scores)
            test_mean = np.mean(test_scores)
            test_std = np.std(test_scores)

            result[metric] = {
                'train_mean': train_mean,
                'train_std': train_std,
                'test_mean': test_mean,
                'test_std': test_std
            }

            print(f"  {metric.capitalize()}:")
            print(f"    Train: {train_mean:.4f} ± {train_std:.4f}")
            print(f"    Test:  {test_mean:.4f} ± {test_std:.4f}")

        cv_results.append({
            'Model': name,
            'Accuracy': result['accuracy']['test_mean'],
            'Accuracy_std': result['accuracy']['test_std'],
            'Precision': result['precision']['test_mean'],
            'Precision_std': result['precision']['test_std'],
            'Recall': result['recall']['test_mean'],
            'Recall_std': result['recall']['test_std'],
            'F1': result['f1']['test_mean'],
            'F1_std': result['f1']['test_std']
        })

        plot_fold_performance(scores, name, cv_folds)

    results_df = pd.DataFrame(cv_results)

    plot_model_comparison(results_df)

    results_df.to_csv('../reports/cross_validation/cv_results.csv', index=False)

    return results_df

def visualize_cv_splits(X, y, cv):
    plt.figure(figsize=(12, cv.n_splits * 1.2))

    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    class_colors = {cls: color for cls, color in zip(classes, colors)}

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        plt.subplot(cv.n_splits, 1, i+1)

        train_classes = y[train_idx]
        test_classes = y[test_idx]

        for cls in classes:
            cls_indices = train_idx[train_classes == cls]
            if len(cls_indices) > 0:
                plt.scatter(
                    cls_indices, 
                    np.ones(len(cls_indices)) * 0,
                    c=[class_colors[cls]], 
                    marker='|', 
                    s=20,
                    alpha=0.7,
                    label=f'Train - Class {cls}' if i == 0 else None
                )

        for cls in classes:
            cls_indices = test_idx[test_classes == cls]
            if len(cls_indices) > 0:
                plt.scatter(
                    cls_indices, 
                    np.ones(len(cls_indices)) * 0,
                    c=[class_colors[cls]], 
                    marker='o', 
                    s=30,
                    edgecolors='black',
                    label=f'Validation - Class {cls}' if i == 0 else None
                )

        plt.yticks([])
        plt.title(f'Fold {i+1}', loc='right')

        class_dist_train = {cls: np.sum(train_classes == cls) for cls in classes}
        class_dist_test = {cls: np.sum(test_classes == cls) for cls in classes}

        class_info = ', '.join([f"Class {cls}: {class_dist_train[cls]}/{class_dist_test[cls]} (train/val)" for cls in classes])
        plt.text(0.02, 0.5, class_info, transform=plt.gca().transAxes, fontsize=8)

        if i == 0:
            plt.legend(loc='upper right', ncol=len(classes))

        if i == cv.n_splits - 1:
            plt.xlabel('Data points')

    plt.suptitle('Cross-Validation Splits with Class Distribution')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('../reports/cross_validation/cv_splits.png')
    plt.close()

def plot_fold_performance(scores, model_name, cv_folds):
    plt.figure(figsize=(10, 8))

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)

        plt.plot(range(1, cv_folds+1), scores[f'train_{metric}'], 
                 'o-', label=f'Training {metric}')
        plt.plot(range(1, cv_folds+1), scores[f'test_{metric}'], 
                 'o-', label=f'Validation {metric}')

        plt.xlabel('Fold')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'{metric.capitalize()} Across Folds')
        plt.xticks(range(1, cv_folds+1))
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.suptitle(f'Cross-Validation Performance: {model_name}', y=1.02)
    plt.savefig(f'../reports/cross_validation/{model_name.replace(" ", "_")}_cv_performance.png')
    plt.close()

def plot_model_comparison(results_df):
    plt.figure(figsize=(12, 8))

    barWidth = 0.2
    r1 = np.arange(len(results_df['Model']))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    
    plt.bar(r1, results_df['Accuracy'], width=barWidth, yerr=results_df['Accuracy_std'],
           label='Accuracy', color='skyblue', edgecolor='black', capsize=7)
    plt.bar(r2, results_df['Precision'], width=barWidth, yerr=results_df['Precision_std'],
           label='Precision', color='lightgreen', edgecolor='black', capsize=7)
    plt.bar(r3, results_df['Recall'], width=barWidth, yerr=results_df['Recall_std'],
           label='Recall', color='salmon', edgecolor='black', capsize=7)
    plt.bar(r4, results_df['F1'], width=barWidth, yerr=results_df['F1_std'],
           label='F1 Score', color='purple', edgecolor='black', capsize=7)

    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.title('Cross-Validation Performance by Model')
    plt.xticks([r + barWidth*1.5 for r in range(len(results_df['Model']))], 
               results_df['Model'], rotation=45, ha='right')
    plt.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.savefig('../reports/cross_validation/model_comparison.png')
    plt.close()
