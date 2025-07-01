import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from src.logger import setup_logger

logger = setup_logger('visualization')

def create_output_dir(output_dir='../reports/figures'):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Correlation Matrix
def plot_standard_visualizations(df, feature_cols, target_col='diagnosis', output_dir='../reports/figures'):
    logger.info("Creating standard visualizations.")
    output_dir = create_output_dir(output_dir)

    plt.figure(figsize=(12, 10))
    corr_matrix = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmax=.9, vmin=-.9,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()

    display_features = feature_cols[:6] if len(feature_cols) > 6 else feature_cols
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(display_features):
        plt.subplot(2, 3, i+1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_histograms.png")
    plt.close()
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(display_features):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=df[feature])
        plt.title(f'Box Plot of {feature}')
        plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_boxplots.png")
    plt.close()

    logger.info("Standard visualizations created and saved.")

def plot_feature_distribution_by_class(df, feature_cols, target_col='diagnosis', output_dir='../reports/figures'):
    logger.info("Creating feature distribution by class visualizations.")
    output_dir = create_output_dir(output_dir)

    display_features = feature_cols[:6] if len(feature_cols) > 6 else feature_cols

    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(display_features):
        plt.subplot(2, 3, i+1)
    
        for target_val in df[target_col].unique():
            sns.kdeplot(
                df[df[target_col] == target_val][feature],
                label=f"Class {target_val}"
            )
            
        plt.title(f'{feature} Distribution by Class')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distribution_by_class.png")
    plt.close()

    pair_df = df[[target_col] + display_features].copy()
    pair_df[target_col] = pair_df[target_col].astype(str)
    
    plt.figure(figsize=(12, 10))
    pair_plot = sns.pairplot(pair_df, hue=target_col, palette="Set1", 
                           plot_kws={'alpha': 0.6}, diag_kind='kde', 
                           diag_kws={'alpha': 0.6})
    pair_plot.fig.suptitle('Feature Relationships by Class', y=1.02)
    plt.savefig(f"{output_dir}/feature_pairplot_by_class.png")
    plt.close()
    
    logger.info("Feature distribution visualizations created and saved.")

def plot_roc_curves_with_ci(y_test, y_probs_dict, n_bootstraps=1000, output_dir='../reports/figures'):
    logger.info("Creating ROC curves with confidence intervals.")
    output_dir = create_output_dir(output_dir)

    plt.figure(figsize=(10, 8))
    
    for model_name, y_proba in y_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, alpha=0.8,
                 label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        n_samples = len(y_test)
        tprs = []
        aucs = []

        rng = np.random.RandomState(42)
        for i in range(n_bootstraps):
            indices = rng.randint(0, n_samples, n_samples)
            
            if len(np.unique(y_test[indices])) < 2:
                continue
                
            fpr_bootstrap, tpr_bootstrap, _ = roc_curve(y_test[indices], y_proba[indices])
            tprs.append(np.interp(np.linspace(0, 1, 100), fpr_bootstrap, tpr_bootstrap))
            tprs[-1][0] = 0.0
            aucs.append(auc(fpr_bootstrap, tpr_bootstrap))

        mean_tprs = np.mean(tprs, axis=0)
        std_tprs = np.std(tprs, axis=0)
        
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
        
        plt.fill_between(np.linspace(0, 1, 100), tprs_lower, tprs_upper, 
                         alpha=0.2, label=f'{model_name} 95% CI')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with Confidence Intervals')
    plt.legend(loc="lower right")
    
    plt.savefig(f"{output_dir}/roc_curves_with_ci.png")
    plt.close()
    
    logger.info("ROC curves with confidence intervals created and saved.")
