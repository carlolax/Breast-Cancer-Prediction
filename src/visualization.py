import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc
from src.logger import setup_logger

logger = setup_logger('visualization')

def create_output_dir(output_dir='../reports/figures'):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_standard_visualizations(df, feature_cols, target_col='diagnosis', output_dir='../reports/figures'):
    logger.info("Creating standard visualizations.")
    output_dir = create_output_dir(output_dir)

    plt.figure(figsize=(12, 10))
    corr_matrix = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmax=.9, vmin=-.9,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    
    plt.figtext(0.5, 0.01, 
                "This correlation matrix shows relationships between features.\n"
                "Strong positive correlations (close to 1) appear in dark red, while strong negative correlations\n"
                "appear in dark blue. Look for clusters of correlated features that might be redundant.", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", bbox_inches='tight')
    plt.close()

    display_features = feature_cols[:6] if len(feature_cols) > 6 else feature_cols
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(display_features):
        plt.subplot(2, 3, i+1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
    
    plt.figtext(0.5, 0.01, 
                "These histograms show the distribution of each feature.\n"
                "Look for normal (bell curve) or skewed distributions, and check for potential outliers\n"
                "at the extreme ends of the distributions.", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    plt.savefig(f"{output_dir}/feature_histograms.png", bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(display_features):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=df[feature])
        plt.title(f'Box Plot of {feature}')
        plt.tight_layout()
        
    plt.figtext(0.5, 0.01, 
                "Box plots show the median, quartiles, and potential outliers for each feature.\n"
                "The box represents the interquartile range (IQR), the whiskers extend to 1.5 * IQR,\n"
                "and points beyond the whiskers are potential outliers.", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    plt.savefig(f"{output_dir}/feature_boxplots.png", bbox_inches='tight')
    plt.close()

    create_interactive_visualizations(df, feature_cols, target_col, output_dir)
    
    logger.info("Standard visualizations created and saved.")

def create_interactive_visualizations(df, feature_cols, target_col='diagnosis', output_dir='../reports/figures'):
    logger.info("Creating interactive visualizations with Plotly.")
    
    plot_df = df.copy()
    plot_df[target_col] = plot_df[target_col].map({1: 'Malignant', 0: 'Benign'})    
    display_features = feature_cols[:6] if len(feature_cols) > 6 else feature_cols    
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f'{feat}' for feat in display_features])
    
    for i, feature in enumerate(display_features):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        for target_val in plot_df[target_col].unique():
            data = plot_df[plot_df[target_col] == target_val][feature]
            
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=target_val,
                    opacity=0.7,
                    histnorm='probability density',
                    nbinsx=20
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text='Interactive Feature Distributions by Class',
        height=800,
        width=1000,
        showlegend=True,
        template="plotly_white",
        annotations=[
            dict(
                x=0.5,
                y=-0.12,
                xref="paper",
                yref="paper",
                text="Look for features where the distributions are well-separated between classes.<br>"
                     "These features are likely to be more useful for classification.",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 165, 0, 0.1)",
                bordercolor="orange",
                borderwidth=1,
                borderpad=5
            )
        ]
    )
    
    fig.write_html(f"{output_dir}/interactive_distributions.html")
    
    scatter_features = display_features[:4]
    fig = px.scatter_matrix(
        plot_df, 
        dimensions=scatter_features,
        color=target_col,
        symbol=target_col,
        title="Interactive Scatter Matrix",
        labels={col: col.replace('_', ' ').capitalize() for col in scatter_features},
        height=800, width=800
    )
    
    fig.update_layout(
        template="plotly_white",
        annotations=[
            dict(
                x=0.5,
                y=-0.12,
                xref="paper",
                yref="paper",
                text="This scatter matrix shows relationships between pairs of features.<br>"
                     "Look for feature combinations where the classes (benign/malignant) form separate clusters.",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 165, 0, 0.1)",
                bordercolor="orange",
                borderwidth=1,
                borderpad=5
            )
        ]
    )
    
    fig.write_html(f"{output_dir}/interactive_scatter_matrix.html")
    
    if len(display_features) >= 3:
        fig = px.scatter_3d(
            plot_df, 
            x=display_features[0],
            y=display_features[1],
            z=display_features[2],
            color=target_col,
            symbol=target_col,
            title="Interactive 3D Visualization of Top Features",
            labels={
                display_features[0]: display_features[0].replace('_', ' ').capitalize(),
                display_features[1]: display_features[1].replace('_', ' ').capitalize(),
                display_features[2]: display_features[2].replace('_', ' ').capitalize(),
            }
        )
        
        fig.update_layout(
            height=800,
            width=900,
            template="plotly_white",
            annotations=[
                dict(
                    x=0.5,
                    y=-0.1,
                    xref="paper",
                    yref="paper",
                    text="This 3D plot shows the relationship between three key features.<br>"
                         "Rotate the plot to find perspectives where the classes are well-separated.",
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255, 165, 0, 0.1)",
                    bordercolor="orange",
                    borderwidth=1,
                    borderpad=5
                )
            ]
        )
        
        fig.write_html(f"{output_dir}/interactive_3d_scatter.html")
    
    logger.info("Interactive visualizations created and saved to HTML files.")
    
def plot_feature_distribution_by_class(df, feature_cols, target_col='diagnosis', output_dir='../reports/figures'):
    logger.info("Creating feature distribution by class visualizations.")
    output_dir = create_output_dir(output_dir)

    display_features = feature_cols[:6] if len(feature_cols) > 6 else feature_cols

    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(display_features):
        plt.subplot(2, 3, i+1)
    
        for target_val in df[target_col].unique():
            label = 'Malignant' if target_val == 1 else 'Benign'
            sns.kdeplot(
                df[df[target_col] == target_val][feature],
                label=label
            )
            
        plt.title(f'{feature} Distribution by Class')
        plt.legend()
    
    plt.figtext(0.5, 0.01, 
                "These density plots show the distribution of each feature by class (Benign vs. Malignant).\n"
                "Features with clearly separated distributions are strong predictors for classification.\n"
                "Look for features where the two class distributions have minimal overlap.", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distribution_by_class.png", bbox_inches='tight')
    plt.close()

    pair_df = df[[target_col] + display_features].copy()
    pair_df[target_col] = pair_df[target_col].map({1: 'Malignant', 0: 'Benign'})
    
    plt.figure(figsize=(12, 10))
    pair_plot = sns.pairplot(pair_df, hue=target_col, palette="Set1", 
                           plot_kws={'alpha': 0.6}, diag_kind='kde', 
                           diag_kws={'alpha': 0.6})
    pair_plot.fig.suptitle('Feature Relationships by Class', y=1.02)
    
    plt.figure(figsize=(12, 1))
    plt.axis('off')
    plt.text(0.5, 0.5, 
             "This pairplot shows relationships between all pairs of features, colored by class.\n"
             "Look for combinations of features where the red and blue points form separate clusters.\n"
             "These feature combinations are likely to be most useful for classification.", 
             ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pairplot_caption.png", bbox_inches='tight')
    
    plt.close('all')
    
    logger.info("Feature distribution visualizations created and saved.")

def plot_roc_curves_with_ci(y_test, y_probs_dict, n_bootstraps=1000, output_dir='../reports/figures'):
    logger.info("Creating ROC curves with confidence intervals.")
    output_dir = create_output_dir(output_dir)

    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    else:
        y_test = np.asarray(y_test)
    
    plt.figure(figsize=(10, 8))
    
    for model_name, y_proba in y_probs_dict.items():
        if isinstance(y_proba, pd.Series):
            y_proba = y_proba.values
        else:
            y_proba = np.asarray(y_proba)
            
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, alpha=0.8,
                 label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        n_samples = len(y_test)
        tprs = []
        aucs = []

        rng = np.random.RandomState(42)
        for i in range(n_bootstraps):
            bootstrap_indices = rng.randint(0, n_samples, n_samples)            
            y_test_bootstrap = y_test[bootstrap_indices]
            y_proba_bootstrap = y_proba[bootstrap_indices]
            
            if len(np.unique(y_test_bootstrap)) < 2:
                continue
                
            fpr_bootstrap, tpr_bootstrap, _ = roc_curve(y_test_bootstrap, y_proba_bootstrap)
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
    
    plt.figtext(0.5, 0.01, 
                "ROC curves show the trade-off between true positive rate and false positive rate.\n"
                "A perfect classifier would reach the top-left corner (TPR=1, FPR=0).\n"
                "The Area Under the Curve (AUC) is a measure of model performance: higher is better.\n"
                "The shaded areas represent 95% confidence intervals from bootstrapping.", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    plt.savefig(f"{output_dir}/roc_curves_with_ci.png", bbox_inches='tight')
    plt.close()
    
    fig = go.Figure()
    
    for model_name, y_proba in y_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{model_name} (AUC = {roc_auc:.2f})',
            mode='lines',
            line=dict(width=2),
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='black'),
        name='Random Classifier'
    ))
    
    fig.update_layout(
        title='Interactive ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=800, height=700,
        legend=dict(x=0.6, y=0.1),
        template="plotly_white",
        annotations=[
            dict(
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                text="The ROC curve shows the trade-off between the true positive rate and false positive rate.<br>"
                     "A perfect classifier would reach the top-left corner (TPR=1, FPR=0).<br>"
                     "The Area Under the Curve (AUC) is a measure of model performance: higher is better.",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 165, 0, 0.1)",
                bordercolor="orange",
                borderwidth=1,
                borderpad=5
            )
        ]
    )
    
    fig.write_html(f"{output_dir}/interactive_roc_curves.html")
    
    logger.info("ROC curves with confidence intervals created and saved.")
