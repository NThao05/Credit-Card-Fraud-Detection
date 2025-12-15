"""
Visualization Module - Matplotlib and Seaborn
Functions for creating visualizations of data and results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """
    Plot class distribution as bar chart and pie chart
    
    Parameters:
    -----------
    y : ndarray
        Labels
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    unique, counts = np.unique(y, return_counts=True)
    
    # Check for NaN or invalid data
    if np.isnan(unique).any():
        print("[ERROR] Cannot plot - target contains NaN values")
        print("Please restart kernel and reload data with the fixed load_csv_numpy function")
        return
    
    if len(unique) < 2:
        print(f"[WARNING] Only {len(unique)} class found - cannot create comparison plots")
        print("Please check data loading")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    axes[0].bar(unique, counts, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{title} - Bar Chart')
    axes[0].set_xticks(unique)
    axes[0].set_xticklabels(['Legitimate (0)', 'Fraud (1)'])
    
    # Add value labels on bars
    for i, (cls, count) in enumerate(zip(unique, counts)):
        axes[0].text(cls, count, f'{count}\n({count/len(y)*100:.2f}%)', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)  # Explode fraud slice
    axes[1].pie(counts, labels=['Legitimate (0)', 'Fraud (1)'], autopct='%1.2f%%',
               colors=colors, explode=explode, startangle=90, shadow=True)
    axes[1].set_title(f'{title} - Pie Chart')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_feature_distributions(X, feature_names=None, n_features=6, save_path=None):
    """
    Plot distributions of multiple features
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    feature_names : list, optional
        List of feature names
    n_features : int
        Number of features to plot
    save_path : str, optional
        Path to save figure
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    n_features = min(n_features, X.shape[1])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(n_features):
        axes[i].hist(X[:, i], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{feature_names[i]}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(X, feature_names=None, save_path=None):
    """
    Plot correlation matrix as heatmap
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    feature_names : list, optional
        List of feature names
    save_path : str, optional
        Path to save figure
    """
    from data_processing import correlation_matrix
    
    corr = correlation_matrix(X)
    
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(X.shape[1])]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importances, feature_names, top_n=10, save_path=None):
    """
    Plot feature importance
    
    Parameters:
    -----------
    importances : ndarray
        Feature importance scores
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to show
    save_path : str, optional
        Path to save figure
    """
    # Sort features by importance
    indices = np.argsort(np.abs(importances))[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), np.abs(importances[indices]), color='steelblue', alpha=0.7)
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names=['Legitimate', 'Fraud'], save_path=None, ax=None, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : ndarray
        Confusion matrix
    class_names : list
        List of class names
    save_path : str, optional
        Path to save figure
    ax : matplotlib axis, optional
        Axis to plot on (for subplots)
    title : str
        Title for the plot
    """
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black',
                ax=ax)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add percentages
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    if ax is None:
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if ax is None:
        plt.show()


def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    fpr : ndarray
        False positive rates
    tpr : ndarray
        True positive rates
    auc_score : float
        AUC score
    save_path : str, optional
        Path to save figure
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(precision, recall, ap_score=None, save_path=None):
    """
    Plot precision-recall curve
    
    Parameters:
    -----------
    precision : ndarray
        Precision values
    recall : ndarray
        Recall values
    ap_score : float, optional
        Average precision score
    save_path : str, optional
        Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    if ap_score is not None:
        label = f'PR curve (AP = {ap_score:.4f})'
    else:
        label = 'PR curve'
    
    plt.plot(recall, precision, color='purple', lw=2, label=label)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_loss_history(loss_history, save_path=None):
    """
    Plot training loss history
    
    Parameters:
    -----------
    loss_history : list or ndarray
        Loss values over iterations
    save_path : str, optional
        Path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, color='blue', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Iterations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot comparison of multiple metrics
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metric names and values
    save_path : str, optional
        Path to save figure
    """
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_fraud_vs_legit_feature(X, y, feature_idx, feature_name=None, save_path=None):
    """
    Plot distribution of a feature for fraud vs legitimate transactions
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Labels
    feature_idx : int
        Index of feature to plot
    feature_name : str, optional
        Name of feature
    save_path : str, optional
        Path to save figure
    """
    if feature_name is None:
        feature_name = f'Feature {feature_idx}'
    
    fraud_data = X[y == 1, feature_idx]
    legit_data = X[y == 0, feature_idx]
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(legit_data, bins=50, alpha=0.7, label='Legitimate', color='green', edgecolor='black')
    plt.hist(fraud_data, bins=50, alpha=0.7, label='Fraud', color='red', edgecolor='black')
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{feature_name} Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = [legit_data, fraud_data]
    bp = plt.boxplot(data_to_plot, labels=['Legitimate', 'Fraud'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.7)
    plt.ylabel(feature_name, fontsize=12)
    plt.title(f'{feature_name} Box Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_scatter_2d(X, y, feature_idx1, feature_idx2, feature_names=None, save_path=None):
    """
    Plot 2D scatter plot of two features
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Labels
    feature_idx1 : int
        Index of first feature
    feature_idx2 : int
        Index of second feature
    feature_names : list, optional
        List of feature names
    save_path : str, optional
        Path to save figure
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    plt.figure(figsize=(10, 8))
    
    # Plot legitimate transactions
    legit_mask = y == 0
    plt.scatter(X[legit_mask, feature_idx1], X[legit_mask, feature_idx2], 
               c='green', alpha=0.3, s=10, label='Legitimate', edgecolors='none')
    
    # Plot fraudulent transactions
    fraud_mask = y == 1
    plt.scatter(X[fraud_mask, feature_idx1], X[fraud_mask, feature_idx2], 
               c='red', alpha=0.8, s=30, label='Fraud', edgecolors='black', linewidth=0.5)
    
    plt.xlabel(feature_names[feature_idx1], fontsize=12)
    plt.ylabel(feature_names[feature_idx2], fontsize=12)
    plt.title(f'{feature_names[feature_idx1]} vs {feature_names[feature_idx2]}', 
             fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_time_distribution(time_feature, y, save_path=None):
    """
    Plot transaction distribution over time
    
    Parameters:
    -----------
    time_feature : ndarray
        Time feature (in seconds)
    y : ndarray
        Labels
    save_path : str, optional
        Path to save figure
    """
    # Convert seconds to hours
    time_hours = time_feature / 3600
    
    plt.figure(figsize=(14, 6))
    
    # Histogram for all transactions
    plt.subplot(1, 2, 1)
    plt.hist(time_hours, bins=48, color='skyblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.title('Transaction Distribution Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Separate distributions for fraud vs legitimate
    plt.subplot(1, 2, 2)
    plt.hist(time_hours[y == 0], bins=48, alpha=0.6, label='Legitimate', 
            color='green', edgecolor='black')
    plt.hist(time_hours[y == 1], bins=48, alpha=0.8, label='Fraud', 
            color='red', edgecolor='black')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.title('Fraud vs Legitimate Over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_amount_distribution(amount_feature, y, save_path=None):
    """
    Plot transaction amount distribution
    
    Parameters:
    -----------
    amount_feature : ndarray
        Amount feature
    y : ndarray
        Labels
    save_path : str, optional
        Path to save figure
    """
    fraud_amounts = amount_feature[y == 1]
    legit_amounts = amount_feature[y == 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram - Legitimate
    axes[0, 0].hist(legit_amounts, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Amount ($)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Legitimate Transaction Amounts', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram - Fraud
    axes[0, 1].hist(fraud_amounts, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Amount ($)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Fraudulent Transaction Amounts', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot comparison
    axes[1, 0].boxplot([legit_amounts, fraud_amounts], labels=['Legitimate', 'Fraud'], 
                       patch_artist=True)
    axes[1, 0].set_ylabel('Amount ($)', fontsize=12)
    axes[1, 0].set_title('Amount Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Statistics table
    axes[1, 1].axis('off')
    stats_text = f"""
    STATISTICS SUMMARY
    
    Legitimate Transactions:
    Mean: ${np.mean(legit_amounts):.2f}
    Median: ${np.median(legit_amounts):.2f}
    Std Dev: ${np.std(legit_amounts):.2f}
    Max: ${np.max(legit_amounts):.2f}
    
    Fraudulent Transactions:
    Mean: ${np.mean(fraud_amounts):.2f}
    Median: ${np.median(fraud_amounts):.2f}
    Std Dev: ${np.std(fraud_amounts):.2f}
    Max: ${np.max(fraud_amounts):.2f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test visualizations
    print("Visualization Module - Testing")
    
    # Create sample data
    np.random.seed(42)
    y = np.array([0] * 900 + [1] * 100)
    
    # Test class distribution plot
    plot_class_distribution(y, "Sample Class Distribution")
    
    print("\nVisualization module working correctly!")
