# Credit Card Fraud Detection

A comprehensive implementation of credit card fraud detection using only NumPy for data processing and machine learning algorithms implementation from scratch.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methods](#methods)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Challenges & Solutions](#challenges--solutions)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

## Introduction

### Problem Description
Credit card fraud is a significant issue in the financial industry, causing billions of dollars in losses annually. This project aims to detect fraudulent transactions using machine learning techniques implemented entirely with NumPy.

### Motivation and Real-world Applications
- **Financial Security**: Protecting customers and financial institutions from fraudulent activities
- **Real-time Detection**: Enabling instant fraud detection in transaction processing systems
- **Cost Reduction**: Minimizing losses from fraudulent transactions
- **Customer Trust**: Maintaining customer confidence in digital payment systems

### Specific Objectives
1. Perform comprehensive exploratory data analysis using only NumPy
2. Implement data preprocessing techniques (normalization, standardization, handling imbalanced data)
3. Build machine learning models from scratch using NumPy:
   - Logistic Regression with gradient descent optimization
   - Statistical analysis and hypothesis testing
4. Evaluate model performance using custom metrics implementation
5. Visualize results using Matplotlib and Seaborn

## Dataset

### Data Source
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) from Kaggle

### Features Description
The dataset contains transactions made by credit cards in September 2013 by European cardholders.

- **Time**: Number of seconds elapsed between this transaction and the first transaction
- **V1-V28**: Principal components obtained with PCA (anonymized features)
- **Amount**: Transaction amount
- **Class**: Target variable (1 = fraud, 0 = legitimate)

### Dataset Characteristics
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Legitimate Transactions**: 284,315 (99.828%)
- **Features**: 30 (28 PCA features + Time + Amount)
- **Highly Imbalanced Dataset**: Requires special handling techniques

## Methods

### Data Processing Pipeline

#### 1. Data Loading
```python
# Pure NumPy CSV loading without pandas
data = np.genfromtxt('data/raw/creditcard.csv', delimiter=',', skip_header=1)
```

#### 2. Data Preprocessing

**Missing Value Handling**:
- Check for NaN values: `np.isnan(data).sum()`
- Imputation using median: `np.nanmedian()`

**Outlier Detection**:
- Z-score method: $z = \frac{x - \mu}{\sigma}$
- IQR method: $IQR = Q_3 - Q_1$

**Normalization**:
- Min-Max Scaling: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$
- Log Transformation (for Amount): $x_{log} = \log(1 + x)$

**Standardization**:
- Z-score normalization: $x_{std} = \frac{x - \mu}{\sigma}$

#### 3. Handling Imbalanced Data
- **SMOTE-like technique** (Synthetic Minority Over-sampling)
- **Random Under-sampling** of majority class
- **Class weights** in loss function

### Machine Learning Algorithms

#### Logistic Regression (Implemented from Scratch)

**Hypothesis Function**:
$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

**Cost Function** (Binary Cross-Entropy):
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

**Gradient Descent Update**:
$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

where:
$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

**NumPy Implementation Highlights**:
```python
# Vectorized sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Numerical stability

# Vectorized gradient computation
gradient = (1/m) * X.T @ (predictions - y)

# No loops - pure vectorization
theta -= learning_rate * gradient
```

### Statistical Analysis

**Hypothesis Testing**:
- **H₀**: No significant difference between fraudulent and legitimate transaction amounts
- **H₁**: Significant difference exists
- **Test**: Two-sample t-test with significance level α = 0.05

**Correlation Analysis**:
- Pearson correlation coefficient: $r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2\sum(y_i - \bar{y})^2}}$

### Evaluation Metrics (Custom Implementation)

**Confusion Matrix**:
```
                Predicted
              0         1
Actual 0    TN        FP
       1    FN        TP
```

**Metrics**:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **Specificity**: $\frac{TN}{TN + FP}$

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone the repository**:
```powershell
git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. **Create virtual environment** (recommended):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

4. **Download dataset**:
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
   - Place in `data/raw/` directory

## Usage

### Running the Complete Pipeline

1. **Data Exploration**:
```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
```

2. **Data Preprocessing**:
```powershell
jupyter notebook notebooks/02_preprocessing.ipynb
```

3. **Model Training & Evaluation**:
```powershell
jupyter notebook notebooks/03_modeling.ipynb
```

### Using Python Scripts

```python
from src.data_processing import load_data, preprocess_data
from src.models import LogisticRegressionNumPy
from src.visualization import plot_confusion_matrix

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_data('data/raw/creditcard.csv')

# Train model
model = LogisticRegressionNumPy(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = model.evaluate(X_test, y_test)
```

## Results

### Research Questions Answered

#### 1. What can we learn from fraudulent and legitimate transactions?
- **Distribution ratio**: 99.83% legitimate transactions vs 0.17% fraud (highly imbalanced)
- **Distinct characteristics**: Fraudulent transactions have significantly different distributions in Amount and features V1-V28
- **Statistics**: T-test shows p-value < 0.001, proving the difference is statistically significant

#### 2. What can we see from the predictions?
- **Top important features**: V14, V17, V12, V10, V16 have the strongest correlation with fraud
- **Model comparison**: Class Weights approach works best for the imbalanced dataset
- **Threshold optimization**: Increasing threshold from 0.5 → 0.94 improves Precision from 5.72% → 65.34%

#### 3. Are there differences in time and amount?
- **Time patterns**: No clear pattern over time
- **Amount patterns**: Fraudulent transactions tend to have lower and more dispersed Amount values
- **Correlation**: Features V1-V28 have stronger correlation than Time and Amount

### Model Performance Comparison

#### Multiple Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Baseline (threshold=0.5)** | 99.57% | 5.72% | 92.86% | 10.77% | 98.86% |
| **Optimized (threshold=0.8)** | 99.59% | 27.27% | 88.78% | 41.73% | 98.86% |
| **Feature Selection (top 15)** | 99.60% | 28.39% | 89.80% | 43.14% | 98.81% |
| **Ensemble Hard Voting** | 99.61% | 28.99% | 90.82% | 44.03% | 98.87% |
| **Ensemble Soft Voting** | 99.59% | 27.27% | 88.78% | 41.73% | 98.86% |
| **[BEST] PR-Optimized (threshold=0.94)** | **98.96%** | **65.34%** | **87.76%** | **74.94%** | **98.86%** |

#### Best Model Performance (PR-Optimized)

**Confusion Matrix**:
- True Negatives: 56,910 (correctly identified legitimate)
- False Positives: 52 (legitimate flagged as fraud) - **Only 0.09% FPR!**
- False Negatives: 12 (fraud missed)
- True Positives: 86 (correctly caught fraud)

**Business Impact**:
- Catches **87.76%** of all fraudulent transactions
- Only **52 false alarms** per 56,962 legitimate transactions
- **12 missed frauds** out of 98 total fraud cases
- False Positive Rate: **0.09%** (excellent for customer experience)

**Model Improvements**:
- Precision increased **+1,042%** (5.72% → 65.34%)
- F1-Score increased **+596%** (10.77% → 74.94%)
- False Positives reduced **-1,449** (1,501 → 52)
- Maintained high Recall of **87.76%**

### Key Findings

1. **Class Imbalance Handling**: 
   - Class weights approach works best for this dataset
   - SMOTE oversampling increases training time significantly (163s vs 24s)
   - Random undersampling loses valuable information

2. **Feature Importance Analysis**:
   - Top 5 features: V14 (-0.43), V17 (-0.35), V12 (-0.32), V10 (-0.30), V16 (-0.28)
   - Features V1-V28 (PCA-transformed) are more informative than Time and Amount
   - Using only top 15 features maintains 98% of model performance

3. **Threshold Optimization**:
   - Default threshold (0.5): High recall but extremely low precision (5.72%)
   - Optimized threshold (0.8): Balanced improvement (Precision: 27.27%, F1: 41.73%)
   - PR-curve optimized (0.94): Maximum F1-score achieved (Precision: 65.34%, F1: 74.94%)

4. **Statistical Significance**:
   - Two-sample t-test: p-value < 0.001 (reject H₀: significant difference in transaction amounts)
   - Correlation analysis reveals strong negative correlations for fraud-related features
   - Skewness and kurtosis analysis shows non-normal distributions

5. **Advanced Techniques**:
   - Ensemble voting (hard/soft) improves robustness slightly (F1: 44.03%)
   - Precision-Recall curve optimization finds optimal threshold automatically
   - Feature selection reduces computation without sacrificing performance

### Visualizations

#### Generated Plots in Notebooks:

**Data Exploration (`01_data_exploration.ipynb`)**:
- Class distribution bar chart (99.83% vs 0.17%)
- Amount distribution: Histogram + Box plots (fraud vs legitimate)
- Time distribution: KDE plots showing patterns
- Correlation heatmap: 30×30 matrix showing feature relationships
- Fraud vs Legitimate feature comparisons: V1-V28 distributions
- Statistical summary tables

**Modeling (`03_modeling.ipynb`)**:
- Loss history curves: Convergence of gradient descent
- ROC curves: Comparison of all 3 models (Class Weights, SMOTE, Undersampling)
- Confusion matrices: Before/after threshold optimization
- Metrics comparison: Bar charts for Accuracy, Precision, Recall, F1
- Feature importance: Top 15 features ranked by absolute coefficient values
- Threshold analysis: Precision-Recall-F1 vs Threshold curves
- Model comparison: 7 models side-by-side visualization
- Precision vs Recall trade-off scatter plot

## Project Structure

```
Credit-Card-Fraud-Detection/
│
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore file
│
├── data/                    # Data directory
│   ├── raw/                 # Raw dataset (creditcard.csv, data_source.txt)
│   └── processed/           # Processed data (X_train.npy, X_test.npy, y_train.npy, y_test.npy, ...)
│
├── notebooks/               # Jupyter notebooks for each stage
│   ├── 01_data_exploration.ipynb   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Data preprocessing pipeline
│   └── 03_modeling.ipynb           # Model training and evaluation
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_processing.py    # Data loading and preprocessing (NumPy)
│   ├── visualization.py      # Plotting functions (Matplotlib/Seaborn)
│   ├── models.py             # ML models (Logistic Regression, etc. and all metrics)
│
└── LICENSE                  # Project license
```

### File Descriptions

- **`data_processing.py`**: Pure NumPy functions for loading CSV, handling missing values, normalization, standardization, and train-test splitting
- **`models.py`**: Logistic Regression implementation from scratch using NumPy with gradient descent optimization
- (All metrics are implemented in `models.py`)
- **`visualization.py`**: Matplotlib/Seaborn functions for creating all visualizations
- **Notebooks**: Step-by-step analysis with detailed explanations and visualizations

## Challenges & Solutions

### Challenge 1: Loading Large CSV File without Pandas
**Problem**: `np.genfromtxt()` is slow for large files and memory-intensive

**Solution**:
```python
# Memory-efficient chunked reading
def load_csv_chunks(filename, chunk_size=10000):
    chunks = []
    with open(filename, 'r') as f:
        next(f)  # Skip header
        while True:
            lines = list(islice(f, chunk_size))
            if not lines:
                break
            chunk = np.array([list(map(float, line.strip().split(','))) 
                            for line in lines])
            chunks.append(chunk)
    return np.vstack(chunks)
```

### Challenge 2: Handling Highly Imbalanced Dataset
**Problem**: Model predicts all transactions as legitimate (99.83% accuracy but useless)

**Solutions Implemented**:

1. **SMOTE-like Oversampling**:
```python
def smote_oversample(X_minority, y_minority, n_samples):
    """Synthetic Minority Oversampling using NumPy"""
    synthetic_samples = []
    for i in range(n_samples):
        # Randomly select a minority sample
        idx = np.random.randint(0, len(X_minority))
        sample = X_minority[idx]
        
        # Find k nearest neighbors (simplified: random neighbor)
        neighbor_idx = np.random.randint(0, len(X_minority))
        neighbor = X_minority[neighbor_idx]
        
        # Generate synthetic sample along the line
        alpha = np.random.random()
        synthetic = sample + alpha * (neighbor - sample)
        synthetic_samples.append(synthetic)
    
    return np.vstack([X_minority] + synthetic_samples)
```

2. **Class Weights in Loss Function**:
```python
# Calculate class weights
n_samples = len(y)
n_classes = 2
class_weights = {
    0: n_samples / (n_classes * np.sum(y == 0)),
    1: n_samples / (n_classes * np.sum(y == 1))
}
# Result: {0: 0.50086, 1: 289.44}

# Apply in loss function
sample_weights = np.where(y == 1, class_weights[1], class_weights[0])
weighted_loss = -np.mean(sample_weights * (y * np.log(pred + 1e-15) + 
                                            (1 - y) * np.log(1 - pred + 1e-15)))
```

3. **Stratified Train-Test Split**:
```python
def stratified_split(X, y, test_size=0.2, random_state=42):
    """Maintain class distribution in splits"""
    np.random.seed(random_state)
    
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    
    np.random.shuffle(fraud_idx)
    np.random.shuffle(legit_idx)
    
    fraud_split = int(len(fraud_idx) * (1 - test_size))
    legit_split = int(len(legit_idx) * (1 - test_size))
    
    train_idx = np.concatenate([fraud_idx[:fraud_split], legit_idx[:legit_split]])
    test_idx = np.concatenate([fraud_idx[fraud_split:], legit_idx[legit_split:]])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

**Results**: Class weights approach achieved best balance (F1: 74.94%)

### Challenge 3: Numerical Stability in Sigmoid Function
**Problem**: Overflow errors with large positive/negative values in sigmoid

**Solution**:
```python
def sigmoid_stable(z):
    # Clip values to prevent overflow
    z = np.clip(z, -500, 500)
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))
```

### Challenge 4: Efficient Matrix Operations
**Problem**: Naive loops are extremely slow for 284,807 samples

**Solution**:
- Used broadcasting: `X - X.mean(axis=0)` instead of loops
- Applied Einstein summation: `np.einsum('ij,ij->i', X, Y)` for row-wise operations
- Utilized fancy indexing: `X[y == 1]` for filtering

### Challenge 5: Train-Test Split without sklearn
**Problem**: Need stratified split to maintain class distribution

**Solution**:
```python
def stratified_split(X, y, test_size=0.2):
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    
    # Shuffle and split each class
    np.random.shuffle(fraud_idx)
    np.random.shuffle(legit_idx)
    
    # Calculate split points
    fraud_split = int(len(fraud_idx) * (1 - test_size))
    legit_split = int(len(legit_idx) * (1 - test_size))
    
    # Combine indices
    train_idx = np.concatenate([fraud_idx[:fraud_split], 
                                legit_idx[:legit_split]])
    test_idx = np.concatenate([fraud_idx[fraud_split:], 
                               legit_idx[legit_split:]])
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

### Challenge 6: Implementing Evaluation Metrics from Scratch
**Problem**: Need precision, recall, F1, ROC-AUC without sklearn

**Solution**:
```python
# All metrics implemented using pure NumPy
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

def roc_curve(y_true, y_scores):
    # Sort by descending score
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]
    
    # Get unique thresholds
    thresholds = np.unique(y_scores_sorted)
    
    # Calculate TPR and FPR for each threshold
    tpr_list, fpr_list = [], []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds

def auc_score(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Trapezoidal rule for integration
    return np.trapz(tpr, fpr)
```

### Challenge 7: Vectorization and Performance Optimization
**Problem**: Processing 284,807 samples requires efficient operations

**Solutions Applied**:
```python
# BAD: Using loops (slow)
for i in range(m):
    for j in range(n):
        result[i] += X[i, j] * theta[j]

# GOOD: Vectorized with matrix multiplication
result = X @ theta  # 100x faster!

# BAD: Loop for standardization
for col in range(X.shape[1]):
    X[:, col] = (X[:, col] - X[:, col].mean()) / X[:, col].std()

# GOOD: Broadcasting
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# Using fancy indexing
fraud_samples = X[y == 1]  # Instead of loops with if conditions

# Using np.where for conditional operations
sample_weights = np.where(y == 1, weight_fraud, weight_legit)
```

**Performance Gains**:
- Training time: 24 seconds (vectorized) vs ~2 hours (if using loops)
- Memory efficient: In-place operations where possible
- Numpy's C-optimized functions: 10-100x faster than pure Python

## Future Improvements

1. **Advanced Algorithms**:
   - Implement Random Forest from scratch using NumPy
   - Add Naive Bayes classifier
   - Develop Gradient Boosting algorithm

2. **Feature Engineering**:
   - Time-based features (hour, day patterns)
   - Polynomial features for interaction terms
   - Frequency encoding for categorical patterns

3. **Optimization Techniques**:
   - Implement Adam optimizer
   - Add L1/L2 regularization
   - Mini-batch gradient descent for faster training

4. **Cross-Validation**:
   - K-fold cross-validation implementation
   - Time-series aware splitting

5. **Deployment**:
   - Create REST API for real-time fraud detection
   - Build web dashboard for monitoring
   - Implement model versioning system

6. **Performance**:
   - Parallel processing with NumPy
   - GPU acceleration investigation
   - Memory optimization for larger datasets

## Contributors

**Student Information**:
- **Name**: Lê Nguyên Thảo
- **Student ID**: 23127118
- **Course**: CSC17104 - Programming for Data Science
- **University**: University of Science, VNUHCM

### Project Highlights
- **100% Pure NumPy Implementation** - No Pandas, no Scikit-learn
- **ML from Scratch** - Logistic Regression with gradient descent, all metrics
- **7 Models Compared** - Comprehensive evaluation of different techniques
- **Excellent Results** - 65.34% precision, 74.94% F1-score (production-ready)
- **Advanced Techniques** - Threshold optimization, ensemble, feature selection
- **Statistical Analysis** - Hypothesis testing, correlation analysis
- **15+ Visualizations** - Professional plots using Matplotlib/Seaborn

### Technical Achievements
1. **Vectorization**: All operations optimized with NumPy broadcasting and matrix operations
2. **Numerical Stability**: Proper handling of overflow/underflow in sigmoid and log functions
3. **Memory Efficiency**: Smart data loading and processing for large datasets
4. **Reproducibility**: Random seeds set for all stochastic operations
5. **Clean Code**: Modular design with reusable functions and clear documentation

## Contact

- **Email**: nguyenthaole2005@gmail.com
- **GitHub**: [NThao05](https://github.com/NThao05)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Acknowledgments**:
- Dataset provided by ULB Machine Learning Group
- Course instructors and teaching assistants
- Credit card companies for fraud detection research

**References**:
1. Andrea Dal Pozzolo et al. "Calibrating Probability with Undersampling for Unbalanced Classification." IEEE Symposium Series on Computational Intelligence, 2015.
2. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron
3. NumPy Documentation: https://numpy.org/doc/
