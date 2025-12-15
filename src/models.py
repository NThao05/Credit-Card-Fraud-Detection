"""
Machine Learning Models and Evaluation Metrics - Pure NumPy Implementation
Includes ML algorithms, loss functions, and all evaluation metrics

VECTORIZATION & EFFICIENCY:
- Gradient descent uses matrix operations (X.T @ error)
- Confusion matrix: vectorized with np.bincount
- All metrics: vectorized boolean operations
- Mathematical stability: np.clip, epsilon for log(0), stable sigmoid
- Broadcasting for efficient element-wise operations
"""

import numpy as np


# ============================================================================
# EVALUATION METRICS - All implemented from scratch using NumPy
# ============================================================================

def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix using vectorized operations
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
        
    Returns:
    --------
    cm : ndarray
        Confusion matrix [[TN, FP], [FN, TP]]
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Vectorized confusion matrix computation
    # Use np.bincount for efficient counting
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Create linear indices: row * n_classes + col
    indices = y_true * n_classes + y_pred
    cm = np.bincount(indices, minlength=n_classes**2).reshape(n_classes, n_classes)
    
    return cm


def accuracy_score(y_true, y_pred):
    """
    Compute accuracy: (TP + TN) / (TP + TN + FP + FN)
    """
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, pos_label=1):
    """
    Compute precision: TP / (TP + FP)
    """
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred, pos_label=1):
    """
    Compute recall (sensitivity): TP / (TP + FN)
    """
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, pos_label=1):
    """
    Compute F1 score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision = precision_score(y_true, y_pred, pos_label)
    recall = recall_score(y_true, y_pred, pos_label)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def specificity_score(y_true, y_pred, pos_label=1):
    """
    Compute specificity: TN / (TN + FP)
    """
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def roc_curve(y_true, y_scores):
    """
    Compute ROC curve
    
    Returns:
    --------
    fpr, tpr, thresholds : tuple of ndarrays
    """
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    thresholds = np.unique(y_scores)
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])
    
    tpr_list, fpr_list = [], []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc_score(y_true, y_scores):
    """
    Compute Area Under ROC Curve using trapezoidal rule
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    return np.trapz(tpr, fpr)


def precision_recall_curve(y_true, y_scores):
    """
    Compute precision-recall curve
    
    Returns:
    --------
    precision, recall, thresholds : tuple of ndarrays
    """
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    thresholds = np.unique(y_scores)
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])
    
    precision_list, recall_list = [], []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    return np.array(precision_list), np.array(recall_list), thresholds


def average_precision_score(y_true, y_scores):
    """
    Compute average precision score (area under PR curve)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]
    return np.trapz(precision, recall)


def classification_report(y_true, y_pred, y_scores=None):
    """
    Generate classification report with all metrics
    
    Returns:
    --------
    report : dict
        Dictionary with all metrics
    """
    report = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'specificity': specificity_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_scores is not None:
        report['auc'] = auc_score(y_true, y_scores)
    
    return report


def print_classification_report(y_true, y_pred, y_scores=None):
    """
    Print formatted classification report
    """
    report = classification_report(y_true, y_pred, y_scores)
    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(f"Accuracy:    {report['accuracy']:.4f}")
    print(f"Precision:   {report['precision']:.4f}")
    print(f"Recall:      {report['recall']:.4f}")
    print(f"F1-Score:    {report['f1_score']:.4f}")
    print(f"Specificity: {report['specificity']:.4f}")
    if 'auc' in report:
        print(f"AUC-ROC:     {report['auc']:.4f}")
    print("\nConfusion Matrix:")
    cm = report['confusion_matrix']
    # Print confusion matrix as a clean 2x2 table
    for row in cm:
        print("[" + "  ".join(f"{int(x):5d}" for x in row) + "]")
    print("=" * 50)


def compute_metrics(y_true, y_pred, y_scores=None):
    """
    Compute all metrics at once (wrapper for classification_report)
    """
    return classification_report(y_true, y_pred, y_scores)


def cross_validation_score(model, X, y, cv=5, random_seed=42):
    """
    Perform k-fold cross-validation
    
    Parameters:
    -----------
    model : object
        Model with fit and predict methods
    X : ndarray
        Features
    y : ndarray
        Labels
    cv : int
        Number of folds
    random_seed : int
        Random seed
        
    Returns:
    --------
    scores : ndarray
        Accuracy scores for each fold
    """
    np.random.seed(random_seed)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    fold_size = len(X) // cv
    scores = []
    
    for i in range(cv):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < cv - 1 else len(X)
        
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_test_fold = X[test_idx]
        y_test_fold = y[test_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        
        score = accuracy_score(y_test_fold, y_pred)
        scores.append(score)
        
        print(f"Fold {i+1}/{cv}: Accuracy = {score:.4f}")
    
    scores = np.array(scores)
    print(f"\nMean CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return scores


# ============================================================================
# MACHINE LEARNING MODELS - All implemented from scratch using NumPy
# ============================================================================


class LogisticRegressionNumPy:
    """
    Logistic Regression implemented from scratch using NumPy
    
    Uses gradient descent optimization with binary cross-entropy loss
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=None, 
                 lambda_reg=0.01, class_weights=None, verbose=True):
        """
        Initialize Logistic Regression model
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        iterations : int
            Number of training iterations
        regularization : str or None
            'l1', 'l2', or None
        lambda_reg : float
            Regularization strength
        class_weights : dict
            Dictionary mapping class to weight
        verbose : bool
            Print training progress
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.class_weights = class_weights
        self.verbose = verbose
        
        self.theta = None
        self.bias = None
        self.loss_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid activation function with numerical stability
        
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Parameters:
        -----------
        z : ndarray
            Input values
            
        Returns:
        --------
        activation : ndarray
            Sigmoid activation
        """
        # Clip values to prevent overflow
        z = np.clip(z, -500, 500)
        
        # Numerically stable sigmoid
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    def compute_loss(self, y_true, y_pred, theta):
        """
        Compute binary cross-entropy loss with optional regularization
        
        Formula: J(θ) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
        
        Parameters:
        -----------
        y_true : ndarray
            True labels
        y_pred : ndarray
            Predicted probabilities
        theta : ndarray
            Model parameters
            
        Returns:
        --------
        loss : float
            Loss value
        """
        m = len(y_true)
        
        # Prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy
        if self.class_weights is not None:
            # Apply class weights
            sample_weights = np.array([self.class_weights[int(y)] for y in y_true])
            loss = -np.mean(sample_weights * (
                y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            ))
        else:
            loss = -np.mean(
                y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            )
        
        # Add regularization
        if self.regularization == 'l2':
            loss += (self.lambda_reg / (2 * m)) * np.sum(theta ** 2)
        elif self.regularization == 'l1':
            loss += (self.lambda_reg / m) * np.sum(np.abs(theta))
        
        return loss
    
    def compute_gradients(self, X, y_true, y_pred):
        """
        Compute gradients for gradient descent
        
        Formula: ∂J/∂θ = 1/m * X^T * (h(x) - y)
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
        y_true : ndarray
            True labels
        y_pred : ndarray
            Predicted probabilities
            
        Returns:
        --------
        grad_theta, grad_bias : tuple
            Gradients for theta and bias
        """
        m = len(y_true)
        
        # Compute error
        error = y_pred - y_true
        
        # Apply class weights if provided
        if self.class_weights is not None:
            sample_weights = np.array([self.class_weights[int(y)] for y in y_true])
            error = error * sample_weights
        
        # Compute gradients using vectorized operations
        grad_theta = (1 / m) * (X.T @ error)
        grad_bias = (1 / m) * np.sum(error)
        
        # Add regularization gradient
        if self.regularization == 'l2':
            grad_theta += (self.lambda_reg / m) * self.theta
        elif self.regularization == 'l1':
            grad_theta += (self.lambda_reg / m) * np.sign(self.theta)
        
        return grad_theta, grad_bias
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        
        Parameters:
        -----------
        X : ndarray
            Training features (m x n)
        y : ndarray
            Training labels (m,)
        """
        m, n = X.shape
        
        # Initialize parameters
        self.theta = np.zeros(n)
        self.bias = 0
        self.loss_history = []
        
        # Gradient descent
        for iteration in range(self.iterations):
            # Forward pass
            z = X @ self.theta + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred, self.theta)
            self.loss_history.append(loss)
            
            # Compute gradients
            grad_theta, grad_bias = self.compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.theta -= self.learning_rate * grad_theta
            self.bias -= self.learning_rate * grad_bias
            
            # Print progress every 100 iterations and at the end
            if self.verbose:
                if iteration % 100 == 0 or iteration == self.iterations - 1:
                    print(f"Iteration {iteration}/{self.iterations}, Loss: {loss:.6f}", flush=True)
        
        if self.verbose:
            print(f"\nTraining completed!", flush=True)
            print(f"Final loss: {self.loss_history[-1]:.6f}", flush=True)
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
            
        Returns:
        --------
        probabilities : ndarray
            Predicted probabilities
        """
        z = X @ self.theta + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
        threshold : float
            Classification threshold
            
        Returns:
        --------
        predictions : ndarray
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X, y, threshold=0.5):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
        y : ndarray
            True labels
        threshold : float
            Classification threshold
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X, threshold)
        y_pred_proba = self.predict_proba(X)
        
        metrics = compute_metrics(y, y_pred, y_pred_proba)
        
        return metrics


class LinearRegressionNumPy:
    """
    Linear Regression implemented from scratch using NumPy
    
    For comparison or regression tasks
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=None, lambda_reg=0.01, verbose=True):
        """
        Initialize Linear Regression model
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        iterations : int
            Number of training iterations
        regularization : str or None
            'l1', 'l2', or None
        lambda_reg : float
            Regularization strength
        verbose : bool
            Print training progress
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        
        self.theta = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Train using gradient descent
        
        Parameters:
        -----------
        X : ndarray
            Training features
        y : ndarray
            Training target
        """
        m, n = X.shape
        
        # Initialize parameters
        self.theta = np.zeros(n)
        self.bias = 0
        self.loss_history = []
        
        # Gradient descent
        for iteration in range(self.iterations):
            # Predictions
            y_pred = X @ self.theta + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            
            if self.regularization == 'l2':
                loss += (self.lambda_reg / (2 * m)) * np.sum(self.theta ** 2)
            
            self.loss_history.append(loss)
            
            # Compute gradients
            grad_theta = (2 / m) * (X.T @ (y_pred - y))
            grad_bias = (2 / m) * np.sum(y_pred - y)
            
            if self.regularization == 'l2':
                grad_theta += (self.lambda_reg / m) * self.theta
            
            # Update parameters
            self.theta -= self.learning_rate * grad_theta
            self.bias -= self.learning_rate * grad_bias
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """
        Predict target values
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
            
        Returns:
        --------
        predictions : ndarray
            Predicted values
        """
        return X @ self.theta + self.bias


class NaiveBayesNumPy:
    """
    Gaussian Naive Bayes implemented from scratch using NumPy
    """
    
    def __init__(self, verbose=True):
        """
        Initialize Naive Bayes model
        """
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.vars = {}
        self.verbose = verbose
    
    def fit(self, X, y):
        """
        Train Naive Bayes model
        
        Parameters:
        -----------
        X : ndarray
            Training features
        y : ndarray
            Training labels
        """
        self.classes = np.unique(y)
        
        for cls in self.classes:
            X_cls = X[y == cls]
            
            # Calculate prior probability
            self.class_priors[cls] = len(X_cls) / len(X)
            
            # Calculate mean and variance for each feature
            self.means[cls] = np.mean(X_cls, axis=0)
            self.vars[cls] = np.var(X_cls, axis=0) + 1e-6  # Add small value to avoid division by zero
        
        if self.verbose:
            print("Naive Bayes model trained successfully")
            print(f"Classes: {self.classes}")
            print(f"Class priors: {self.class_priors}")
    
    def _gaussian_pdf(self, x, mean, var):
        """
        Gaussian probability density function
        
        Formula: p(x) = 1/√(2πσ²) * exp(-(x-μ)²/(2σ²))
        
        Parameters:
        -----------
        x : ndarray
            Input value
        mean : float
            Mean
        var : float
            Variance
            
        Returns:
        --------
        probability : float
            PDF value
        """
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
            
        Returns:
        --------
        probabilities : ndarray
            Class probabilities
        """
        posteriors = []
        
        for cls in self.classes:
            prior = np.log(self.class_priors[cls])
            
            # Calculate conditional probability for each feature
            conditional = np.sum(np.log(self._gaussian_pdf(X, self.means[cls], self.vars[cls])), axis=1)
            
            posterior = prior + conditional
            posteriors.append(posterior)
        
        posteriors = np.array(posteriors).T
        
        # Convert log probabilities to probabilities
        posteriors_exp = np.exp(posteriors - np.max(posteriors, axis=1, keepdims=True))
        probabilities = posteriors_exp / np.sum(posteriors_exp, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
            
        Returns:
        --------
        predictions : ndarray
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]


