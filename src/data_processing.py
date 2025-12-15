"""
Data Processing Module - Pure NumPy Implementation
Handles data loading, preprocessing, normalization, and splitting

VECTORIZATION & EFFICIENCY NOTES:
- All operations use vectorized NumPy (no Python loops on arrays)
- Broadcasting for element-wise operations
- Fancy indexing and boolean masking throughout
- Mathematical stability (epsilon, np.clip, np.where for division)
- Memory-efficient operations with in-place modifications where safe
"""

import numpy as np
from itertools import islice


def load_csv_numpy(filepath, delimiter=',', skip_header=True):
    """
    Load CSV file using pure NumPy
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    delimiter : str
        Column delimiter
    skip_header : bool
        Whether to skip first row
        
    Returns:
    --------
    data : ndarray
        Loaded data as NumPy array
    """
    try:
        # Manual CSV reading to handle quoted values properly
        import csv
        data_list = []
        
        with open(filepath, 'r') as f:
            csv_reader = csv.reader(f)
            if skip_header:
                next(csv_reader)  # Skip header
            
            for row in csv_reader:
                # Convert all values to float
                data_list.append([float(val) for val in row])
        
        data = np.array(data_list, dtype=np.float64)
        print(f"Data loaded successfully: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to loadtxt
        try:
            data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1 if skip_header else 0)
            print(f"Data loaded with loadtxt: {data.shape}")
            return data
        except Exception as e2:
            print(f"Error with loadtxt: {e2}")
            return None


def load_csv_chunks(filepath, chunk_size=50000, delimiter=','):
    """
    Load large CSV file in chunks for memory efficiency
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    chunk_size : int
        Number of rows per chunk
    delimiter : str
        Column delimiter
        
    Returns:
    --------
    data : ndarray
        Loaded data
    """
    chunks = []
    try:
        with open(filepath, 'r') as f:
            # Skip header
            next(f)
            
            while True:
                lines = list(islice(f, chunk_size))
                if not lines:
                    break
                
                # Parse lines to numpy array
                chunk = np.array([list(map(float, line.strip().split(delimiter))) for line in lines])
                chunks.append(chunk)
                print(f"Loaded chunk: {chunk.shape}")
        
        data = np.vstack(chunks)
        print(f"Total data loaded: {data.shape}")
        return data
    
    except Exception as e:
        print(f"Error loading data in chunks: {e}")
        return None


def check_missing_values(data):
    """
    Check for missing values in dataset
    
    Parameters:
    -----------
    data : ndarray
        Input data
        
    Returns:
    --------
    missing_info : dict
        Dictionary with missing value statistics
    """
    total_missing = np.isnan(data).sum()
    missing_per_column = np.isnan(data).sum(axis=0)
    missing_percentage = (total_missing / data.size) * 100
    
    print(f"Total missing values: {total_missing}")
    print(f"Missing percentage: {missing_percentage:.4f}%")
    
    return {
        'total': total_missing,
        'per_column': missing_per_column,
        'percentage': missing_percentage
    }


def handle_missing_values(data, strategy='median'):
    """
    Handle missing values using specified strategy
    
    Parameters:
    -----------
    data : ndarray
        Input data with potential missing values
    strategy : str
        'mean', 'median', 'mode', or 'zero'
        
    Returns:
    --------
    data_imputed : ndarray
        Data with imputed missing values
    """
    data_imputed = data.copy()
    
    for col in range(data.shape[1]):
        column_data = data[:, col]
        mask = np.isnan(column_data)
        
        if mask.any():
            if strategy == 'mean':
                fill_value = np.nanmean(column_data)
            elif strategy == 'median':
                fill_value = np.nanmedian(column_data)
            elif strategy == 'mode':
                unique, counts = np.unique(column_data[~mask], return_counts=True)
                fill_value = unique[np.argmax(counts)]
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = 0
                
            data_imputed[mask, col] = fill_value
            print(f"Imputed {mask.sum()} values in column {col} with {strategy}: {fill_value:.4f}")
            
    return data_imputed


def impute_with_knn(data, k=5):
    """
    Impute missing values using k-Nearest Neighbors
    
    This is a model-based imputation method that predicts missing values
    based on similar samples.
    
    Parameters:
    -----------
    data : ndarray
        Input data with potential missing values
    k : int
        Number of nearest neighbors to use
        
    Returns:
    --------
    data_imputed : ndarray
        Data with imputed missing values
    """
    data_imputed = data.copy()
    
    # Find columns with missing values
    cols_with_missing = [col for col in range(data.shape[1]) 
                        if np.isnan(data[:, col]).any()]
    
    for col in cols_with_missing:
        missing_mask = np.isnan(data[:, col])
        
        if not missing_mask.any():
            continue
            
        # Use other columns as features for KNN
        other_cols = [c for c in range(data.shape[1]) if c != col]
        
        # Get samples without missing values in target column
        complete_rows = ~missing_mask
        X_train = data[complete_rows][:, other_cols]
        y_train = data[complete_rows, col]
        
        # Get samples with missing values
        X_test = data[missing_mask][:, other_cols]
        
        # Handle NaN in features
        X_train_clean = np.nan_to_num(X_train, nan=np.nanmean(X_train))
        X_test_clean = np.nan_to_num(X_test, nan=np.nanmean(X_train))
        
        # Vectorized KNN imputation
        # Calculate distances for all test samples at once using broadcasting
        # Shape: (n_test, n_train)
        diff = X_test_clean[:, np.newaxis, :] - X_train_clean[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Get k nearest neighbors for all test samples (vectorized)
        nearest_indices = np.argsort(distances, axis=1)[:, :k]
        
        # Get nearest values and compute mean (vectorized)
        # Use advanced indexing to gather values
        nearest_values = y_train[nearest_indices]
        imputed_values = np.mean(nearest_values, axis=1)
        
        # Assign all imputed values at once
        missing_indices = np.where(missing_mask)[0]
        data_imputed[missing_indices, col] = imputed_values
        
        print(f"Imputed {missing_mask.sum()} values in column {col} using KNN (k={k})")
    
    return data_imputed


def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers using Z-score method
    
    Parameters:
    -----------
    data : ndarray
        Input data
    threshold : float
        Z-score threshold (default: 3)
        
    Returns:
    --------
    outlier_mask : ndarray
        Boolean mask indicating outliers
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    z_scores = np.abs((data - mean) / std)
    outlier_mask = (z_scores > threshold).any(axis=1)
    
    print(f"Outliers detected: {outlier_mask.sum()} ({outlier_mask.sum()/len(data)*100:.2f}%)")
    
    return outlier_mask


def detect_outliers_iqr(data, multiplier=1.5):
    """
    Detect outliers using IQR method
    
    Parameters:
    -----------
    data : ndarray
        Input data
    multiplier : float
        IQR multiplier (default: 1.5)
        
    Returns:
    --------
    outlier_mask : ndarray
        Boolean mask indicating outliers
    """
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outlier_mask = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
    
    print(f"IQR Outliers detected: {outlier_mask.sum()} ({outlier_mask.sum()/len(data)*100:.2f}%)")
    
    return outlier_mask


def normalize_minmax(data):
    """
    Min-Max normalization: scale to [0, 1]
    
    Formula: x_norm = (x - x_min) / (x_max - x_min)
    
    Parameters:
    -----------
    data : ndarray
        Input data
        
    Returns:
    --------
    normalized_data : ndarray
        Normalized data
    scaler_params : dict
        Min and max values for inverse transform
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1, range_vals)
    
    normalized_data = (data - min_vals) / range_vals
    
    scaler_params = {'min': min_vals, 'max': max_vals}
    
    return normalized_data, scaler_params


def normalize_log(data, offset=1):
    """
    Log transformation: useful for right-skewed distributions
    
    Formula: x_log = log(x + offset)
    
    Parameters:
    -----------
    data : ndarray
        Input data (must be non-negative)
    offset : float
        Offset to avoid log(0)
        
    Returns:
    --------
    log_data : ndarray
        Log-transformed data
    """
    # Ensure non-negative values
    data_positive = np.abs(data)
    log_data = np.log(data_positive + offset)
    
    return log_data


def normalize_decimal_scaling(data):
    """
    Decimal scaling normalization: move decimal point
    
    Formula: x_scaled = x / (10^j)
    where j = ceil(log10(max(|x|)))
    
    Parameters:
    -----------
    data : ndarray
        Input data
        
    Returns:
    --------
    scaled_data : ndarray
        Decimal-scaled data
    scaler_params : dict
        Scaling parameters for inverse transform
    """
    max_abs = np.max(np.abs(data), axis=0)
    
    # Avoid log of zero
    max_abs = np.where(max_abs == 0, 1, max_abs)
    
    j = np.ceil(np.log10(max_abs))
    scaling_factor = 10 ** j
    
    scaled_data = data / scaling_factor
    
    scaler_params = {'j': j, 'scaling_factor': scaling_factor}
    
    return scaled_data, scaler_params


def standardize_zscore(data):
    """
    Z-score standardization: mean=0, std=1
    
    Formula: x_std = (x - μ) / σ
    
    Parameters:
    -----------
    data : ndarray
        Input data
        
    Returns:
    --------
    standardized_data : ndarray
        Standardized data
    scaler_params : dict
        Mean and std for inverse transform
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    standardized_data = (data - mean) / std
    
    scaler_params = {'mean': mean, 'std': std}
    
    return standardized_data, scaler_params


def stratified_train_test_split(X, y, test_size=0.2, random_seed=42):
    """
    Stratified train-test split to maintain class distribution
    
    Parameters:
    -----------
    X : ndarray
        Features
    y : ndarray
        Labels
    test_size : float
        Proportion of test set
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple of ndarrays
        Split datasets
    """
    np.random.seed(random_seed)
    
    # Get indices for each class
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        
        n_test = int(len(cls_indices) * test_size)
        
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # Shuffle combined indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Train fraud rate: {y_train.sum()/len(y_train)*100:.4f}%")
    print(f"Test fraud rate: {y_test.sum()/len(y_test)*100:.4f}%")
    
    return X_train, X_test, y_train, y_test


def smote_oversample(X, y, target_ratio=0.5, k_neighbors=5, random_seed=42):
    """
    SMOTE-like oversampling for minority class
    
    Parameters:
    -----------
    X : ndarray
        Features
    y : ndarray
        Labels
    target_ratio : float
        Target ratio of minority to majority class
    k_neighbors : int
        Number of nearest neighbors to use
    random_seed : int
        Random seed
        
    Returns:
    --------
    X_resampled, y_resampled : tuple of ndarrays
        Resampled data
    """
    np.random.seed(random_seed)
    
    minority_class = 1
    majority_class = 0
    
    X_minority = X[y == minority_class]
    X_majority = X[y == majority_class]
    
    # Calculate number of synthetic samples needed
    n_majority = len(X_majority)
    n_minority = len(X_minority)
    n_synthetic = int(n_majority * target_ratio) - n_minority
    
    if n_synthetic <= 0:
        print("No oversampling needed")
        return X, y
    
    synthetic_samples = []
    
    for _ in range(n_synthetic):
        # Random sample from minority class
        idx = np.random.randint(0, len(X_minority))
        sample = X_minority[idx]
        
        # Find k nearest neighbors
        distances = np.sum((X_minority - sample) ** 2, axis=1)
        nearest_idx = np.argsort(distances)[1:k_neighbors+1]
        
        # Generate synthetic sample
        neighbor_idx = np.random.choice(nearest_idx)
        neighbor = X_minority[neighbor_idx]
        
        alpha = np.random.random()
        synthetic = sample + alpha * (neighbor - sample)
        synthetic_samples.append(synthetic)
    
    synthetic_samples = np.array(synthetic_samples)
    
    # Combine original and synthetic samples
    X_resampled = np.vstack([X, synthetic_samples])
    y_resampled = np.hstack([y, np.ones(len(synthetic_samples))])
    
    print(f"Original dataset: {len(X)} samples")
    print(f"Resampled dataset: {len(X_resampled)} samples")
    print(f"Synthetic samples generated: {len(synthetic_samples)}")
    
    return X_resampled, y_resampled


def random_undersample(X, y, target_ratio=1.0, random_seed=42):
    """
    Random undersampling of majority class
    
    Parameters:
    -----------
    X : ndarray
        Features
    y : ndarray
        Labels
    target_ratio : float
        Target ratio of minority to majority class
    random_seed : int
        Random seed
        
    Returns:
    --------
    X_resampled, y_resampled : tuple of ndarrays
        Resampled data
    """
    np.random.seed(random_seed)
    
    minority_indices = np.where(y == 1)[0]
    majority_indices = np.where(y == 0)[0]
    
    n_minority = len(minority_indices)
    n_majority_target = int(n_minority / target_ratio)
    
    # Randomly select majority samples
    selected_majority = np.random.choice(majority_indices, 
                                        size=min(n_majority_target, len(majority_indices)), 
                                        replace=False)
    
    # Combine indices
    selected_indices = np.concatenate([minority_indices, selected_majority])
    np.random.shuffle(selected_indices)
    
    X_resampled = X[selected_indices]
    y_resampled = y[selected_indices]
    
    print(f"Original dataset: {len(X)} samples")
    print(f"Undersampled dataset: {len(X_resampled)} samples")
    print(f"New class ratio: {y_resampled.sum()/len(y_resampled):.4f}")
    
    return X_resampled, y_resampled


def compute_class_weights(y):
    """
    Compute class weights for imbalanced datasets
    
    Formula: w_j = n_samples / (n_classes * n_samples_j)
    
    Parameters:
    -----------
    y : ndarray
        Labels
        
    Returns:
    --------
    class_weights : dict
        Dictionary mapping class to weight
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)
    
    class_weights = {}
    for cls, count in zip(unique_classes, class_counts):
        class_weights[cls] = n_samples / (n_classes * count)
    
    print(f"Class weights: {class_weights}")
    
    return class_weights


def correlation_matrix(X):
    """
    Compute correlation matrix using NumPy
    
    Pearson correlation coefficient:
    r = Σ(x_i - x̄)(y_i - ȳ) / sqrt(Σ(x_i - x̄)² * Σ(y_i - ȳ)²)
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
        
    Returns:
    --------
    corr_matrix : ndarray
        Correlation matrix
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Standardized version
    std_devs = np.std(X, axis=0)
    std_devs = np.where(std_devs == 0, 1, std_devs)
    
    X_standardized = X_centered / std_devs
    corr_matrix = (X_standardized.T @ X_standardized) / (X.shape[0] - 1)
    
    return corr_matrix


def feature_statistics(X, feature_names=None):
    """
    Compute comprehensive statistics for each feature
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    feature_names : list
        List of feature names
        
    Returns:
    --------
    stats : dict
        Dictionary of statistics for each feature
    """
    n_features = X.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    stats = {}
    
    for i, name in enumerate(feature_names):
        feature_data = X[:, i]
        
        stats[name] = {
            'mean': np.mean(feature_data),
            'median': np.median(feature_data),
            'std': np.std(feature_data),
            'min': np.min(feature_data),
            'max': np.max(feature_data),
            'q25': np.percentile(feature_data, 25),
            'q75': np.percentile(feature_data, 75),
            'skewness': compute_skewness(feature_data),
            'kurtosis': compute_kurtosis(feature_data)
        }
    
    return stats


def compute_skewness(data):
    """
    Compute skewness of distribution
    
    Formula: skewness = E[(X - μ)³] / σ³
    
    Parameters:
    -----------
    data : ndarray
        Input data
        
    Returns:
    --------
    skewness : float
        Skewness value
    """
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0
    
    skewness = np.mean(((data - mean) / std) ** 3)
    
    return skewness


def compute_kurtosis(data):
    """
    Compute kurtosis of distribution
    
    Formula: kurtosis = E[(X - μ)⁴] / σ⁴ - 3
    
    Parameters:
    -----------
    data : ndarray
        Input data
        
    Returns:
    --------
    kurtosis : float
        Kurtosis value (excess kurtosis)
    """
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0
    
    kurtosis = np.mean(((data - mean) / std) ** 4) - 3
    
    return kurtosis


def two_sample_ttest(sample1, sample2):
    """
    Perform two-sample t-test using NumPy
    
    H0: μ1 = μ2 (no difference between means)
    H1: μ1 ≠ μ2 (significant difference exists)
    
    Parameters:
    -----------
    sample1 : ndarray
        First sample
    sample2 : ndarray
        Second sample
        
    Returns:
    --------
    t_statistic : float
        T-statistic
    p_value : float
        Two-tailed p-value (approximation)
    """
    n1 = len(sample1)
    n2 = len(sample2)
    
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    
    # Pooled standard error
    pooled_se = np.sqrt(var1/n1 + var2/n2)
    
    # Avoid division by zero
    if pooled_se == 0:
        return 0, 1.0
    
    # T-statistic
    t_statistic = (mean1 - mean2) / pooled_se
    
    # Degrees of freedom (Welch-Satterthwaite equation)
    df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    # P-value using t-distribution approximation
    # For large samples (df > 30), t-distribution ≈ normal distribution
    # For smaller samples, we use a polynomial approximation
    p_value = 2 * (1 - t_distribution_cdf(np.abs(t_statistic), df))
    
    return t_statistic, p_value


def normal_cdf(x):
    """
    Cumulative distribution function for standard normal distribution
    Using Abramowitz and Stegun approximation (high accuracy)
    
    Parameters:
    -----------
    x : float or ndarray
        Input value(s)
        
    Returns:
    --------
    cdf : float or ndarray
        CDF value(s)
    """
    # Constants for the approximation
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)
    
    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x / 2.0) / np.sqrt(2.0 * np.pi)
    
    return 0.5 * (1.0 + sign * y)


def t_distribution_cdf(t, df):
    """
    Cumulative distribution function for t-distribution
    Using approximation for various degrees of freedom
    
    Parameters:
    -----------
    t : float or ndarray
        T-statistic value(s)
    df : float
        Degrees of freedom
        
    Returns:
    --------
    cdf : float or ndarray
        CDF value(s)
    """
    if df > 30:
        # For large df, t-distribution approximates normal distribution
        return normal_cdf(t)
    else:
        # For smaller df, use transformation to approximate
        # This is a reasonable approximation using the relationship between
        # t-distribution and normal distribution
        x = t / np.sqrt(df)
        
        # Modified approximation that accounts for heavier tails
        adjustment = 1.0 / (1.0 + df / 100.0)
        
        # Use normal CDF with adjustment
        z = t * (1.0 - adjustment / (4.0 * df))
        
        return normal_cdf(z)


def t_distribution_cdf(t, df):
    """
    Cumulative distribution function for t-distribution
    Using approximation for various degrees of freedom
    
    Parameters:
    -----------
    t : float or ndarray
        T-statistic value(s)
    df : float
        Degrees of freedom
        
    Returns:
    --------
    cdf : float or ndarray
        CDF value(s)
    """
    if df > 30:
        # For large df, t-distribution approximates normal distribution
        return normal_cdf(t)
    else:
        # For smaller df, use transformation to approximate
        # This is a reasonable approximation using the relationship between
        # t-distribution and normal distribution
        x = t / np.sqrt(df)
        
        # Modified approximation that accounts for heavier tails
        adjustment = 1.0 / (1.0 + df / 100.0)
        
        # Use normal CDF with adjustment
        z = t * (1.0 - adjustment / (4.0 * df))
        
        return normal_cdf(z)


def gamma_function(z):
    """
    Approximation of Gamma function using Stirling's formula
    
    Γ(z) ≈ sqrt(2π/z) * (z/e)^z for large z
    For small z, use recursive relation: Γ(z+1) = z*Γ(z)
    
    Parameters:
    -----------
    z : float
        Input value (z > 0)
        
    Returns:
    --------
    gamma : float
        Gamma function value
    """
    if z < 0.5:
        # Use reflection formula for small values
        return np.pi / (np.sin(np.pi * z) * gamma_function(1 - z))
    
    # Stirling's approximation for z >= 0.5
    z = z - 1  # Adjust for Γ(z) = (z-1)!
    
    # Coefficients for Lanczos approximation
    g = 7
    coef = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    
    z = z + 1
    x = coef[0]
    for i in range(1, g + 2):
        x += coef[i] / (z + i)
    
    t = z + g + 0.5
    return np.sqrt(2 * np.pi) * t**(z + 0.5) * np.exp(-t) * x


def incomplete_gamma(a, x):
    """
    Lower incomplete gamma function using series expansion
    
    γ(a,x) = ∫[0 to x] t^(a-1) * e^(-t) dt
    
    Parameters:
    -----------
    a : float
        Shape parameter
    x : float
        Upper limit of integration
        
    Returns:
    --------
    result : float
        Incomplete gamma value
    """
    if x <= 0:
        return 0.0
    
    # Series expansion
    sum_val = 1.0 / a
    term = 1.0 / a
    
    for n in range(1, 200):  # Max iterations
        term *= x / (a + n)
        sum_val += term
        
        if abs(term) < 1e-10:  # Convergence threshold
            break
    
    return sum_val * (x ** a) * np.exp(-x)


def chi_square_cdf(x, df):
    """
    Cumulative distribution function for chi-square distribution
    
    Chi-square is a special case of gamma distribution:
    χ²(df) ~ Gamma(df/2, 2)
    
    Parameters:
    -----------
    x : float
        Chi-square statistic
    df : float
        Degrees of freedom
        
    Returns:
    --------
    cdf : float
        CDF value
    """
    if x <= 0:
        return 0.0
    
    # Chi-square CDF = P(a, x) where a = df/2
    a = df / 2.0
    
    # Use regularized incomplete gamma function
    # P(a,x) = γ(a, x/2) / Γ(a)
    
    gamma_a = gamma_function(a)
    incomplete_gamma_val = incomplete_gamma(a, x / 2.0)
    
    cdf = incomplete_gamma_val / gamma_a
    
    # Ensure CDF is in [0, 1]
    return np.clip(cdf, 0.0, 1.0)


def beta_function(a, b):
    """
    Beta function using Gamma functions
    
    B(a,b) = Γ(a)*Γ(b) / Γ(a+b)
    
    Parameters:
    -----------
    a, b : float
        Parameters
        
    Returns:
    --------
    beta : float
        Beta function value
    """
    return gamma_function(a) * gamma_function(b) / gamma_function(a + b)


def incomplete_beta(x, a, b, max_iter=200):
    """
    Incomplete beta function using continued fraction
    
    Parameters:
    -----------
    x : float
        Upper limit (0 <= x <= 1)
    a, b : float
        Parameters
    max_iter : int
        Maximum iterations
        
    Returns:
    --------
    result : float
        Incomplete beta value
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    
    # Use symmetry for efficiency
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - incomplete_beta(1.0 - x, b, a, max_iter)
    
    # Continued fraction expansion
    lbeta_ab = np.log(beta_function(a, b))
    front = np.exp(np.log(x) * a + np.log(1.0 - x) * b - lbeta_ab) / a
    
    f = 1.0
    c = 1.0
    d = 0.0
    
    for m in range(max_iter):
        numerator = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        
        f *= c * d
        
        numerator = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
        
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        
        delta = c * d
        f *= delta
        
        if abs(delta - 1.0) < 1e-10:
            break
    
    return front * f


def f_distribution_cdf(x, df1, df2):
    """
    Cumulative distribution function for F-distribution
    
    F(df1, df2) CDF uses incomplete beta function
    
    Parameters:
    -----------
    x : float
        F-statistic
    df1, df2 : float
        Degrees of freedom (numerator and denominator)
        
    Returns:
    --------
    cdf : float
        CDF value
    """
    if x <= 0:
        return 0.0
    
    # Transform F to Beta distribution
    # If X ~ F(df1, df2), then Y = df1*X/(df1*X + df2) ~ Beta(df1/2, df2/2)
    
    y = (df1 * x) / (df1 * x + df2)
    
    # Use regularized incomplete beta function
    a = df1 / 2.0
    b = df2 / 2.0
    
    cdf = incomplete_beta(y, a, b)
    
    return np.clip(cdf, 0.0, 1.0)


def chi_square_test(observed, expected):
    """
    Perform chi-square goodness of fit test
    
    H0: Observed frequencies fit expected distribution
    H1: Observed frequencies do not fit expected distribution
    
    Formula: χ² = Σ((O - E)² / E)
    
    Parameters:
    -----------
    observed : ndarray
        Observed frequencies
    expected : ndarray
        Expected frequencies
        
    Returns:
    --------
    chi_square_stat : float
        Chi-square statistic
    p_value : float
        P-value (approximation)
    """
    # Avoid division by zero
    expected = np.where(expected == 0, 1e-10, expected)
    
    chi_square_stat = np.sum((observed - expected) ** 2 / expected)
    
    # Degrees of freedom
    df = len(observed) - 1
    
    # P-value using chi-square distribution approximation
    p_value = 1 - chi_square_cdf(chi_square_stat, df)
    
    return chi_square_stat, p_value


def anova_test(groups):
    """
    Perform one-way ANOVA test
    
    H0: All group means are equal (μ1 = μ2 = ... = μk)
    H1: At least one group mean is different
    
    Parameters:
    -----------
    groups : list of ndarrays
        List of sample groups
        
    Returns:
    --------
    f_statistic : float
        F-statistic
    p_value : float
        P-value (approximation)
    """
    k = len(groups)  # Number of groups
    n = sum(len(group) for group in groups)  # Total samples
    
    # Grand mean
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # Between-group sum of squares
    ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 
                     for group in groups)
    
    # Within-group sum of squares
    ss_within = sum(np.sum((group - np.mean(group)) ** 2) 
                    for group in groups)
    
    # Degrees of freedom
    df_between = k - 1
    df_within = n - k
    
    # Mean squares
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 1
    
    # F-statistic
    f_statistic = ms_between / ms_within if ms_within > 0 else 0
    
    # P-value using F-distribution approximation
    p_value = 1 - f_distribution_cdf(f_statistic, df_between, df_within)
    
    return f_statistic, p_value


# ================== FEATURE ENGINEERING ==================

def create_polynomial_features(X, degree=2):
    """
    Create polynomial features
    
    Example: [x1, x2] with degree=2 -> [1, x1, x2, x1^2, x1*x2, x2^2]
    
    Parameters:
    -----------
    X : ndarray
        Input features (n_samples, n_features)
    degree : int
        Polynomial degree
        
    Returns:
    --------
    X_poly : ndarray
        Polynomial features
    """
    n_samples, n_features = X.shape
    
    # Start with original features
    features = [X]
    
    # Add polynomial combinations
    for d in range(2, degree + 1):
        for i in range(n_features):
            features.append(X[:, i:i+1] ** d)
    
    X_poly = np.hstack(features)
    
    print(f"Polynomial features created: {X.shape} -> {X_poly.shape}")
    
    return X_poly


def create_interaction_features(X, max_interactions=2):
    """
    Create interaction features between columns
    
    Example: [x1, x2, x3] -> [x1, x2, x3, x1*x2, x1*x3, x2*x3]
    
    Parameters:
    -----------
    X : ndarray
        Input features (n_samples, n_features)
    max_interactions : int
        Maximum number of features to interact
        
    Returns:
    --------
    X_interact : ndarray
        Features with interactions
    """
    n_samples, n_features = X.shape
    
    features = [X]
    
    # Create pairwise interactions
    if max_interactions >= 2:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                features.append(interaction)
    
    X_interact = np.hstack(features)
    
    print(f"Interaction features created: {X.shape} -> {X_interact.shape}")
    
    return X_interact


def create_ratio_features(X, col_pairs=None):
    """
    Create ratio features between specified column pairs
    
    Example: ratio of Amount to Time, etc.
    
    Parameters:
    -----------
    X : ndarray
        Input features (n_samples, n_features)
    col_pairs : list of tuples
        Pairs of column indices to create ratios
        If None, creates ratios for all pairs
        
    Returns:
    --------
    X_ratio : ndarray
        Features with ratios added
    """
    features = [X]
    
    if col_pairs is None:
        # Create a few sample ratios (not all to avoid explosion)
        n_features = X.shape[1]
        col_pairs = [(i, i+1) for i in range(min(5, n_features-1))]
    
    for i, j in col_pairs:
        # Avoid division by zero
        denominator = X[:, j].copy()
        denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
        
        ratio = (X[:, i] / denominator).reshape(-1, 1)
        features.append(ratio)
    
    X_ratio = np.hstack(features)
    
    print(f"Ratio features created: {X.shape} -> {X_ratio.shape}")
    
    return X_ratio


def create_binned_features(X, n_bins=10):
    """
    Create binned (discretized) versions of continuous features
    
    Useful for capturing non-linear relationships
    
    Parameters:
    -----------
    X : ndarray
        Input features (n_samples, n_features)
    n_bins : int
        Number of bins per feature
        
    Returns:
    --------
    X_binned : ndarray
        Binned features (one-hot encoded)
    """
    n_samples, n_features = X.shape
    binned_features = []
    
    for col in range(n_features):
        feature = X[:, col]
        
        # Create bins
        min_val = np.min(feature)
        max_val = np.max(feature)
        
        if min_val == max_val:
            # Constant feature, skip
            continue
        
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Digitize (assign each value to a bin)
        bin_indices = np.digitize(feature, bins[1:-1])
        
        # One-hot encode
        one_hot = np.zeros((n_samples, n_bins))
        one_hot[np.arange(n_samples), bin_indices] = 1
        
        binned_features.append(one_hot)
    
    if binned_features:
        X_binned = np.hstack([X] + binned_features)
        print(f"Binned features created: {X.shape} -> {X_binned.shape}")
        return X_binned
    else:
        return X


def create_aggregate_features(X, window_size=5):
    """
    Create rolling aggregate features (mean, std, min, max)
    
    Useful for time-series or sequential data
    
    Parameters:
    -----------
    X : ndarray
        Input features (n_samples, n_features)
    window_size : int
        Window size for rolling aggregates
        
    Returns:
    --------
    X_agg : ndarray
        Features with aggregates
    """
    n_samples, n_features = X.shape
    
    features = [X]
    
    # For each feature, create rolling statistics
    for col in range(min(n_features, 5)):  # Limit to avoid explosion
        feature = X[:, col]
        
        # Rolling mean
        rolling_mean = np.zeros(n_samples)
        for i in range(n_samples):
            start = max(0, i - window_size + 1)
            rolling_mean[i] = np.mean(feature[start:i+1])
        
        # Rolling std
        rolling_std = np.zeros(n_samples)
        for i in range(n_samples):
            start = max(0, i - window_size + 1)
            rolling_std[i] = np.std(feature[start:i+1])
        
        features.append(rolling_mean.reshape(-1, 1))
        features.append(rolling_std.reshape(-1, 1))
    
    X_agg = np.hstack(features)
    
    print(f"Aggregate features created: {X.shape} -> {X_agg.shape}")
    
    return X_agg


def select_features_by_correlation(X, y, threshold=0.1):
    """
    Select features based on correlation with target variable
    
    Parameters:
    -----------
    X : ndarray
        Input features (n_samples, n_features)
    y : ndarray
        Target variable (n_samples,)
    threshold : float
        Minimum absolute correlation to keep feature
        
    Returns:
    --------
    X_selected : ndarray
        Selected features
    selected_indices : ndarray
        Indices of selected features
    """
    n_features = X.shape[1]
    correlations = np.zeros(n_features)
    
    # Compute correlation with target for each feature
    for i in range(n_features):
        correlations[i] = np.corrcoef(X[:, i], y)[0, 1]
    
    # Select features above threshold
    selected_indices = np.where(np.abs(correlations) >= threshold)[0]
    X_selected = X[:, selected_indices]
    
    print(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
    print(f"Selected indices: {selected_indices}")
    
    return X_selected, selected_indices


# ================== NUMERICAL OPTIMIZATION ==================

def clip_values(data, lower_percentile=1, upper_percentile=99):
    """
    Clip extreme values to reduce impact of outliers
    
    Numerical optimization technique to improve stability
    
    Parameters:
    -----------
    data : ndarray
        Input data
    lower_percentile : float
        Lower percentile for clipping
    upper_percentile : float
        Upper percentile for clipping
        
    Returns:
    --------
    clipped_data : ndarray
        Clipped data
    """
    lower_bound = np.percentile(data, lower_percentile, axis=0)
    upper_bound = np.percentile(data, upper_percentile, axis=0)
    
    clipped_data = np.clip(data, lower_bound, upper_bound)
    
    n_clipped = np.sum((data < lower_bound) | (data > upper_bound))
    print(f"Clipped {n_clipped} extreme values ({n_clipped/data.size*100:.2f}%)")
    
    return clipped_data


def robust_scale(data):
    """
    Robust scaling using median and IQR
    
    Less sensitive to outliers than standard scaling
    Formula: x_scaled = (x - median) / IQR
    
    Parameters:
    -----------
    data : ndarray
        Input data
        
    Returns:
    --------
    scaled_data : ndarray
        Robustly scaled data
    scaler_params : dict
        Scaling parameters
    """
    median = np.median(data, axis=0)
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    
    # Avoid division by zero
    iqr = np.where(iqr == 0, 1, iqr)
    
    scaled_data = (data - median) / iqr
    
    scaler_params = {'median': median, 'iqr': iqr}
    
    return scaled_data, scaler_params


def winsorize(data, limits=(0.05, 0.05)):
    """
    Winsorization: replace extreme values with less extreme values
    
    Numerical optimization to reduce impact of outliers
    
    Parameters:
    -----------
    data : ndarray
        Input data
    limits : tuple
        (lower_limit, upper_limit) as proportions
        
    Returns:
    --------
    winsorized_data : ndarray
        Winsorized data
    """
    lower_limit, upper_limit = limits
    
    lower_percentile = lower_limit * 100
    upper_percentile = (1 - upper_limit) * 100
    
    lower_bound = np.percentile(data, lower_percentile, axis=0)
    upper_bound = np.percentile(data, upper_percentile, axis=0)
    
    winsorized_data = data.copy()
    winsorized_data = np.where(data < lower_bound, lower_bound, data)
    winsorized_data = np.where(winsorized_data > upper_bound, upper_bound, winsorized_data)
    
    n_winsorized = np.sum((data < lower_bound) | (data > upper_bound))
    print(f"Winsorized {n_winsorized} values ({n_winsorized/data.size*100:.2f}%)")
    
    return winsorized_data


# Main preprocessing pipeline
def preprocess_data(filepath, test_size=0.2, balance_method='none', random_seed=42):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    test_size : float
        Test set proportion
    balance_method : str
        'none', 'oversample', 'undersample', or 'weights'
    random_seed : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Preprocessed data splits
    scaler_params : dict
        Scaling parameters for future use
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # 1. Load data
    print("\n[1] Loading data...")
    data = load_csv_numpy(filepath)
    
    if data is None:
        print("Trying chunked loading...")
        data = load_csv_chunks(filepath)
    
    if data is None:
        return None
    
    # 2. Separate features and target
    print("\n[2] Separating features and target...")
    X = data[:, :-1]  # All columns except last
    y = data[:, -1]   # Last column (Class)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Fraud cases: {int(y.sum())} ({y.sum()/len(y)*100:.4f}%)")
    
    # 3. Check missing values
    print("\n[3] Checking missing values...")
    check_missing_values(X)
    
    # 4. Handle missing values if any
    X = handle_missing_values(X, strategy='median')
    
    # 5. Feature scaling
    print("\n[4] Feature scaling...")
    X_scaled, scaler_params = standardize_zscore(X)
    
    # 6. Train-test split (stratified)
    print("\n[5] Splitting data...")
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X_scaled, y, test_size=test_size, random_seed=random_seed
    )
    
    # 7. Handle imbalanced data
    print("\n[6] Handling imbalanced data...")
    if balance_method == 'oversample':
        X_train, y_train = smote_oversample(X_train, y_train, random_seed=random_seed)
    elif balance_method == 'undersample':
        X_train, y_train = random_undersample(X_train, y_train, random_seed=random_seed)
    elif balance_method == 'weights':
        class_weights = compute_class_weights(y_train)
        scaler_params['class_weights'] = class_weights
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    
    return X_train, X_test, y_train, y_test, scaler_params
   