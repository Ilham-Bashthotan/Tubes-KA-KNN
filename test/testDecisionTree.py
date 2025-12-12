"""
================================================================================
TUGAS BESAR: KLASIFIKASI WATER POTABILITY
Implementasi Decision Tree dari Nol dengan Python
Dataset: Water Potability (Kaggle)
================================================================================

Deskripsi:
- Dataset berisi 9 parameter kualitas air untuk memprediksi potability (0/1)
- Implementasi Decision Tree manual tanpa library ML
- Menggunakan Information Gain (Entropy) sebagai kriteria split
- Binning numerik ke kategorikal untuk mengurangi overfitting
- Evaluasi manual dengan confusion matrix dan metrik klasifikasi

Author: [Nama Anda]
Mata Kuliah: Data Mining / Machine Learning
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# STEP 0: LOCATE DATASET
# ============================================================================

print("="*80)
print(" STEP 0: LOCATE DATASET")
print("="*80)

# Get absolute path to dataset (consistent with KNN and Naive Bayes)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "water_potability.csv")

if os.path.exists(csv_path):
    print(f"\n✓ Dataset found at: {os.path.basename(csv_path)}")
else:
    print(f"\n❌ Dataset not found at: {csv_path}")
    print("Please ensure water_potability.csv exists in the data folder")
    exit(1)

# ============================================================================
# STEP 1: LOAD DATA & EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*80)
print(" STEP 1: LOAD DATA & EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# Load dataset
try:
    df = pd.read_csv(csv_path)
    print(f"\n✓ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"\n❌ ERROR: Dataset file not found!")
    print(f"   Expected location: {csv_path}")
    exit(1)
except Exception as e:
    print(f"\n❌ ERROR loading dataset: {e}")
    exit(1)

print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

missing = df.isnull().sum()
total_missing = missing.sum()
print(f"Total Missing Values: {total_missing} ({(total_missing / df.size * 100):.2f}%)")

target_counts = df['Potability'].value_counts().sort_index()
target_prop = df['Potability'].value_counts(normalize=True).sort_index()
print(f"Target Distribution - Class 0: {target_prop[0]:.2%}, Class 1: {target_prop[1]:.2%}")

feature_cols = df.columns[:-1]

# ============================================================================
# STEP 3: DATA PREPROCESSING - HANDLING MISSING VALUES
# ============================================================================

print("\n" + "="*80)
print(" STEP 3: DATA PREPROCESSING - HANDLING MISSING VALUES")
print("="*80)

df_imputed = df.copy()

for col in df_imputed.columns[:-1]:
    if df_imputed[col].isnull().sum() > 0:
        median_val = df_imputed[col].median()
        df_imputed[col].fillna(median_val, inplace=True)

print(f"Missing values after imputation: {df_imputed.isnull().sum().sum()}")

# ============================================================================
# STEP 3a: OUTLIER DETECTION AND HANDLING
# ============================================================================

print("\n" + "="*80)
print(" STEP 3a: OUTLIER DETECTION AND HANDLING")
print("="*80)

df_outlier_handled = df_imputed.copy()

print("\nMenggunakan IQR (Interquartile Range) method untuk outlier detection:")
print("Outliers: Q1 - 1.5*IQR dan Q3 + 1.5*IQR\n")

outlier_count = 0
for col in df_outlier_handled.columns[:-1]:
    Q1 = df_outlier_handled[col].quantile(0.25)
    Q3 = df_outlier_handled[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers = ((df_outlier_handled[col] < lower_bound) | (df_outlier_handled[col] > upper_bound)).sum()
    outlier_count += outliers
    
    if outliers > 0:
        print(f"  {col}: {outliers} outliers detected")
        # Cap outliers (Winsorization)
        df_outlier_handled[col] = np.clip(df_outlier_handled[col], lower_bound, upper_bound)

print(f"\n✓ Total {outlier_count} outliers handled (capped to bounds)")
print("  Metode: Winsorization (capping to min/max bounds)")

# ============================================================================
# STEP 4: FEATURE BINNING (NUMERIC → CATEGORICAL)
# ============================================================================

print("\n" + "="*80)
print(" STEP 4: FEATURE BINNING (NUMERIC → CATEGORICAL)")
print("="*80)

df_binned = df_outlier_handled.copy()

print("Menggunakan QUANTILE-BASED BINNING untuk distribusi yang lebih seimbang:\n")

for col in df_binned.columns[:-1]:
    # Quantile-based binning (lebih robust daripada equal-width)
    # Setiap bin memiliki jumlah sampel yang hampir sama
    labels = [f'{col}_Low', f'{col}_MedLow', f'{col}_MedHigh', f'{col}_High']
    df_binned[col] = pd.qcut(df_binned[col], q=4, labels=labels, duplicates='drop')
    
    # Check distribution
    dist = df_binned[col].value_counts()
    print(f"  {col}: {dist.min()}-{dist.max()} samples per bin")

print("\n✓ Quantile-based binning completed")
print("  Keuntungan: Setiap bin memiliki distribusi data yang seimbang")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print(" STEP 5: TRAIN-TEST SPLIT")
print("="*80)

X = df_binned.drop('Potability', axis=1)
y = df_binned['Potability']

# ============================================================================
# STEP 5a: FEATURE ENGINEERING - INTERACTION FEATURES
# ============================================================================

print("\n" + "="*80)
print(" STEP 5a: FEATURE ENGINEERING - INTERACTION FEATURES")
print("="*80)

print("\nMembuat interaction features dari top correlated features:\n")

# Calculate correlation with target for feature selection (on full data)
X_encoded = X.apply(lambda x: pd.factorize(x)[0])
correlations = {}
for col in X.columns:
    corr = np.corrcoef(X_encoded[col], y)[0, 1]
    correlations[col] = abs(corr)

# Get top 2 features only (reduce complexity)
top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:2]
print("Top 2 features by correlation with target:")
for feat, corr in top_features:
    print(f"  {feat}: {corr:.4f}")

# Create only 1 interaction feature (top 2 combined)
feat1, feat2 = top_features[0][0], top_features[1][0]
new_col = f"{feat1}_x_{feat2}"
X[new_col] = X[feat1].astype(str) + "_" + X[feat2].astype(str)
print(f"\n  Created: {new_col}")

print(f"\n✓ Feature engineering completed")
print(f"  Total features: {len(X.columns)} (original: {len(feature_cols)}, added: 1)")
print(f"  Strategy: Hanya 1 interaction untuk balance performa vs kecepatan")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining: {len(X_train)} samples, Test: {len(X_test)} samples")

# ============================================================================
# STEP 6: IMPLEMENTASI DECISION TREE DARI NOL
# ============================================================================

print("\n" + "="*80)
print(" STEP 6: IMPLEMENTASI DECISION TREE DARI NOL")
print("="*80)

# Node direpresentasikan sebagai dictionary
def create_node(feature=None, threshold=None, left=None, right=None, value=None):
    """
    I.S: Parameter node (feature, threshold, left, right, value)
    F.S: Dictionary yang merepresentasikan node dalam tree
    """
    return {
        'feature': feature,
        'threshold': threshold,
        'left': left,
        'right': right,
        'value': value
    }

def is_leaf(node):
    """
    I.S: Node dalam bentuk dictionary
    F.S: Boolean, True jika node adalah leaf
    """
    return node['value'] is not None

# Fungsi-fungsi Decision Tree
def dt_entropy(y):
    """
    I.S: Array target variable y
    F.S: Nilai entropy (float)
    """
    proportions = np.bincount(y) / len(y)
    entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
    return entropy

def dt_information_gain(X_column, y, threshold):
    """
    I.S: Column feature, target y, dan threshold
    F.S: Nilai information gain (float)
    """
    parent_entropy = dt_entropy(y)
    
    left_mask = X_column == threshold
    right_mask = X_column != threshold
    
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0
    
    n = len(y)
    n_left, n_right = np.sum(left_mask), np.sum(right_mask)
    entropy_left = dt_entropy(y[left_mask])
    entropy_right = dt_entropy(y[right_mask])
    
    weighted_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
    ig = parent_entropy - weighted_entropy
    return ig

def dt_best_split(X, y):
    """
    I.S: Feature matrix X dan target y
    F.S: Tuple (best_feature_idx, best_threshold, best_gain)
    """
    best_gain = -1
    best_feature = None
    best_threshold = None
    n_features = X.shape[1]
    
    for feature_idx in range(n_features):
        X_column = X[:, feature_idx]
        unique_values = np.unique(X_column)
        
        for threshold in unique_values:
            gain = dt_information_gain(X_column, y, threshold)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain

def dt_build_tree(X, y, depth, max_depth, min_samples_split, min_samples_leaf):
    """
    I.S: Data X, y, depth, dan hyperparameters
    F.S: Node (dictionary) sebagai root dari tree/subtree
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    if (depth >= max_depth or n_classes == 1 or n_samples < min_samples_split):
        leaf_value = np.argmax(np.bincount(y))
        return create_node(value=leaf_value)
    
    best_feature, best_threshold, best_gain = dt_best_split(X, y)
    
    if best_gain == 0:
        leaf_value = np.argmax(np.bincount(y))
        return create_node(value=leaf_value)
    
    left_mask = X[:, best_feature] == best_threshold
    right_mask = X[:, best_feature] != best_threshold
    
    if np.sum(left_mask) < min_samples_leaf or np.sum(right_mask) < min_samples_leaf:
        leaf_value = np.argmax(np.bincount(y))
        return create_node(value=leaf_value)
    
    left_child = dt_build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_split, min_samples_leaf)
    right_child = dt_build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_split, min_samples_leaf)
    
    return create_node(best_feature, best_threshold, left_child, right_child)

def dt_fit(X, y, max_depth, min_samples_split, min_samples_leaf):
    """
    I.S: Data X, y, dan hyperparameters
    F.S: Tuple (root_node, feature_names)
    """
    feature_names = None
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    root = dt_build_tree(X, y, 0, max_depth, min_samples_split, min_samples_leaf)
    return root, feature_names

def dt_predict_sample(x, node):
    """
    I.S: Single sample x dan node
    F.S: Predicted class label (int)
    """
    if is_leaf(node):
        return node['value']
    
    if x[node['feature']] == node['threshold']:
        return dt_predict_sample(x, node['left'])
    else:
        return dt_predict_sample(x, node['right'])

def dt_predict(X, root):
    """
    I.S: Feature matrix X dan root node
    F.S: Array prediksi untuk semua samples
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    predictions = [dt_predict_sample(x, root) for x in X]
    return np.array(predictions)

def dt_get_depth(node):
    """
    I.S: Node (bisa None atau dictionary)
    F.S: Depth dari tree/subtree (int)
    """
    if node is None or is_leaf(node):
        return 0
    
    left_depth = dt_get_depth(node['left']) if node['left'] else 0
    right_depth = dt_get_depth(node['right']) if node['right'] else 0
    
    return 1 + max(left_depth, right_depth)

def dt_count_nodes(node):
    """
    I.S: Node (bisa None atau dictionary)
    F.S: Jumlah total nodes (int)
    """
    if node is None:
        return 0
    if is_leaf(node):
        return 1
    
    left_count = dt_count_nodes(node['left']) if node['left'] else 0
    right_count = dt_count_nodes(node['right']) if node['right'] else 0
    
    return 1 + left_count + right_count

def dt_count_leaves(node):
    """
    I.S: Node (bisa None atau dictionary)
    F.S: Jumlah leaf nodes (int)
    """
    if node is None:
        return 0
    if is_leaf(node):
        return 1
    
    left_leaves = dt_count_leaves(node['left']) if node['left'] else 0
    right_leaves = dt_count_leaves(node['right']) if node['right'] else 0
    
    return left_leaves + right_leaves

def dt_print_tree(node, feature_names, depth=0, prefix="Root", max_depth=3):
    """
    I.S: Node, feature_names, depth, prefix, max_depth
    F.S: Print tree structure ke console
    """
    if depth > max_depth or node is None:
        return
    
    if is_leaf(node):
        print(f"{'  ' * depth}{prefix} → Leaf: Class {node['value']}")
        return
    
    feature_name = feature_names[node['feature']] if feature_names else f"Feature_{node['feature']}"
    print(f"{'  ' * depth}{prefix} → [{feature_name} == {node['threshold']}]")
    
    if node['left']:
        dt_print_tree(node['left'], feature_names, depth + 1, "├─ Left (Yes)", max_depth)
    if node['right']:
        dt_print_tree(node['right'], feature_names, depth + 1, "└─ Right (No)", max_depth)


print("\n✓ Decision Tree Functions implemented successfully!")
print("\n📋 KEY COMPONENTS:")
print("   1. Node class: Merepresentasikan setiap node dalam tree")
print("   2. Entropy calculation: Mengukur impurity/ketidakpastian")
print("   3. Information Gain calculation: Mengukur kualitas split")
print("   4. Best split finding: Mencari split terbaik dengan IG tertinggi")
print("   5. Recursive tree building: Membangun tree secara top-down")
print("   6. Prediction method: Traverse tree untuk prediksi")
print("   7. Tree statistics: Depth, nodes count, leaves count")

# ============================================================================
# HELPER FUNCTIONS FOR CROSS-VALIDATION AND EVALUATION
# ============================================================================

def cross_validate_dt(X, y, max_depth, min_samples_split, min_samples_leaf, n_folds=5):
    """
    Cross-validation untuk Decision Tree.
    I.S. X, y: data; hyperparameters; n_folds: jumlah fold
    F.S : mean accuracy, std accuracy
    """
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    fold_size = len(X) // n_folds
    scores = []
    
    for fold in range(n_folds):
        # Split indices
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(X)
        
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        # Split data
        X_fold_train = X.iloc[train_indices].reset_index(drop=True)
        y_fold_train = y.iloc[train_indices].reset_index(drop=True)
        X_fold_test = X.iloc[test_indices].reset_index(drop=True)
        y_fold_test = y.iloc[test_indices].reset_index(drop=True)
        
        # Train and predict
        tree_root, _ = dt_fit(X_fold_train, y_fold_train, max_depth, min_samples_split, min_samples_leaf)
        y_pred = dt_predict(X_fold_test, tree_root)
        
        # Score
        accuracy = np.sum(y_fold_test.values == y_pred) / len(y_fold_test)
        scores.append(accuracy)
    
    return np.mean(scores), np.std(scores)


def accuracy(y_true, y_pred):
    """
    Hitung akurasi.
    I.S. y_true: label asli, y_pred: label prediksi
    F.S : akurasi (float)
    """
    return np.sum(y_true == y_pred) / len(y_true)


def calculate_confusion_matrix_simple(y_true, y_pred):
    """
    Hitung confusion matrix untuk evaluasi singkat.
    I.S. y_true: label asli, y_pred: label prediksi
    F.S : confusion matrix (2D array)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        true_idx = np.where(classes == true)[0][0]
        pred_idx = np.where(classes == pred)[0][0]
        cm[true_idx, pred_idx] += 1
    
    return cm


def print_evaluation(y_true, y_pred, dataset_name):
    """
    Print metrik evaluasi.
    I.S. y_true: label asli, y_pred: label prediksi, dataset_name: nama dataset
    F.S : print metrik evaluasi
    """
    acc = accuracy(y_true, y_pred)
    cm = calculate_confusion_matrix_simple(y_true, y_pred)
    
    print(f"\n=== {dataset_name} ===")
    print(f"Akurasi: {acc:.4f}")
    print(f"Confusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    print(f"\nClassification Report:")
    print(f"              precision    recall  f1-score   support")
    print(f"           0     {precision_0:.4f}    {recall_0:.4f}    {f1_0:.4f}       {tn+fp}")
    print(f"           1     {precision_1:.4f}    {recall_1:.4f}    {f1_1:.4f}       {fn+tp}")
    print(f"    accuracy                         {acc:.4f}       {len(y_true)}")
    print(f"   macro avg     {(precision_0+precision_1)/2:.4f}    {(recall_0+recall_1)/2:.4f}    {(f1_0+f1_1)/2:.4f}       {len(y_true)}")

# ============================================================================
# STEP 7: GRID SEARCH (CROSS-VALIDATION)
# ============================================================================

print("\n" + "="*80)
print(" STEP 7: GRID SEARCH (CROSS-VALIDATION 5-FOLD)")
print("="*80)

# Define hyperparameter grid (optimized - reduced combinations)
max_depth_values = [5, 7, 10, 15]  # Reduced from 5 to 4 values
min_samples_split_values = [5, 10, 20]  # Reduced from 4 to 3 values
min_samples_leaf_values = [2, 5, 10]

total_combinations = len(max_depth_values) * len(min_samples_split_values) * len(min_samples_leaf_values)
print(f"\n📋 HYPERPARAMETER GRID (Optimized):")
print(f"   - max_depth: {max_depth_values}")
print(f"   - min_samples_split: {min_samples_split_values}")
print(f"   - min_samples_leaf: {min_samples_leaf_values}")
print(f"\n   Total kombinasi: {total_combinations} (reduced for faster execution)")

best_score = 0
best_params = {}
results = []
current = 0

print("\n🚀 Starting grid search...")
for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        for min_samples_leaf in min_samples_leaf_values:
            current += 1
            print(f"[{current}/{total_combinations}] Testing max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}...", end=" ")
            
            mean_score, std_score = cross_validate_dt(
                X_train, y_train, 
                max_depth, min_samples_split, min_samples_leaf, 
                n_folds=5
            )
            results.append((max_depth, min_samples_split, min_samples_leaf, mean_score, std_score))
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                }
            
            print(f"CV Acc: {mean_score:.4f} ± {std_score:.4f}")

print(f"\n>>> BEST PARAMETERS: {best_params}")
print(f"    CV Accuracy: {best_score:.4f}")

# ============================================================================
# STEP 7a: GRID SEARCH RESULTS
# ============================================================================

results_df = pd.DataFrame(results, columns=['max_depth', 'min_samples_split', 'min_samples_leaf', 'mean_score', 'std_score'])

print("\nTop 10 Konfigurasi Terbaik:")
print(results_df.sort_values('mean_score', ascending=False).head(10).to_string(index=False))

# ============================================================================
# STEP 8: TRAINING WITH BEST PARAMETERS
# ============================================================================

print("\n" + "="*80)
print(" STEP 8: TRAINING WITH BEST PARAMETERS")
print("="*80)

print(f"\n🔧 BEST HYPERPARAMETER CONFIGURATION:")
print(f"   - max_depth: {best_params['max_depth']}")
print(f"   - min_samples_split: {best_params['min_samples_split']}")
print(f"   - min_samples_leaf: {best_params['min_samples_leaf']}")

print("\n🚀 Training model with best parameters...")
tree_root, feature_names = dt_fit(
    X_train, y_train,
    best_params['max_depth'],
    best_params['min_samples_split'],
    best_params['min_samples_leaf']
)
print("✓ Model training completed!")

print("\n" + "-"*80)
print("MODEL STATISTICS:")
print("-"*80)
tree_depth = dt_get_depth(tree_root)
total_nodes = dt_count_nodes(tree_root)
leaf_nodes = dt_count_leaves(tree_root)
internal_nodes = total_nodes - leaf_nodes

print(f"  Tree Depth:        {tree_depth}")
print(f"  Total Nodes:       {total_nodes}")
print(f"  Leaf Nodes:        {leaf_nodes}")
print(f"  Internal Nodes:    {internal_nodes}")
print(f"  Features Used:     {len(feature_names)}")

print("\n" + "-"*80)
print("TREE STRUCTURE (First 3 levels):")
print("-"*80)
dt_print_tree(tree_root, feature_names, max_depth=3)

# ============================================================================
# STEP 9: COMPREHENSIVE EVALUATION
# ============================================================================

print("\n" + "="*80)
print(" STEP 9: COMPREHENSIVE EVALUATION")
print("="*80)

# 9a. TRAINING SET EVALUATION
print("\n" + "="*80)
print(" 9a. TRAINING SET EVALUATION")
print("="*80)

print("Menghitung prediksi training set...")
y_train_pred = dt_predict(X_train, tree_root)
print_evaluation(y_train.values, y_train_pred, "TRAINING SET")

# 9b. TEST SET EVALUATION
print("\n" + "="*80)
print(" 9b. TEST SET EVALUATION")
print("="*80)

print("Menghitung prediksi test set...")
y_test_pred = dt_predict(X_test, tree_root)
print_evaluation(y_test.values, y_test_pred, "TEST SET")

# 9c. FULL DATASET EVALUATION
print("\n" + "="*80)
print(" 9c. FULL DATASET EVALUATION")
print("="*80)

# Use X and y that already have interaction features
X_full = X
y_full = y

print("Menghitung prediksi full dataset...")
y_pred_full = dt_predict(X_full, tree_root)
print_evaluation(y_full.values, y_pred_full, "FULL DATASET (SEMUA DATA AWAL)")

# 9d. CROSS-VALIDATION PREDICTIONS
print("\n" + "="*80)
print(" 9d. CROSS-VALIDATION PREDICTIONS")
print("="*80)

print("Menghitung prediksi CV (5-Fold)...")

np.random.seed(42)
indices = np.arange(len(X_train))
np.random.shuffle(indices)

fold_size = len(X_train) // 5
y_pred_cv_all = np.zeros(len(X_train))

for fold in range(5):
    print(f"Processing fold {fold+1}/5...")
    test_start = fold * fold_size
    test_end = test_start + fold_size if fold < 4 else len(X_train)
    
    test_indices = indices[test_start:test_end]
    train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
    
    X_fold_train = X_train.iloc[train_indices].reset_index(drop=True)
    y_fold_train = y_train.iloc[train_indices].reset_index(drop=True)
    X_fold_test = X_train.iloc[test_indices].reset_index(drop=True)
    y_fold_test = y_train.iloc[test_indices].reset_index(drop=True)
    
    fold_tree_root, _ = dt_fit(
        X_fold_train, y_fold_train,
        best_params['max_depth'],
        best_params['min_samples_split'],
        best_params['min_samples_leaf']
    )
    y_pred_fold = dt_predict(X_fold_test, fold_tree_root)
    
    y_pred_cv_all[test_indices] = y_pred_fold

print_evaluation(y_train.values, y_pred_cv_all, "CROSS-VALIDATION (5-Fold)")

# 9e. PERFORMANCE ANALYSIS
print("\n" + "="*80)
print(" ANALISIS KINERJA (CV Score vs Test Score)")
print("="*80)

train_acc = accuracy(y_train.values, y_train_pred)
cv_acc_actual = accuracy(y_train.values, y_pred_cv_all)
cv_acc_mean = best_score
test_acc = accuracy(y_test.values, y_test_pred)
full_acc = accuracy(y_full.values, y_pred_full)

print(f"Akurasi Training:        {train_acc:.4f}")
print(f"Akurasi CV (Actual):     {cv_acc_actual:.4f}")
print(f"Akurasi CV (Mean):       {cv_acc_mean:.4f}")
print(f"Akurasi Test:            {test_acc:.4f}")
print(f"Akurasi Full:            {full_acc:.4f}")

print("\nKESIMPULAN:")
print(f"Gap (CV - Test):    {cv_acc_actual - test_acc:.4f}")
print(f"Gap (Train - Test): {train_acc - test_acc:.4f}")

if train_acc - test_acc > 0.10:
    print("\n! Indikasi OVERFITTING (Gap Train-Test besar).")
    print("  Model terlalu 'hafal' training data.")
    print("  Solusi: Tingkatkan min_samples_split/leaf atau kurangi max_depth")
elif cv_acc_actual - test_acc > 0.08:
    print("\n! Indikasi OVERFITTING ringan (berdasarkan CV vs Test).")
elif test_acc < 0.60:
    print("\n! UNDERFITTING.")
    print("  Performa keseluruhan rendah.")
else:
    print("\n✓ MODEL KONSISTEN (Good Fit).")
    print("  Gap train-test dalam batas wajar.")

# ============================================================================
# STEP 11: MANUAL CALCULATION OF EVALUATION METRICS
# ============================================================================

print("\n" + "="*80)
print(" STEP 11: MANUAL CALCULATION OF EVALUATION METRICS")
print("="*80)

def calculate_confusion_matrix(y_true, y_pred):
    """
    Menghitung confusion matrix secara manual
    I.S: y_true (array), y_pred (array)
    F.S : confusion matrix (2D array)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])


def calculate_metrics(confusion_matrix):
    """
    Menghitung metrik evaluasi dari confusion matrix
    I.S: confusion_matrix (2D array)
    F.S : accuracy, precision, recall, f1_score (floats)
    """
    TN, FP, FN, TP = confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[1, 0], confusion_matrix[1, 1]
    
    # Accuracy = (TP + TN) / Total
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Precision = TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall (Sensitivity) = TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


# ========== TRAINING SET EVALUATION ==========
print("\n" + "="*80)
print(" TRAINING SET EVALUATION")
print("="*80)

cm_train = calculate_confusion_matrix(y_train, y_train_pred)
acc_train, prec_train, rec_train, f1_train = calculate_metrics(cm_train)

print("\n📊 CONFUSION MATRIX:")
print("\n                Predicted")
print("                 0      1")
print(f"Actual    0   [{cm_train[0,0]:4d}  {cm_train[0,1]:4d}]  → TN={cm_train[0,0]}, FP={cm_train[0,1]}")
print(f"          1   [{cm_train[1,0]:4d}  {cm_train[1,1]:4d}]  → FN={cm_train[1,0]}, TP={cm_train[1,1]}")

print("\n" + "-"*80)
print("MANUAL CALCULATION STEPS:")
print("-"*80)
print(f"  True Negative  (TN) = {cm_train[0,0]}")
print(f"  False Positive (FP) = {cm_train[0,1]}")
print(f"  False Negative (FN) = {cm_train[1,0]}")
print(f"  True Positive  (TP) = {cm_train[1,1]}")
print(f"  Total Samples       = {np.sum(cm_train)}")

print("\n  Accuracy  = (TP + TN) / Total")
print(f"            = ({cm_train[1,1]} + {cm_train[0,0]}) / {np.sum(cm_train)}")
print(f"            = {cm_train[1,1] + cm_train[0,0]} / {np.sum(cm_train)}")
print(f"            = {acc_train:.6f}")

print("\n  Precision = TP / (TP + FP)")
print(f"            = {cm_train[1,1]} / ({cm_train[1,1]} + {cm_train[0,1]})")
print(f"            = {cm_train[1,1]} / {cm_train[1,1] + cm_train[0,1]}")
print(f"            = {prec_train:.6f}")

print("\n  Recall    = TP / (TP + FN)")
print(f"            = {cm_train[1,1]} / ({cm_train[1,1]} + {cm_train[1,0]})")
print(f"            = {cm_train[1,1]} / {cm_train[1,1] + cm_train[1,0]}")
print(f"            = {rec_train:.6f}")

print("\n  F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)")
print(f"            = 2 × ({prec_train:.6f} × {rec_train:.6f}) / ({prec_train:.6f} + {rec_train:.6f})")
print(f"            = 2 × {prec_train * rec_train:.6f} / {prec_train + rec_train:.6f}")
print(f"            = {f1_train:.6f}")

print("\n" + "-"*80)
print("📈 TRAINING METRICS SUMMARY:")
print("-"*80)
print(f"  Accuracy:  {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"  Precision: {prec_train:.4f} ({prec_train*100:.2f}%)")
print(f"  Recall:    {rec_train:.4f} ({rec_train*100:.2f}%)")
print(f"  F1-Score:  {f1_train:.4f} ({f1_train*100:.2f}%)")


# ========== TEST SET EVALUATION ==========
print("\n" + "="*80)
print(" TEST SET EVALUATION")
print("="*80)

cm_test = calculate_confusion_matrix(y_test, y_test_pred)
acc_test, prec_test, rec_test, f1_test = calculate_metrics(cm_test)

print("\n📊 CONFUSION MATRIX:")
print("\n                Predicted")
print("                 0      1")
print(f"Actual    0   [{cm_test[0,0]:4d}  {cm_test[0,1]:4d}]  → TN={cm_test[0,0]}, FP={cm_test[0,1]}")
print(f"          1   [{cm_test[1,0]:4d}  {cm_test[1,1]:4d}]  → FN={cm_test[1,0]}, TP={cm_test[1,1]}")

print("\n" + "-"*80)
print("MANUAL CALCULATION STEPS:")
print("-"*80)
print(f"  True Negative  (TN) = {cm_test[0,0]}")
print(f"  False Positive (FP) = {cm_test[0,1]}")
print(f"  False Negative (FN) = {cm_test[1,0]}")
print(f"  True Positive  (TP) = {cm_test[1,1]}")
print(f"  Total Samples       = {np.sum(cm_test)}")

print("\n  Accuracy  = (TP + TN) / Total")
print(f"            = ({cm_test[1,1]} + {cm_test[0,0]}) / {np.sum(cm_test)}")
print(f"            = {cm_test[1,1] + cm_test[0,0]} / {np.sum(cm_test)}")
print(f"            = {acc_test:.6f}")

print("\n  Precision = TP / (TP + FP)")
print(f"            = {cm_test[1,1]} / ({cm_test[1,1]} + {cm_test[0,1]})")
print(f"            = {cm_test[1,1]} / {cm_test[1,1] + cm_test[0,1]}")
print(f"            = {prec_test:.6f}")

print("\n  Recall    = TP / (TP + FN)")
print(f"            = {cm_test[1,1]} / ({cm_test[1,1]} + {cm_test[1,0]})")
print(f"            = {cm_test[1,1]} / {cm_test[1,1] + cm_test[1,0]}")
print(f"            = {rec_test:.6f}")

print("\n  F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)")
print(f"            = 2 × ({prec_test:.6f} × {rec_test:.6f}) / ({prec_test:.6f} + {rec_test:.6f})")
print(f"            = 2 × {prec_test * rec_test:.6f} / {prec_test + rec_test:.6f}")
print(f"            = {f1_test:.6f}")

print("\n" + "-"*80)
print("📈 TEST METRICS SUMMARY:")
print("-"*80)
print(f"  Accuracy:  {acc_test:.4f} ({acc_test*100:.2f}%)")
print(f"  Precision: {prec_test:.4f} ({prec_test*100:.2f}%)")
print(f"  Recall:    {rec_test:.4f} ({rec_test*100:.2f}%)")
print(f"  F1-Score:  {f1_test:.4f} ({f1_test*100:.2f}%)")

# ============================================================================
# STEP 11: INTERPRETASI HASIL & ANALISIS
# ============================================================================

print("\n" + "="*80)
print(" STEP 11: INTERPRETASI HASIL & ANALISIS")
print("="*80)

print("\n" + "-"*80)
print("📊 RINGKASAN PERFORMA MODEL:")
print("-"*80)
print("\nTraining Set:")
print(f"  Accuracy:  {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"  Precision: {prec_train:.4f}")
print(f"  Recall:    {rec_train:.4f}")
print(f"  F1-Score:  {f1_train:.4f}")

print("\nTest Set:")
print(f"  Accuracy:  {acc_test:.4f} ({acc_test*100:.2f}%)")
print(f"  Precision: {prec_test:.4f}")
print(f"  Recall:    {rec_test:.4f}")
print(f"  F1-Score:  {f1_test:.4f}")

# Calculate overfitting metrics
acc_gap = acc_train - acc_test
f1_gap = f1_train - f1_test

print("\n" + "-"*80)
print("🔍 ANALISIS OVERFITTING:")
print("-"*80)
print(f"  Gap Accuracy (Train - Test):  {acc_gap:.4f} ({acc_gap*100:.2f}%)")
print(f"  Gap F1-Score (Train - Test):  {f1_gap:.4f} ({f1_gap*100:.2f}%)")

if acc_gap > 0.10:
    print("\n  ⚠️ OVERFITTING DETECTED: Model mengalami overfitting yang cukup signifikan")
elif acc_gap > 0.05:
    print("\n  ⚠️ SLIGHT OVERFITTING: Model mengalami overfitting ringan")
else:
    print("\n  ✓ GOOD GENERALIZATION: Model menggeneralisasi dengan baik")

print("\n" + "="*80)
print(" ANALYSIS COMPLETE")
print("="*80)
