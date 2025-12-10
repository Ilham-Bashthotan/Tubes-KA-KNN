# =====================================================
# K-NEAREST NEIGHBORS
# =====================================================
# Implementasi dari scratch:
# 1. Drop missing values
# 2. Train-test split stratified
# 3. Standardisasi
# 4. KNN dengan berbagai metric
# 5. Cross-validation
# 6. Grid search
# =====================================================

import pandas as pd
import numpy as np

# =====================================================
# FUNGSI-FUNGSI
# =====================================================

def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Split data menjadi train dan test set dengan stratifikasi.
    I.S. X: fitur, y: target
    F.S : X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    combined = X.copy()
    combined['_target_'] = y.values
    
    class_0 = combined[combined['_target_'] == 0]
    class_1 = combined[combined['_target_'] == 1]
    
    class_0_shuffled = class_0.sample(frac=1, random_state=random_state).reset_index(drop=True)
    class_1_shuffled = class_1.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    test_count_0 = int(len(class_0) * test_size)
    test_count_1 = int(len(class_1) * test_size)
    
    test_0 = class_0_shuffled[:test_count_0]
    train_0 = class_0_shuffled[test_count_0:]
    test_1 = class_1_shuffled[:test_count_1]
    train_1 = class_1_shuffled[test_count_1:]
    
    train_combined = pd.concat([train_0, train_1], ignore_index=True)
    test_combined = pd.concat([test_0, test_1], ignore_index=True)
    
    train_combined = train_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_combined = test_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    X_train = train_combined.drop('_target_', axis=1)
    y_train = train_combined['_target_']
    X_test = test_combined.drop('_target_', axis=1)
    y_test = test_combined['_target_']
    
    return X_train, X_test, y_train, y_test


def standardize(X_train, X_test):
    """
    Standardisasi Z-score.
    I.S. X_train, X_test: fitur
    F.S : X_train_scaled, X_test_scaled
    """
    mean = X_train.mean()
    std = X_train.std()
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled


def calculate_distances_vectorized(X_test_point, X_train, metric='euclidean'):
    """
    Hitung jarak dari 1 test point ke semua training points (vectorized).
    I.S. X_test_point: 1 data point, X_train: semua training points
    F.S : array jarak
    """
    if metric == 'euclidean':
        return np.sqrt(np.sum((X_train - X_test_point) ** 2, axis=1))
    elif metric == 'manhattan':
        return np.sum(np.abs(X_train - X_test_point), axis=1)
    else:
        raise ValueError(f"Metric {metric} tidak didukung")


def knn_predict(X_train, y_train, X_test, k=5, metric='euclidean', weights='uniform', use_loo=False):
    """
    KNN dengan berbagai metric dan weights (optimized).
    I.S. X_train, y_train: data training; X_test: data test
        use_loo: jika True, gunakan Leave-One-Out (auto-detect jika X_test == X_train)
    F.S : array prediksi untuk X_test
    """
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    
    # Auto-detect LOO: jika X_test sama dengan X_train
    if not use_loo and X_test_np.shape == X_train_np.shape:
        if np.array_equal(X_test_np, X_train_np):
            use_loo = True
            print("!  Auto-detect: X_test == X_train, menggunakan Leave-One-Out")
    
    predictions = []
    for idx, x_test in enumerate(X_test_np):
        # Leave-One-Out: exclude point itu sendiri jika sedang prediksi training data
        if use_loo:
            X_train_excl = np.delete(X_train_np, idx, axis=0)
            y_train_excl = np.delete(y_train_np, idx)
            distances = calculate_distances_vectorized(x_test, X_train_excl, metric)
            k_actual = min(k, len(distances))
            nearest_indices = np.argpartition(distances, k_actual-1)[:k_actual]
            nearest_labels = y_train_excl[nearest_indices]
            nearest_distances = distances[nearest_indices]
        else:
            # Normal prediction
            distances = calculate_distances_vectorized(x_test, X_train_np, metric)
            nearest_indices = np.argpartition(distances, k)[:k]
            nearest_labels = y_train_np[nearest_indices]
            nearest_distances = distances[nearest_indices]
        
        # Voting
        if weights == 'uniform':
            unique, counts = np.unique(nearest_labels, return_counts=True)
            pred = unique[np.argmax(counts)]
        elif weights == 'distance':
            weights_arr = 1 / (nearest_distances + 1e-10)
            unique_labels = np.unique(nearest_labels)
            weighted_votes = np.array(
                [
                    np.sum(
                        weights_arr[nearest_labels == label]
                    ) for label in unique_labels
                ]
            )
            pred = unique_labels[np.argmax(weighted_votes)]
        else:
            raise ValueError(f"Weights {weights} tidak didukung")
        
        predictions.append(pred)
    
    return np.array(predictions)


def accuracy(y_true, y_pred):
    """
    Hitung akurasi.
    I.S. y_true: label asli, y_pred: label prediksi
    F.S : akurasi (float)
    """
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    """
    Hitung confusion matrix.
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


def cross_validate(X, y, k, metric, weights, n_folds=5):
    """
    Cross-validation KNN.
    I.S. X, y: data; k, metric, weights: hyperparameter; n_folds: jumlah fold
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
        
        # Preprocess (hanya standardisasi, data sudah bersih)
        X_fold_train, X_fold_test = standardize(X_fold_train, X_fold_test)
        
        # Predict
        y_pred = knn_predict(X_fold_train, y_fold_train, X_fold_test, k, metric, weights)
        
        # Score
        scores.append(accuracy(y_fold_test.values, y_pred))
    
    return np.mean(scores), np.std(scores)


# =====================================================
# MAIN PROGRAM
# =====================================================

# -----------------------------------------------------
# 1. LOAD DATASET & CLEANING
# -----------------------------------------------------
df = pd.read_csv("data/water_potability.csv")

print(f"Data awal: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

df = df.drop_duplicates()
df = df.replace([np.inf, -np.inf], np.nan) # ganti inf dengan nan
df = df.dropna()  # Drop lagi jika ada inf yang jadi NaN
print(f"Setelah drop data NaN: {df.shape}")

# Cap outlier
for col in df.columns:
    if col == "Potability":
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

print(f"Data final setelah cleaning: {df.shape}")

# -----------------------------------------------------
# 2. SPLIT DATA
# -----------------------------------------------------
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# -----------------------------------------------------
# 3. PREPROCESSING
# -----------------------------------------------------
# Tidak perlu imputasi karena missing values sudah di-drop
X_train_prep, X_test_prep = standardize(X_train, X_test)

print("Preprocessing selesai (standardisasi saja, missing values sudah dihapus)")

# -----------------------------------------------------
# 4. GRID SEARCH (OPTIMIZED)
# -----------------------------------------------------
# Perluas k values untuk mengurangi overfitting (k besar = lebih smooth/general)
k_values = [5, 7, 9, 11, 15, 19, 25, 31, 41, 51]  # Tambah k besar untuk regularisasi
metrics = ['euclidean', 'manhattan']
weights_options = ['uniform', 'distance']

total_combinations = len(k_values) * len(metrics) * len(weights_options)
print("\n=== Grid Search (Cross-Validation 5-Fold) ===")
print(f"Total kombinasi: {total_combinations}")

best_score = 0
best_params = {}
results = []
current = 0

for k in k_values:
    for metric in metrics:
        for weights in weights_options:
            current += 1
            print(f"[{current}/{total_combinations}] Testing k={k}, metric={metric}, weights={weights}...", end=" ")
            
            mean_score, std_score = cross_validate(X_train, y_train, k, metric, weights, n_folds=5)
            results.append((k, metric, weights, mean_score, std_score))
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'k': k, 'metric': metric, 'weights': weights}
            
            print(f"CV Acc: {mean_score:.4f} ± {std_score:.4f}")

print(f"\n>>> BEST: {best_params}")
print(f"    CV Accuracy: {best_score:.4f}")

# -----------------------------------------------------
# 5. EVALUASI LENGKAP: TRAIN, TEST, DAN FULL DATASET
# -----------------------------------------------------

def print_evaluation(y_true, y_pred, dataset_name):
    """Helper function untuk print evaluasi."""
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
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


# 5a. Evaluasi di TRAINING SET dengan Leave-One-Out (untuk cek overfitting)
print("\n" + "="*70)
print("EVALUASI MODEL DENGAN HYPERPARAMETER TERBAIK")
print("="*70)

print("Menghitung prediksi training set dengan Leave-One-Out (lambat)...")
# Gunakan use_loo=True atau biarkan auto-detect (X_test == X_train)
y_pred_train = knn_predict(
    X_train_prep, y_train, X_train_prep,  # X_test = X_train akan auto-detect LOO
    k=best_params['k'], 
    metric=best_params['metric'], 
    weights=best_params['weights'],
    use_loo=True  # Explicit LOO mode
)
print_evaluation(y_train.values, y_pred_train, "TRAINING SET (Leave-One-Out)")

# 5b. Evaluasi di TEST SET (held-out data)
y_pred_test = knn_predict(
    X_train_prep, y_train, X_test_prep, 
    k=best_params['k'], 
    metric=best_params['metric'], 
    weights=best_params['weights']
)
print_evaluation(y_test.values, y_pred_test, "TEST SET")

# 5c. Evaluasi di FULL DATASET (keseluruhan data awal)
# Preprocess full dataset
X_full = df.drop("Potability", axis=1)
y_full = df["Potability"]

# Tidak perlu imputasi karena missing values sudah di-drop
X_full_scaled = (X_full - X_train.mean()) / X_train.std()  # Gunakan mean/std dari train

y_pred_full = knn_predict(
    X_train_prep, y_train, X_full_scaled,
    k=best_params['k'], 
    metric=best_params['metric'], 
    weights=best_params['weights']
)
print_evaluation(y_full.values, y_pred_full, "FULL DATASET (SEMUA DATA AWAL)")

# Analisis Overfitting
print("\n" + "="*70)
print("ANALISIS OVERFITTING")
print("="*70)
train_acc = accuracy(y_train.values, y_pred_train)
test_acc = accuracy(y_test.values, y_pred_test)
full_acc = accuracy(y_full.values, y_pred_full)

print(f"Akurasi Training: {train_acc:.4f}")
print(f"Akurasi Test:     {test_acc:.4f}")
print(f"Akurasi Full:     {full_acc:.4f}")
print(f"\nGap Train-Test:   {train_acc - test_acc:.4f}")

if train_acc - test_acc > 0.15:
    print("! OVERFITTING PARAH (gap > 15%)")
    print("    Solusi: 1) Perbesar k, 2) Kurangi fitur, 3) Tambah data training")
elif train_acc - test_acc > 0.05:
    print("! Model OVERFITTING (gap 5-15%)")
    print("    Perlu tuning: coba k lebih besar atau feature selection")
elif abs(train_acc - test_acc) < 0.02:
    print("✓ Model BAIK (gap < 2%, generalisasi sangat bagus)")
else:
    print("✓ Model CUKUP BAIK (gap 2-5%, wajar untuk KNN)")

print("\n" + "="*70)
print("ANALISIS CLASS IMBALANCE & REKOMENDASI")
print("="*70)
