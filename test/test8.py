# =====================================================
# K-NEAREST NEIGHBORS (NO LOO VERSION)
# =====================================================
# Versi modifikasi:
# - Menghapus logika Leave-One-Out (LOO)
# - Training accuracy dihitung dengan metode standar
# =====================================================

import pandas as pd
import numpy as np

# =====================================================
# FUNGSI-FUNGSI
# =====================================================

def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Split data menjadi train dan test set dengan stratifikasi.
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
    """Standardisasi Z-score."""
    mean = X_train.mean()
    std = X_train.std()
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled


def calculate_distances_vectorized(X_test_point, X_train, metric='euclidean'):
    """Hitung jarak dari 1 test point ke semua training points."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((X_train - X_test_point) ** 2, axis=1))
    elif metric == 'manhattan':
        return np.sum(np.abs(X_train - X_test_point), axis=1)
    else:
        raise ValueError(f"Metric {metric} tidak didukung")


def knn_predict(X_train, y_train, X_test, k=5, metric='euclidean', weights='uniform'):
    """
    KNN Standar (Tanpa LOO).
    Saat memprediksi data training, jarak ke diri sendiri (0) akan dihitung.
    """
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    
    predictions = []
    
    # Loop untuk setiap data point di test set
    for x_test in X_test_np:
        # 1. Hitung jarak
        distances = calculate_distances_vectorized(x_test, X_train_np, metric)
        
        # 2. Cari K tetangga terdekat
        # Menggunakan argpartition (lebih cepat dari sort penuh)
        # Jika k >= jumlah data, ambil semua
        k_actual = min(k, len(distances))
        
        if k_actual < len(distances):
             nearest_indices = np.argpartition(distances, k_actual)[:k_actual]
        else:
             nearest_indices = np.arange(len(distances))
             
        nearest_labels = y_train_np[nearest_indices]
        nearest_distances = distances[nearest_indices]
        
        # 3. Voting
        if weights == 'uniform':
            unique, counts = np.unique(nearest_labels, return_counts=True)
            pred = unique[np.argmax(counts)]
            
        elif weights == 'distance':
            # Tambah epsilon (1e-10) untuk menghindari pembagian dengan 0 
            # (karena jarak ke diri sendiri adalah 0)
            weights_arr = 1 / (nearest_distances + 1e-10)
            
            unique_labels = np.unique(nearest_labels)
            weighted_votes = np.array(
                [
                    np.sum(weights_arr[nearest_labels == label]) 
                    for label in unique_labels
                ]
            )
            pred = unique_labels[np.argmax(weighted_votes)]
        else:
            raise ValueError(f"Weights {weights} tidak didukung")
        
        predictions.append(pred)
    
    return np.array(predictions)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        true_idx = np.where(classes == true)[0][0]
        pred_idx = np.where(classes == pred)[0][0]
        cm[true_idx, pred_idx] += 1
    return cm


def cross_validate(X, y, k, metric, weights, n_folds=5):
    """Cross-validation menggunakan fungsi knn_predict standar."""
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    fold_size = len(X) // n_folds
    scores = []
    
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(X)
        
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_fold_train = X.iloc[train_indices].reset_index(drop=True)
        y_fold_train = y.iloc[train_indices].reset_index(drop=True)
        X_fold_test = X.iloc[test_indices].reset_index(drop=True)
        y_fold_test = y.iloc[test_indices].reset_index(drop=True)
        
        X_fold_train, X_fold_test = standardize(X_fold_train, X_fold_test)
        
        y_pred = knn_predict(X_fold_train, y_fold_train, X_fold_test, k, metric, weights)
        scores.append(accuracy(y_fold_test.values, y_pred))
    
    return np.mean(scores), np.std(scores)


# =====================================================
# MAIN PROGRAM
# =====================================================

# 1. LOAD & CLEAN
df = pd.read_csv("data/water_potability.csv")
df = df.drop_duplicates().replace([np.inf, -np.inf], np.nan).dropna()

for col in df.columns:
    if col == "Potability": continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# 2. SPLIT
X = df.drop("Potability", axis=1)
y = df["Potability"]
X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

# 3. PREPROCESS
X_train_prep, X_test_prep = standardize(X_train, X_test)

# 4. GRID SEARCH
k_values = [5, 7, 9, 11, 15, 19, 25, 31, 41, 51]
metrics = ['euclidean', 'manhattan']
weights_options = ['uniform', 'distance']

print(f"\n=== Grid Search (CV 5-Fold) ===")
best_score = 0
best_params = {}

for k in k_values:
    for metric in metrics:
        for weights in weights_options:
            mean_score, std_score = cross_validate(X_train, y_train, k, metric, weights)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'k': k, 'metric': metric, 'weights': weights}
            # Optional: Print progress if needed

print(f"\n>>> BEST: {best_params}")
print(f"    CV Accuracy: {best_score:.4f}")


# =====================================================
# 5. EVALUASI DAN ANALISIS OVERFITTING (TANPA LOO)
# =====================================================

def print_evaluation(y_true, y_pred, dataset_name):
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {dataset_name} ===")
    print(f"Akurasi: {acc:.4f}")
    print(cm)

# best_params['weights'] = "uniform"

# 5a. Evaluasi di TRAINING SET (Standard Method)
print("\n" + "="*70)
print("EVALUASI MODEL DENGAN HYPERPARAMETER TERBAIK")
print("="*70)

print("Menghitung prediksi training set (Metode Standar, Non-LOO)...")
# Kita memasukkan data train sebagai data test untuk mengukur seberapa baik model menghafal data
y_pred_train = knn_predict(
    X_train_prep, y_train, X_train_prep, 
    k=best_params['k'], 
    metric=best_params['metric'], 
    weights=best_params['weights']
)
print_evaluation(y_train.values, y_pred_train, "TRAINING SET (Standard/Biased)")

# 5b. Evaluasi di TEST SET
y_pred_test = knn_predict(
    X_train_prep, y_train, X_test_prep, 
    k=best_params['k'], 
    metric=best_params['metric'], 
    weights=best_params['weights']
)
print_evaluation(y_test.values, y_pred_test, "TEST SET")

# 5c. Evaluasi Full Dataset
X_full = df.drop("Potability", axis=1)
y_full = df["Potability"]
X_full_scaled = (X_full - X_train.mean()) / X_train.std()

y_pred_full = knn_predict(
    X_train_prep, y_train, X_full_scaled,
    k=best_params['k'], 
    metric=best_params['metric'], 
    weights=best_params['weights']
)
print_evaluation(y_full.values, y_pred_full, "FULL DATASET")

# -----------------------------------------------------
# ANALISIS OVERFITTING
# -----------------------------------------------------
print("\n" + "="*70)
print("ANALISIS KINERJA (CV Score vs Test Score)")
print("="*70)

# Ambil score terbaik dari proses Grid Search tadi
cv_acc = best_score 
test_acc = accuracy(y_test.values, y_pred_test)

print(f"CV Accuracy (Validasi): {cv_acc:.4f}")
print(f"Test Accuracy (Hold-out): {test_acc:.4f}")
print(f"Gap (CV - Test):          {cv_acc - test_acc:.4f}")

print("\nKESIMPULAN:")
if best_params['weights'] == 'distance':
    print("Catatan: Training Accuracy diabaikan karena weights='distance' (pasti ~100%).")
    
    if cv_acc - test_acc > 0.10:
        print("! WASPADA: Terindikasi OVERFITTING.")
        print("  Skor saat validasi jauh lebih tinggi daripada test data asli.")
        print("  Solusi: Coba tingkatkan nilai k atau kurangi fitur.")
    elif test_acc < 0.60: # Threshold bisa disesuaikan domain
        print("! UNDERFITTING.")
        print("  Akurasi rendah baik di CV maupun Test.")
    else:
        print("✓ MODEL STABIL (Good Fit).")
        print("  Performa CV dan Test konsisten.")
        
else:
    # Jika uniform, masih bisa intip Training Acc sedikit, tapi CV vs Test tetap lebih valid
    train_acc = accuracy(y_train.values, y_pred_train)
    print(f"(Ref) Train Accuracy:     {train_acc:.4f}")
    if train_acc - test_acc > 0.15:
        print("! Indikasi Overfitting (Gap Train-Test besar).")
    else:
        print("✓ Model cukup konsisten.")