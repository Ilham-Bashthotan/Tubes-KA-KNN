"""
Implementasi Gaussian Naive Bayes dari scratch untuk Water Potability Classification
"""

import numpy as np
import pandas as pd
import os

# ========================================
# FUNGSI-FUNGSI UTILITY
# ========================================

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

def cap_outliers(X_train, X_test, iqr_multiplier=1.5):
    """
    Outlier capping menggunakan metode IQR.
    I.S. X_train, X_test: fitur sebelum capping, iqr_multiplier: multiplier untuk IQR
    F.S : X_train_capped, X_test_capped
    """
    # Hitung IQR bounds dari TRAIN SET saja (mencegah data leakage)
    outlier_bounds = {}
    
    for col in X_train.columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        outlier_bounds[col] = (lower, upper)
    
    # Terapkan bounds ke train dan test set
    X_train_capped = X_train.copy()
    X_test_capped = X_test.copy()
    
    for col in X_train.columns:
        lower, upper = outlier_bounds[col]
        X_train_capped[col] = X_train_capped[col].clip(lower, upper)
        X_test_capped[col] = X_test_capped[col].clip(lower, upper)
    
    return X_train_capped, X_test_capped

# ========================================
# FUNGSI NAIVE BAYES
# ========================================

def gaussian_naive_bayes_fit(X, y, var_smoothing=1e-9):
    """
    Melatih model Gaussian Naive Bayes dengan menghitung mean, variance, dan prior.
    I.S. X: data training (fitur), y: label kelas, var_smoothing: smoothing parameter
    F.S : dictionary parameter model (classes, means, vars, priors)
    """
    model_params = {
        'classes': np.unique(y),
        'means': {},
        'vars': {},
        'priors': {},
        'var_smoothing': var_smoothing
    }
    
    n_samples = len(X)
    
    for c in model_params['classes']:
        X_c = X[y == c]
        
        # Simpan mean dan variance tiap fitur untuk kelas c
        model_params['means'][c] = X_c.mean(axis=0)
        # Tambahkan epsilon kecil untuk menghindari pembagian dengan nol
        model_params['vars'][c] = X_c.var(axis=0) + var_smoothing
        
        # Prior probability: P(class)
        model_params['priors'][c] = len(X_c) / n_samples
    
    return model_params

def gaussian_pdf(x, mean, var):
    """
    Menghitung Probability Density Function (PDF) untuk distribusi Gaussian.
    I.S. x: nilai input, mean: rata-rata, var: variansi
    F.S : nilai kepadatan probabilitas
    """
    numerator = np.exp(-(x - mean)**2 / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator

def gaussian_naive_bayes_predict_single(x, model_params):
    """
    Memprediksi label kelas untuk satu sampel menggunakan log-posterior.
    I.S. x: satu sampel fitur, model_params: parameter model hasil training
    F.S : kelas yang diprediksi
    """
    posteriors = {}
    
    for c in model_params['classes']:
        # Log prior: log(P(c))
        prior = np.log(model_params['priors'][c])
        
        # Log-likelihood: log(P(x|c)) = sum(log(P(x_i|c)))
        class_conditional = np.sum(
            np.log(gaussian_pdf(x, model_params['means'][c], model_params['vars'][c]))
        )
        
        # Log-posterior: log(P(c|x)) ∝ log(P(c)) + log(P(x|c))
        posteriors[c] = prior + class_conditional
    
    # Pilih kelas dengan log-posterior tertinggi
    return max(posteriors, key=posteriors.get)

def gaussian_naive_bayes_predict(X, model_params):
    """
    Memprediksi label kelas untuk beberapa sampel.
    I.S. X: data test (fitur), model_params: parameter model hasil training
    F.S : array prediksi label kelas
    """
    return np.array([gaussian_naive_bayes_predict_single(x, model_params) for x in X])

# ========================================
# FUNGSI EVALUASI
# ========================================

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

def print_evaluation(y_true, y_pred, dataset_name="Dataset"):
    """
    Print metrik evaluasi lengkap.
    I.S. y_true: label asli, y_pred: label prediksi, dataset_name: nama dataset
    F.S : print metrik evaluasi
    """
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"  EVALUASI MODEL GAUSSIAN NAIVE BAYES - {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Akurasi: {acc:.4f}")
    print(f"\nConfusion Matrix:")
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
    print(f"{'='*60}\n")

def cross_validate_nb(X, y, var_smoothing=1e-9, iqr_multiplier=1.5, use_standardize=True, use_capping=True, n_folds=5, random_state=42):
    """
    Cross-validation untuk Naive Bayes dengan parameter tertentu.
    I.S. X: fitur, y: target, var_smoothing: smoothing parameter, 
        iqr_multiplier: multiplier IQR, use_standardize: pakai standardisasi,
        use_capping: pakai outlier capping, n_folds: jumlah fold
    F.S : mean accuracy dari CV
    """
    np.random.seed(random_state)
    combined = X.copy()
    combined['_target_'] = y.values
    
    # Stratified sampling
    class_0 = combined[combined['_target_'] == 0]
    class_1 = combined[combined['_target_'] == 1]
    
    class_0_shuffled = class_0.sample(frac=1, random_state=random_state).reset_index(drop=True)
    class_1_shuffled = class_1.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    fold_scores = []
    
    for fold in range(n_folds):
        # Split each class
        fold_size_0 = len(class_0) // n_folds
        fold_size_1 = len(class_1) // n_folds
        
        start_0 = fold * fold_size_0
        end_0 = start_0 + fold_size_0 if fold < n_folds - 1 else len(class_0)
        start_1 = fold * fold_size_1
        end_1 = start_1 + fold_size_1 if fold < n_folds - 1 else len(class_1)
        
        val_0 = class_0_shuffled.iloc[start_0:end_0]
        val_1 = class_1_shuffled.iloc[start_1:end_1]
        
        train_0 = pd.concat([class_0_shuffled.iloc[:start_0], class_0_shuffled.iloc[end_0:]])
        train_1 = pd.concat([class_1_shuffled.iloc[:start_1], class_1_shuffled.iloc[end_1:]])
        
        train_combined = pd.concat([train_0, train_1], ignore_index=True)
        val_combined = pd.concat([val_0, val_1], ignore_index=True)
        
        train_combined = train_combined.sample(frac=1, random_state=random_state + fold).reset_index(drop=True)
        val_combined = val_combined.sample(frac=1, random_state=random_state + fold).reset_index(drop=True)
        
        X_train_fold = train_combined.drop('_target_', axis=1)
        y_train_fold = train_combined['_target_']
        X_val_fold = val_combined.drop('_target_', axis=1)
        y_val_fold = val_combined['_target_']
        
        # Preprocessing
        if use_capping:
            X_train_fold, X_val_fold = cap_outliers(X_train_fold, X_val_fold, iqr_multiplier)
        
        if use_standardize:
            X_train_fold, X_val_fold = standardize(X_train_fold, X_val_fold)
        
        # Train and predict
        X_train_np = X_train_fold.values
        y_train_np = y_train_fold.values
        X_val_np = X_val_fold.values
        y_val_np = y_val_fold.values
        
        model = gaussian_naive_bayes_fit(X_train_np, y_train_np, var_smoothing)
        y_pred = gaussian_naive_bayes_predict(X_val_np, model)
        
        fold_acc = accuracy(y_val_np, y_pred)
        fold_scores.append(fold_acc)
    
    return np.mean(fold_scores)

def grid_search_nb(X, y, param_grid, n_folds=5, random_state=42):
    """
    Grid search untuk menemukan kombinasi parameter terbaik.
    I.S. X: fitur, y: target, param_grid: dictionary parameter, n_folds: jumlah fold
    F.S : best_params, best_score, results
    """
    print("\n" + "="*60)
    print("  GRID SEARCH DENGAN CROSS-VALIDATION")
    print("="*60)
    
    var_smoothing_values = param_grid.get('var_smoothing', [1e-9])
    iqr_multipliers = param_grid.get('iqr_multiplier', [1.5])
    preprocessing_options = param_grid.get('preprocessing', ['both'])
    
    results = []
    best_score = 0
    best_params = None
    
    total_combinations = len(var_smoothing_values) * len(iqr_multipliers) * len(preprocessing_options)
    current = 0
    
    print(f"\nTotal kombinasi parameter: {total_combinations}")
    print(f"Menggunakan {n_folds}-Fold Cross-Validation\n")
    
    for var_smooth in var_smoothing_values:
        for iqr_mult in iqr_multipliers:
            for preproc in preprocessing_options:
                current += 1
                
                use_standardize = preproc in ['standardize_only', 'both']
                use_capping = preproc in ['cap_only', 'both']
                
                print(
                    f"[{current}/{total_combinations}] Testing: var_smoothing={var_smooth:.0e}, "
                    f"iqr_multiplier={iqr_mult}, preprocessing={preproc}"
                )
                
                cv_score = cross_validate_nb(
                    X, y, 
                    var_smoothing=var_smooth,
                    iqr_multiplier=iqr_mult,
                    use_standardize=use_standardize,
                    use_capping=use_capping,
                    n_folds=n_folds,
                    random_state=random_state
                )
                
                print(f"  → CV Accuracy: {cv_score:.4f}")
                
                result = {
                    'var_smoothing': var_smooth,
                    'iqr_multiplier': iqr_mult,
                    'preprocessing': preproc,
                    'cv_accuracy': cv_score
                }
                results.append(result)
                
                if cv_score > best_score:
                    best_score = cv_score
                    best_params = {
                        'var_smoothing': var_smooth,
                        'iqr_multiplier': iqr_mult,
                        'preprocessing': preproc
                    }
    
    print("\n" + "="*60)
    print("  HASIL GRID SEARCH")
    print("="*60)
    print(f"\nBest Parameters:")
    print(f"  - var_smoothing: {best_params['var_smoothing']:.0e}")
    print(f"  - iqr_multiplier: {best_params['iqr_multiplier']}")
    print(f"  - preprocessing: {best_params['preprocessing']}")
    print(f"\nBest CV Accuracy: {best_score:.4f}")
    print("="*60)
    
    return best_params, best_score, results

# ========================================
# MAIN PROGRAM
# ========================================

if __name__ == "__main__":
    # 1. Load Dataset
    print("\n" + "="*60)
    print("  LOAD DATASET")
    print("="*60)
    
    # Dapatkan path absolut ke file data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "water_potability.csv")
    
    df = pd.read_csv(data_path)
    print(f"Data awal: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # 2. Data Cleaning
    print("\n" + "="*60)
    print("  DATA CLEANING")
    print("="*60)
    
    # Hapus missing values
    df = df.dropna()
    print(f"Setelah drop missing: {df.shape}")
    
    # Hapus duplikat
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"Data final setelah cleaning: {df.shape}")
    
    # 3. Split Data
    print("\n" + "="*60)
    print("  SPLIT DATA (STRATIFIED)")
    print("="*60)
    
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Total Data: {len(df)} baris")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"\nProporsi kelas di y_train: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Proporsi kelas di y_test: {y_test.value_counts(normalize=True).round(3).to_dict()}")
    
    # 4. Grid Search untuk Parameter Tuning
    param_grid = {
        'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7],
        'iqr_multiplier': [1.0, 1.5, 2.0],
        'preprocessing': ['both', 'standardize_only', 'cap_only', 'none']
    }
    
    best_params, best_cv_score, grid_results = grid_search_nb(
        X, y, 
        param_grid=param_grid,
        n_folds=5,
        random_state=42
    )
    
    # 5. Preprocessing dengan Parameter Terbaik
    print("\n" + "="*60)
    print("  PREPROCESSING DENGAN PARAMETER TERBAIK")
    print("="*60)
    
    use_standardize = best_params['preprocessing'] in ['standardize_only', 'both']
    use_capping = best_params['preprocessing'] in ['cap_only', 'both']
    
    X_train_proc = X_train.copy()
    X_test_proc = X_test.copy()
    
    if use_capping:
        print(f"Melakukan outlier capping (IQR multiplier={best_params['iqr_multiplier']})...")
        X_train_proc, X_test_proc = cap_outliers(X_train_proc, X_test_proc, best_params['iqr_multiplier'])
        print("✓ Outlier capping selesai")
    
    if use_standardize:
        print("Melakukan standardisasi Z-score...")
        X_train_proc, X_test_proc = standardize(X_train_proc, X_test_proc)
        print("✓ Standardisasi selesai")
    
    if not use_capping and not use_standardize:
        print("Tidak menggunakan preprocessing (raw data)")
    
    # 6. Training Model dengan Parameter Terbaik
    print("\n" + "="*60)
    print("  TRAINING MODEL GAUSSIAN NAIVE BAYES (BEST PARAMS)")
    print("="*60)
    
    # Convert to numpy arrays
    X_train_np = X_train_proc.values
    y_train_np = y_train.values
    X_test_np = X_test_proc.values
    y_test_np = y_test.values
    
    print("Melatih model dengan parameter terbaik...")
    model = gaussian_naive_bayes_fit(X_train_np, y_train_np, var_smoothing=best_params['var_smoothing'])
    print(f"✓ Model berhasil dilatih")
    print(f"  - Jumlah kelas: {len(model['classes'])}")
    print(f"  - Kelas: {model['classes']}")
    print(f"  - Prior probabilities: {model['priors']}")
    print(f"  - Var smoothing: {best_params['var_smoothing']:.0e}")
    
    # 7. Prediction
    print("\n" + "="*60)
    print("  PREDICTION")
    print("="*60)
    
    print("Melakukan prediksi pada data test...")
    y_pred_test = gaussian_naive_bayes_predict(X_test_np, model)
    print("✓ Prediksi selesai")
    
    # 8. Evaluation
    print_evaluation(y_test_np, y_pred_test, "Test Set")
    
    # 9. Training Set Evaluation (untuk cek overfitting)
    print("Melakukan prediksi pada data training...")
    y_pred_train = gaussian_naive_bayes_predict(X_train_np, model)
    print_evaluation(y_train_np, y_pred_train, "Training Set")
    
    # 10. Analisis Overfitting
    train_acc = accuracy(y_train_np, y_pred_train)
    test_acc = accuracy(y_test_np, y_pred_test)
    gap = train_acc - test_acc
    
    print("\n" + "="*60)
    print("  ANALISIS OVERFITTING")
    print("="*60)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Gap:            {gap:.4f}")
    
    if gap > 0.10:
        print("\n!  Terindikasi OVERFITTING (gap > 0.10)")
        print("   Model terlalu menyesuaikan dengan training data")
    elif test_acc < 0.60:
        print("\n!  Terindikasi UNDERFITTING (test acc < 0.60)")
        print("   Model kurang mampu menangkap pola dalam data")
    else:
        print("\n✓ MODEL STABIL (Good Fit)")
        print("  Performa train dan test konsisten")
    print("="*60)
    
    print("\n" + "="*60)
    print("  SELESAI")
    print("="*60)
