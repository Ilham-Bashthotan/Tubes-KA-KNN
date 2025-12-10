# =====================================================
# K-NEAREST NEIGHBORS UNTUK DATA WATER POTABILITY (UPDATED)
# =====================================================
# Perbaikan mencakup:
# 1. Tuning 'weights' (Uniform vs Distance) sesuai Slide Part 2 
# 2. Tuning 'metric' (Euclidean vs Manhattan) sesuai Slide Part 1 
# 3. Penggunaan GridSearchCV untuk mencoba semua kombinasi hyperparameter 
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------------------
# 1. LOAD DATASET & CLEANING
# -----------------------------------------------------
df = pd.read_csv("data/water_potability.csv")

# Bersihkan data
df = df.drop_duplicates()
df = df.replace([np.inf, -np.inf], np.nan)

# Cap outlier (Winsorizing)
for col in df.columns:
    if col == "Potability":
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

# -----------------------------------------------------
# 1a. FEATURE ENGINEERING - Berdasarkan Standar WHO/EPA
# -----------------------------------------------------
# Buat fitur biner: apakah setiap parameter dalam batas aman?
df['ph_safe'] = ((df['ph'] >= 6.5) & (df['ph'] <= 8.5)).astype(int)
df['Solids_safe'] = (df['Solids'] <= 1000).astype(int)  # TDS max 1000 mg/L
df['Chloramines_safe'] = (df['Chloramines'] <= 4).astype(int)  # max 4 ppm
df['Sulfate_safe'] = (df['Sulfate'] <= 1000).astype(int)  # max 1000 mg/L
df['Conductivity_safe'] = (df['Conductivity'] <= 400).astype(int)  # max 400 μS/cm
df['Organic_carbon_safe'] = (df['Organic_carbon'] < 4).astype(int)  # source water < 4 mg/L
df['Trihalomethanes_safe'] = (df['Trihalomethanes'] <= 80).astype(int)  # max 80 ppm
df['Turbidity_safe'] = (df['Turbidity'] <= 5).astype(int)  # max 5 NTU

# Hitung jumlah parameter yang aman (composite safety score)
safety_cols = ['ph_safe', 'Solids_safe', 'Chloramines_safe', 'Sulfate_safe', 
               'Conductivity_safe', 'Organic_carbon_safe', 'Trihalomethanes_safe', 'Turbidity_safe']
df['safety_score'] = df[safety_cols].sum(axis=1)

# -----------------------------------------------------
# 2. SPLIT DATA
# -----------------------------------------------------
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Stratify=y penting untuk klasifikasi seimbang
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------
# 3. PIPELINE & HYPERPARAMETER TUNING
# -----------------------------------------------------
# Pipeline dasar
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()), # Normalisasi penting untuk KNN
    ("knn", KNeighborsClassifier())
])

# Parameter Grid sesuai materi slide:
# - n_neighbors: Mencari K optimal [cite: 154]
# - weights: Membandingkan Uniform vs Weighted (Distance) 
# - metric: Membandingkan Euclidean vs Manhattan 
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21], # Bilangan ganjil 
    "knn__weights": ["uniform", "distance"], # Slide Part 2 Hal 8
    "knn__metric": ["euclidean", "manhattan"] # Slide Part 1 Hal 11
}

# Menggunakan GridSearchCV untuk mencoba SEMUA kombinasi (Slide Part 2 Hal 11)
print("\n=== Mencari Hyperparameter Terbaik (GridSearch) ===")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# -----------------------------------------------------
# 4. HASIL & EVALUASI
# -----------------------------------------------------
print(f"\nKombinasi Terbaik: {grid_search.best_params_}")
print(f"Akurasi Cross-Validation Terbaik: {grid_search.best_score_:.4f}")

# Evaluasi pada data test (Hold-out set)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=== HASIL DI TEST SET ===")
print(f"Akurasi Akhir: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
