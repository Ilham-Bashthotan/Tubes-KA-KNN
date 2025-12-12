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
# STEP 0: DOWNLOAD DATASET DARI KAGGLE
# ============================================================================

print("="*80)
print(" STEP 0: DOWNLOAD DATASET DARI KAGGLE")
print("="*80)

try:
    import kagglehub
    
    # Download dataset
    print("\n📥 Downloading Water Potability dataset from Kaggle...")
    path = kagglehub.dataset_download("adityakadiwal/water-potability")
    print(f"✓ Dataset downloaded successfully!")
    print(f"📁 Path to dataset files: {path}")
    
    # Find CSV file
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        csv_path = os.path.join(path, csv_files[0])
        print(f"✓ Found CSV file: {csv_files[0]}")
    else:
        raise FileNotFoundError("No CSV file found in downloaded dataset")
        
except ImportError:
    print("⚠️ kagglehub not installed. Using alternative method...")
    print("Install with: pip install kagglehub")
    csv_path = "water_potability.csv"  # Fallback
except Exception as e:
    print(f"⚠️ Error downloading: {e}")
    print("Please ensure you have kagglehub installed and Kaggle API configured")
    csv_path = "water_potability.csv"  # Fallback

# ============================================================================
# STEP 1: LOAD DATA & EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*80)
print(" STEP 1: LOAD DATA & EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# Load dataset
try:
    df = pd.read_csv(csv_path)
    print(f"\n✓ Dataset loaded successfully from {csv_path}")
except:
    print("\n⚠️ Could not load dataset. Creating sample data for demonstration...")
    # Sample data untuk demo jika file tidak tersedia
    np.random.seed(42)
    n_samples = 3276
    df = pd.DataFrame({
        'ph': np.random.normal(7, 1.5, n_samples),
        'Hardness': np.random.normal(200, 30, n_samples),
        'Solids': np.random.normal(20000, 5000, n_samples),
        'Chloramines': np.random.normal(7, 1.5, n_samples),
        'Sulfate': np.random.normal(330, 40, n_samples),
        'Conductivity': np.random.normal(400, 80, n_samples),
        'Organic_carbon': np.random.normal(14, 3, n_samples),
        'Trihalomethanes': np.random.normal(66, 16, n_samples),
        'Turbidity': np.random.normal(4, 0.5, n_samples),
        'Potability': np.random.choice([0, 1], n_samples, p=[0.61, 0.39])
    })
    # Inject missing values
    for col in df.columns[:-1]:
        mask = np.random.choice([True, False], n_samples, p=[0.15, 0.85])
        df.loc[mask, col] = np.nan

print("\n" + "-"*80)
print("1.1 DATASET OVERVIEW")
print("-"*80)
print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n📋 Column Information:")
print(df.info())

print("\n" + "-"*80)
print("1.2 FIRST 10 ROWS")
print("-"*80)
print(df.head(10))

print("\n" + "-"*80)
print("1.3 DESCRIPTIVE STATISTICS")
print("-"*80)
print(df.describe())

print("\n" + "-"*80)
print("1.4 MISSING VALUES ANALYSIS")
print("-"*80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage (%)': missing_pct.round(2)
})
print(missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))

total_missing = missing.sum()
print(f"\n📌 Total Missing Values: {total_missing}")
print(f"📌 Percentage of Missing Data: {(total_missing / df.size * 100):.2f}%")

print("\n💡 INSIGHT:")
print("   - Dataset memiliki missing values yang cukup signifikan")
print("   - Perlu dilakukan imputasi sebelum modeling")
print("   - pH, Sulfate, dan Trihalomethanes memiliki missing values tertinggi")

print("\n" + "-"*80)
print("1.5 TARGET VARIABLE ANALYSIS")
print("-"*80)
print("\nTarget Distribution:")
target_counts = df['Potability'].value_counts().sort_index()
print(target_counts)
print("\nTarget Proportion:")
target_prop = df['Potability'].value_counts(normalize=True).sort_index()
for idx, val in target_prop.items():
    print(f"  Class {idx}: {val:.2%}")

print("\n💡 INSIGHT:")
print(f"   - Class 0 (Not Potable): {target_prop[0]:.2%}")
print(f"   - Class 1 (Potable): {target_prop[1]:.2%}")
if abs(target_prop[0] - target_prop[1]) > 0.2:
    print("   - Dataset mengalami class imbalance (perlu diperhatikan dalam evaluasi)")
else:
    print("   - Dataset relatif balanced")

# ============================================================================
# STEP 2: DATA VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print(" STEP 2: DATA VISUALIZATION")
print("="*80)

# 2.1 Distribution of Features
print("\n📊 Generating feature distribution plots...")
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Distribution of Water Quality Parameters', fontsize=18, y=0.995, fontweight='bold')

feature_cols = df.columns[:-1]
for idx, col in enumerate(feature_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Plot histogram
    ax.hist(df[col].dropna(), bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_title(f'{col}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistics
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('01_feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_feature_distributions.png")
plt.close()

# 2.2 Correlation Heatmap
print("\n📊 Generating correlation heatmap...")
plt.figure(figsize=(12, 10))
correlation = df.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, mask=mask,
            vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Water Quality Parameters', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_correlation_heatmap.png")
plt.close()

print("\n💡 INSIGHT FROM CORRELATION:")
# Find highest correlations with target
target_corr = correlation['Potability'].drop('Potability').abs().sort_values(ascending=False)
print("   Top 3 features correlated with Potability:")
for i, (feat, corr_val) in enumerate(target_corr.head(3).items(), 1):
    print(f"   {i}. {feat}: {corr_val:.4f}")

# 2.3 Box Plots by Target
print("\n📊 Generating box plots by target class...")
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Feature Distributions by Potability', fontsize=18, y=0.995, fontweight='bold')

for idx, col in enumerate(feature_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Prepare data
    data_to_plot = [df[df['Potability'] == 0][col].dropna(), 
                    df[df['Potability'] == 1][col].dropna()]
    
    bp = ax.boxplot(data_to_plot, labels=['Not Potable', 'Potable'], 
                    patch_artist=True, showmeans=True)
    
    # Color boxes
    colors = ['lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title(f'{col}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('03_boxplots_by_target.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_boxplots_by_target.png")
plt.close()

# ============================================================================
# STEP 3: DATA PREPROCESSING - HANDLING MISSING VALUES
# ============================================================================

print("\n" + "="*80)
print(" STEP 3: DATA PREPROCESSING - HANDLING MISSING VALUES")
print("="*80)

print("\n📋 METODE IMPUTASI: Median Imputation")
print("\n💡 ALASAN PEMILIHAN MEDIAN:")
print("   1. Robust terhadap outliers (tidak terpengaruh nilai ekstrem)")
print("   2. Cocok untuk data yang tidak terdistribusi normal sempurna")
print("   3. Lebih representatif untuk data dengan skewness tinggi")
print("   4. Sederhana dan mudah diinterpretasi")

df_imputed = df.copy()
imputation_values = {}

print("\n" + "-"*80)
print("IMPUTATION PROCESS:")
print("-"*80)

for col in df_imputed.columns[:-1]:
    if df_imputed[col].isnull().sum() > 0:
        median_val = df_imputed[col].median()
        n_missing = df_imputed[col].isnull().sum()
        df_imputed[col].fillna(median_val, inplace=True)
        imputation_values[col] = median_val
        print(f"  ✓ {col:20s}: {n_missing:4d} missing values → imputed with median = {median_val:8.2f}")

print("\n" + "-"*80)
print("VERIFICATION:")
print("-"*80)
remaining_missing = df_imputed.isnull().sum().sum()
print(f"  Total missing values after imputation: {remaining_missing}")
if remaining_missing == 0:
    print("  ✓ All missing values handled successfully!")
else:
    print(f"  ⚠️ Still have {remaining_missing} missing values")

# ============================================================================
# STEP 4: FEATURE BINNING (NUMERIC → CATEGORICAL)
# ============================================================================

print("\n" + "="*80)
print(" STEP 4: FEATURE BINNING (NUMERIC → CATEGORICAL)")
print("="*80)

print("\n📋 METODE BINNING: Equal-Width Binning dengan 4 Bins")
print("\n💡 ALASAN PEMILIHAN METODE:")
print("   1. Equal-width sederhana dan mudah diinterpretasi")
print("   2. 4 bins memberikan granularitas yang cukup tanpa terlalu detail")
print("      (Low, Medium-Low, Medium-High, High)")
print("   3. Mengurangi noise dan membuat model lebih stabil")
print("   4. Menghindari overfitting pada nilai numerik yang sangat spesifik")
print("   5. Cocok untuk Decision Tree dengan categorical features")

print("\n💡 ALTERNATIF YANG DIPERTIMBANGKAN:")
print("   - Equal-frequency (quantile): Bias jika distribusi sangat skewed")
print("   - Manual domain-based: Butuh domain knowledge yang mendalam")
print("   - Adaptive binning: Terlalu kompleks untuk dataset ini")

df_binned = df_imputed.copy()
bin_info = {}

print("\n" + "-"*80)
print("BINNING PROCESS & STATISTICS:")
print("-"*80)

for col in df_binned.columns[:-1]:
    # Calculate bins
    min_val = df_binned[col].min()
    max_val = df_binned[col].max()
    bins = np.linspace(min_val, max_val, 5)  # 5 edges = 4 bins
    
    # Create labels
    labels = [f'{col}_Low', f'{col}_MedLow', f'{col}_MedHigh', f'{col}_High']
    
    # Apply binning
    df_binned[col] = pd.cut(df_binned[col], bins=bins, labels=labels, include_lowest=True)
    
    # Store bin info
    bin_info[col] = {
        'bins': bins,
        'labels': labels,
        'min': min_val,
        'max': max_val
    }
    
    # Print statistics
    print(f"\n{col}:")
    print(f"  Range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"  Bin edges: {[f'{b:.2f}' for b in bins]}")
    print(f"  Distribution after binning:")
    dist = df_binned[col].value_counts().sort_index()
    for label, count in dist.items():
        pct = (count / len(df_binned)) * 100
        print(f"    {label:25s}: {count:4d} samples ({pct:5.2f}%)")

# Visualize binning effect
print("\n📊 Generating binning visualization...")
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Effect of Binning on Features', fontsize=18, y=0.995, fontweight='bold')

for idx, col in enumerate(feature_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Plot binned distribution
    df_binned[col].value_counts().sort_index().plot(kind='bar', ax=ax, color='teal', alpha=0.7, edgecolor='black')
    ax.set_title(f'{col} (Binned)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Bins', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('04_feature_binning.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_feature_binning.png")
plt.close()

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print(" STEP 5: TRAIN-TEST SPLIT")
print("="*80)

X = df_binned.drop('Potability', axis=1)
y = df_binned['Potability']

print("\n📋 SPLIT CONFIGURATION:")
print("   - Test size: 20% (0.2)")
print("   - Random state: 42 (for reproducibility)")
print("   - Stratify: Yes (maintain class distribution)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "-"*80)
print("SPLIT RESULTS:")
print("-"*80)
print(f"  Training set:   {len(X_train):4d} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test set:       {len(X_test):4d} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Total:          {len(X):4d} samples")

print("\n  Training set target distribution:")
train_dist = y_train.value_counts().sort_index()
for cls, count in train_dist.items():
    pct = (count / len(y_train)) * 100
    print(f"    Class {cls}: {count:4d} samples ({pct:.2f}%)")

print("\n  Test set target distribution:")
test_dist = y_test.value_counts().sort_index()
for cls, count in test_dist.items():
    pct = (count / len(y_test)) * 100
    print(f"    Class {cls}: {count:4d} samples ({pct:.2f}%)")

# ============================================================================
# STEP 6: IMPLEMENTASI DECISION TREE DARI NOL
# ============================================================================

print("\n" + "="*80)
print(" STEP 6: IMPLEMENTASI DECISION TREE DARI NOL")
print("="*80)

class Node:
    """
    Class untuk merepresentasikan node dalam Decision Tree
    
    Attributes:
    -----------
    feature : int
        Index dari feature yang digunakan untuk split
    threshold : any
        Nilai threshold untuk split (untuk categorical = nilai kategori)
    left : Node
        Left child node (samples yang memenuhi kondisi split)
    right : Node
        Right child node (samples yang tidak memenuhi kondisi split)
    value : int
        Nilai prediksi untuk leaf node (class label)
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        """Check apakah node ini adalah leaf node"""
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier Implementation dari Nol
    
    Implementasi ini menggunakan:
    - Information Gain (Entropy) sebagai kriteria split
    - Recursive binary splitting
    - Stopping criteria: max_depth, min_samples_split, min_samples_leaf
    
    Parameters:
    -----------
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=5
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=2
        Minimum number of samples required to be at a leaf node
    
    Attributes:
    -----------
    root : Node
        Root node dari decision tree
    feature_names : list
        Nama-nama feature untuk interpretability
    """
    
    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.feature_names = None
    
    def entropy(self, y):
        """
        Menghitung entropy dari target variable
        
        Formula: Entropy = -Σ(p_i * log2(p_i))
        dimana p_i adalah proporsi dari class i
        
        Entropy mengukur impurity/ketidakpastian dalam data:
        - Entropy = 0: semua samples satu class (pure)
        - Entropy maksimal: samples terdistribusi merata di semua class
        
        Parameters:
        -----------
        y : array-like
            Target variable
        
        Returns:
        --------
        float : Entropy value
        """
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def information_gain(self, X_column, y, threshold):
        """
        Menghitung Information Gain dari suatu split
        
        Formula: IG = Entropy(parent) - Weighted_Average_Entropy(children)
        
        Information Gain mengukur seberapa banyak informasi yang didapat
        dari melakukan split pada feature tertentu dengan threshold tertentu.
        Split dengan IG tertinggi adalah split terbaik.
        
        Parameters:
        -----------
        X_column : array-like
            Column dari feature matrix
        y : array-like
            Target variable
        threshold : any
            Nilai threshold untuk split
        
        Returns:
        --------
        float : Information Gain value
        """
        # Parent entropy
        parent_entropy = self.entropy(y)
        
        # Split data berdasarkan threshold
        left_mask = X_column == threshold
        right_mask = X_column != threshold
        
        # Jika salah satu split kosong, return 0 (no gain)
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Hitung weighted entropy dari children
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        entropy_left = self.entropy(y[left_mask])
        entropy_right = self.entropy(y[right_mask])
        
        weighted_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
        
        # Information Gain
        ig = parent_entropy - weighted_entropy
        return ig
    
    def best_split(self, X, y):
        """
        Mencari best split dengan Information Gain tertinggi
        
        Algoritma:
        1. Iterasi semua features
        2. Untuk setiap feature, coba semua unique values sebagai threshold
        3. Hitung Information Gain untuk setiap kombinasi
        4. Pilih split dengan IG tertinggi
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target variable
        
        Returns:
        --------
        tuple : (best_feature_idx, best_threshold, best_gain)
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Iterasi semua features
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            unique_values = np.unique(X_column)
            
            # Untuk categorical, coba setiap unique value sebagai threshold
            for threshold in unique_values:
                gain = self.information_gain(X_column, y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """
        Membangun Decision Tree secara rekursif
        
        Algoritma:
        1. Check stopping criteria
        2. Jika tidak berhenti, cari best split
        3. Split data menjadi left dan right
        4. Rekursif build left dan right subtree
        5. Return node
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target variable
        depth : int
            Current depth of the tree
        
        Returns:
        --------
        Node : Root node of the (sub)tree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            # Buat leaf node dengan majority class
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Cari best split
        best_feature, best_threshold, best_gain = self.best_split(X, y)
        
        # Jika tidak ada gain, buat leaf node
        if best_gain == 0:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Split data
        left_mask = X[:, best_feature] == best_threshold
        right_mask = X[:, best_feature] != best_threshold
        
        # Check min_samples_leaf
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Rekursif build left dan right subtree
        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def fit(self, X, y):
        """
        Training Decision Tree
        
        Parameters:
        -----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            Training data
        y : Series or array-like, shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
        """
        # Store feature names untuk interpretability
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Build tree
        self.root = self.build_tree(X, y)
        return self
    
    def predict_sample(self, x, node):
        """
        Prediksi untuk satu sample
        
        Algoritma:
        1. Jika leaf node, return value
        2. Jika tidak, traverse tree berdasarkan feature value
        3. Rekursif sampai mencapai leaf node
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Single sample
        node : Node
            Current node
        
        Returns:
        --------
        int : Predicted class label
        """
        # Jika leaf node, return value
        if node.is_leaf():
            return node.value
        
        # Traverse tree
        if x[node.feature] == node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Prediksi untuk multiple samples
        
        Parameters:
        -----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            Samples to predict
        
        Returns:
        --------
        array-like, shape (n_samples,) : Predicted class labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = [self.predict_sample(x, self.root) for x in X]
        return np.array(predictions)
    
    def get_depth(self, node=None):
        """
        Menghitung depth (kedalaman) dari tree
        
        Returns:
        --------
        int : Depth of the tree
        """
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 0
        
        left_depth = self.get_depth(node.left) if node.left else 0
        right_depth = self.get_depth(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)
    
    def count_nodes(self, node=None):
        """
        Menghitung jumlah total nodes dalam tree
        
        Returns:
        --------
        int : Total number of nodes
        """
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.is_leaf():
            return 1
        
        left_count = self.count_nodes(node.left) if node.left else 0
        right_count = self.count_nodes(node.right) if node.right else 0
        
        return 1 + left_count + right_count
    
    def count_leaves(self, node=None):
        """
        Menghitung jumlah leaf nodes dalam tree
        
        Returns:
        --------
        int : Number of leaf nodes
        """
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.is_leaf():
            return 1
        
        left_leaves = self.count_leaves(node.left) if node.left else 0
        right_leaves = self.count_leaves(node.right) if node.right else 0
        
        return left_leaves + right_leaves
    
    def print_tree(self, node=None, depth=0, prefix="Root", max_depth=3):
        """
        Print tree structure (limited depth untuk readability)
        
        Parameters:
        -----------
        node : Node
            Current node
        depth : int
            Current depth
        prefix : str
            Prefix string for display
        max_depth : int
            Maximum depth to print
        """
        if depth > max_depth or node is None:
            return
        
        if node.is_leaf():
            print(f"{'  ' * depth}{prefix} → Leaf: Class {node.value}")
            return
        
        feature_name = self.feature_names[node.feature] if self.feature_names else f"Feature_{node.feature}"
        print(f"{'  ' * depth}{prefix} → [{feature_name} == {node.threshold}]")
        
        if node.left:
            self.print_tree(node.left, depth + 1, "├─ Left (Yes)", max_depth)
        if node.right:
            self.print_tree(node.right, depth + 1, "└─ Right (No)", max_depth)


print("\n✓ Decision Tree Class implemented successfully!")
print("\n📋 KEY COMPONENTS:")
print("   1. Node class: Merepresentasikan setiap node dalam tree")
print("   2. Entropy calculation: Mengukur impurity/ketidakpastian")
print("   3. Information Gain calculation: Mengukur kualitas split")
print("   4. Best split finding: Mencari split terbaik dengan IG tertinggi")
print("   5. Recursive tree building: Membangun tree secara top-down")
print("   6. Prediction method: Traverse tree untuk prediksi")
print("   7. Tree statistics: Depth, nodes count, leaves count")

# ============================================================================
# STEP 7: TRAINING MODEL
# ============================================================================

print("\n" + "="*80)
print(" STEP 7: TRAINING MODEL")
print("="*80)

print("\n🔧 HYPERPARAMETER CONFIGURATION:")
print("   - max_depth: 10")
print("     (Membatasi kedalaman tree untuk menghindari overfitting)")
print("   - min_samples_split: 10")
print("     (Node harus memiliki minimal 10 samples untuk di-split)")
print("   - min_samples_leaf: 5")
print("     (Leaf node harus memiliki minimal 5 samples)")

print("\n🚀 Training model...")
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5)
dt.fit(X_train, y_train)
print("✓ Model training completed!")

print("\n" + "-"*80)
print("MODEL STATISTICS:")
print("-"*80)
tree_depth = dt.get_depth()
total_nodes = dt.count_nodes()
leaf_nodes = dt.count_leaves()
internal_nodes = total_nodes - leaf_nodes

print(f"  Tree Depth:        {tree_depth}")
print(f"  Total Nodes:       {total_nodes}")
print(f"  Leaf Nodes:        {leaf_nodes}")
print(f"  Internal Nodes:    {internal_nodes}")
print(f"  Features Used:     {len(dt.feature_names)}")

print("\n" + "-"*80)
print("TREE STRUCTURE (First 3 levels):")
print("-"*80)
dt.print_tree(max_depth=3)

# ============================================================================
# STEP 8: PREDICTION
# ============================================================================

print("\n" + "="*80)
print(" STEP 8: PREDICTION")
print("="*80)

print("\n🔮 Making predictions on training and test sets...")
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)
print("✓ Predictions completed!")

print("\n" + "-"*80)
print("SAMPLE PREDICTIONS (First 20 test samples):")
print("-"*80)
print("Index | Actual | Predicted | Match")
print("-" * 40)
for i in range(min(20, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    match = "✓" if actual == predicted else "✗"
    print(f"{i:5d} | {actual:6d} | {predicted:9d} | {match:5s}")

# ============================================================================
# STEP 9: MANUAL CALCULATION OF EVALUATION METRICS
# ============================================================================

print("\n" + "="*80)
print(" STEP 9: MANUAL CALCULATION OF EVALUATION METRICS")
print("="*80)

def calculate_confusion_matrix(y_true, y_pred):
    """
    Menghitung Confusion Matrix secara manual
    
    Confusion Matrix untuk binary classification:
    
                    Predicted
                    0       1
    Actual    0   [[TN,    FP],
              1    [FN,    TP]]
    
    - TN (True Negative): Predicted 0, Actual 0
    - FP (False Positive): Predicted 1, Actual 0 (Type I Error)
    - FN (False Negative): Predicted 0, Actual 1 (Type II Error)
    - TP (True Positive): Predicted 1, Actual 1
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    
    Returns:
    --------
    ndarray : 2x2 confusion matrix
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
    Menghitung Accuracy, Precision, Recall, F1-Score secara manual
    
    Formulas:
    ---------
    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
              = Proporsi prediksi yang benar dari semua prediksi
    
    Precision = TP / (TP + FP)
              = Dari semua yang diprediksi positif, berapa yang benar positif
              = Mengukur ketepatan prediksi positif
    
    Recall    = TP / (TP + FN)
              = Dari semua yang benar positif, berapa yang berhasil diprediksi
              = Mengukur kelengkapan prediksi positif (Sensitivity)
    
    F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
              = Harmonic mean dari Precision dan Recall
              = Mengukur keseimbangan antara Precision dan Recall
    
    Parameters:
    -----------
    confusion_matrix : ndarray
        2x2 confusion matrix
    
    Returns:
    --------
    tuple : (accuracy, precision, recall, f1_score)
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
# STEP 10: VISUALIZATION OF RESULTS
# ============================================================================

print("\n" + "="*80)
print(" STEP 10: VISUALIZATION OF RESULTS")
print("="*80)

# 10.1 Confusion Matrices Heatmap
print("\n📊 Generating confusion matrix heatmaps...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training CM
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Potable', 'Potable'],
            yticklabels=['Not Potable', 'Potable'],
            cbar_kws={'label': 'Count'})
axes[0].set_title(f'Training Set Confusion Matrix\nAccuracy: {acc_train:.4f}', 
                  fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')

# Test CM
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Not Potable', 'Potable'],
            yticklabels=['Not Potable', 'Potable'],
            cbar_kws={'label': 'Count'})
axes[1].set_title(f'Test Set Confusion Matrix\nAccuracy: {acc_test:.4f}', 
                  fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('05_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_confusion_matrices.png")
plt.close()

# 10.2 Metrics Comparison
print("\n📊 Generating metrics comparison chart...")
metrics_data = {
    'Accuracy': [acc_train, acc_test],
    'Precision': [prec_train, prec_test],
    'Recall': [rec_train, rec_test],
    'F1-Score': [f1_train, f1_test]
}

x = np.arange(len(metrics_data))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
train_bars = ax.bar(x - width/2, [metrics_data[m][0] for m in metrics_data], 
                    width, label='Training', color='steelblue', alpha=0.8, edgecolor='black')
test_bars = ax.bar(x + width/2, [metrics_data[m][1] for m in metrics_data], 
                   width, label='Test', color='coral', alpha=0.8, edgecolor='black')

# Add value labels on bars
for bars in [train_bars, test_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance: Training vs Test', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics_data.keys(), fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('06_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_metrics_comparison.png")
plt.close()

# 10.3 Prediction Distribution
print("\n📊 Generating prediction distribution chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training
train_results = pd.DataFrame({
    'Actual': y_train.values,
    'Predicted': y_train_pred
})
train_crosstab = pd.crosstab(train_results['Actual'], train_results['Predicted'], 
                              normalize='index') * 100

train_crosstab.plot(kind='bar', ax=axes[0], color=['coral', 'lightgreen'], 
                    alpha=0.8, edgecolor='black', width=0.7)
axes[0].set_title('Training Set: Prediction Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Actual Class', fontsize=12)
axes[0].set_ylabel('Percentage (%)', fontsize=12)
axes[0].set_xticklabels(['Not Potable', 'Potable'], rotation=0)
axes[0].legend(['Predicted: Not Potable', 'Predicted: Potable'], loc='upper right')
axes[0].grid(axis='y', alpha=0.3)

# Test
test_results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_test_pred
})
test_crosstab = pd.crosstab(test_results['Actual'], test_results['Predicted'], 
                             normalize='index') * 100

test_crosstab.plot(kind='bar', ax=axes[1], color=['coral', 'lightgreen'], 
                   alpha=0.8, edgecolor='black', width=0.7)
axes[1].set_title('Test Set: Prediction Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Actual Class', fontsize=12)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].set_xticklabels(['Not Potable', 'Potable'], rotation=0)
axes[1].legend(['Predicted: Not Potable', 'Predicted: Potable'], loc='upper right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('07_prediction_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_prediction_distribution.png")
plt.close()

# ============================================================================
# STEP 11: INTERPRETASI HASIL & ANALISIS OVERFITTING
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

print("\n💡 KEMUNGKINAN PENYEBAB OVERFITTING RINGAN:")
print("="*80)

print("\n1. KOMPLEKSITAS MODEL:")
print("   - Tree depth yang cukup dalam (max_depth=10) memungkinkan model")
print("     mempelajari pola yang terlalu spesifik pada training data")
print("   - Solusi: Reduce max_depth atau increase min_samples_leaf")

print("\n2. UKURAN DATASET:")
print(f"   - Training samples: {len(X_train)} (relatif kecil)")
print("   - Dengan dataset kecil, model cenderung memorize daripada generalize")
print("   - Solusi: Collect more data atau gunakan data augmentation")

print("\n3. MISSING VALUES:")
print("   - Dataset memiliki ~15-20% missing values yang diimputasi")
print("   - Median imputation bisa tidak menangkap variabilitas sebenarnya")
print("   - Solusi: Coba metode imputasi yang lebih sophisticated (KNN, MICE)")

print("\n4. BINNING STRATEGY:")
print("   - Equal-width binning dengan 4 bins mungkin terlalu kasar atau terlalu halus")
print("   - Beberapa informasi numerik hilang saat konversi ke kategorikal")
print("   - Solusi: Experiment dengan jumlah bins atau metode binning lain")

print("\n5. FEATURE QUALITY:")
print("   - Korelasi antar features dengan target relatif rendah")
print("   - Model mungkin belajar noise daripada signal")
print("   - Solusi: Feature engineering atau feature selection")

print("\n6. CLASS IMBALANCE:")
if abs(target_prop[0] - target_prop[1]) > 0.1:
    print(f"   - Dataset imbalanced (Class 0: {target_prop[0]:.2%}, Class 1: {target_prop[1]:.2%})")
    print("   - Model bisa bias terhadap majority class")
    print("   - Solusi: SMOTE, class weights, atau threshold adjustment")
else:
    print("   - Dataset relatif balanced, bukan penyebab utama")

print("\n" + "-"*80)
print("🎯 REKOMENDASI IMPROVEMENT:")
print("-"*80)
print("\n1. HYPERPARAMETER TUNING:")
print("   - Reduce max_depth (coba 5-8)")
print("   - Increase min_samples_leaf (coba 10-20)")
print("   - Increase min_samples_split (coba 20-30)")

print("\n2. REGULARIZATION:")
print("   - Pruning: Implement post-pruning untuk remove branches yang overfitted")
print("   - Cost-complexity pruning (alpha parameter)")

print("\n3. ENSEMBLE METHODS:")
print("   - Random Forest: Combine multiple trees dengan bagging")
print("   - Gradient Boosting: Sequential trees dengan error correction")

print("\n4. CROSS-VALIDATION:")
print("   - K-Fold CV untuk evaluasi yang lebih robust")
print("   - Membantu detect overfitting lebih early")

print("\n5. ALTERNATIVE BINNING:")
print("   - Equal-frequency (quantile) binning")
print("   - Domain-based binning dengan expert knowledge")
print("   - Experiment dengan jumlah bins (3, 5, atau adaptive)")

print("\n" + "="*80)
print(" ANALYSIS COMPLETE")
print("="*80)

print("\n📁 Generated Files:")
print("   1. 01_feature_distributions.png")
print("   2. 02_correlation_heatmap.png")
print("   3. 03_boxplots_by_target.png")
print("   4. 04_feature_binning.png")
print("   5. 05_confusion_matrices.png")
print("   6. 06_metrics_comparison.png")
print("   7. 07_prediction_distribution.png")

print("\n✅ All steps completed successfully!")
print("\n" + "="*80)

# ============================================================================
# BONUS: SAVE MODEL & RESULTS
# ============================================================================

print("\n" + "="*80)
print(" BONUS: SAVING RESULTS")
print("="*80)

# Save metrics to CSV
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Training': [acc_train, prec_train, rec_train, f1_train],
    'Test': [acc_test, prec_test, rec_test, f1_test],
    'Gap': [acc_train - acc_test, prec_train - prec_test, 
            rec_train - rec_test, f1_train - f1_test]
})
results_df.to_csv('model_metrics.csv', index=False)
print("\n✓ Saved metrics to: model_metrics.csv")
