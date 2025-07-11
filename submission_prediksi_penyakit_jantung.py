# -*- coding: utf-8 -*-
"""Submission Prediksi Penyakit Jantung.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XUHuO6B74yeWhLRLQJaulz9_Z6kFRr8n
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

"""# Data Understanding

## Data Loading
"""

# load the dataset
url = 'https://raw.githubusercontent.com/lailarizzah/Submission-Analisis-Prediksi/refs/heads/main/heart_disease_uci.csv'
heart = pd.read_csv(url)
heart.head()

"""## Exploratory Data Analysis"""

heart.info()

"""Dari output terlihat bahwa:

Terdapat 8 kolom dengan tipe object, yaitu: sex, dataset, cp, fbs, restecg, exang, slope, dan thal. Kolom ini merupakan categorical features (fitur non-numerik).
Terdapat 5 kolom numerik dengan tipe data float64 yaitu: trestbps, chol, thalch, oldpeak, dan ca. Ini merupakan fitur numerik yang merupakan hasil pengukuran secara fisik.
Terdapat 3 kolom numerik dengan tipe data int64, yaitu: id, age, dan num. Kolom 'num' merupakan target fitur.
"""

# Mengecek missing values
heart.isna().sum()

"""Ada perbedaan jumlah data pada kolom *trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca,* dan *thal*. Hal ini menunjukkan adanya missing values pada kolom tersebut."""

# Mengecek data duplikat
print("Jumlah duplikasi: ", heart.duplicated().sum())

# Mengecek inacurate value
heart.describe()

"""Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:

- Count  adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama.
- Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua atau median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

### Distribusi Variabel Numerik
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Ganti thalch ke thalach jika perlu
heart.rename(columns={'thalch': 'thalach'}, inplace=True)

# Pilih kolom numerik
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Plot
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(heart[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribusi {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.title("Histogram Plot")
plt.savefig("histogram_plot.png", bbox_inches='tight')  # simpan dulu
plt.show()

"""### Distribusi Variabel Kategorik"""

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

plt.figure(figsize=(15, 15))
for i, col in enumerate(categorical_cols):
    plt.subplot(4, 2, i+1)
    sns.countplot(data=heart, x=col, palette='pastel')
    plt.title(f'Distribusi {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("distribusi_variabel_kategorik.png", bbox_inches='tight')  # simpan dulu
plt.show()

"""### Distribusi Target"""

heart['target'] = heart['num'].apply(lambda x: 1 if x > 0 else 0)
sns.countplot(data=heart, x='target', palette='Set2')
plt.title('Distribusi Target: Ada/Tidak Ada Penyakit Jantung')
plt.xticks([0, 1], ['Tidak Ada', 'Ada'])
plt.savefig("distribusi_target.png", bbox_inches='tight')  # simpan dulu
plt.show()

"""Berdasarkan visualisasi

### Korelasi Antara Fitur dengan Target
"""

# Ubah target jadi biner
heart['target'] = heart['num'].apply(lambda x: 1 if x > 0 else 0)

import seaborn as sns
import matplotlib.pyplot as plt

# Ambil hanya kolom numerik
numerical_cols = heart.select_dtypes(include=['int64', 'float64']).columns

# Korelasi
correlation = heart[numerical_cols].corr()

# Korelasi terhadap target
target_corr = correlation['target'].drop('target').sort_values(ascending=False)

# Tampilkan hasil
print(target_corr)

# Korelasi antar semua fitur numerik
# corr_matrix = numerical_cols # This was the incorrect line
corr_matrix = correlation # Assign the actual correlation matrix

# Visualisasi heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title('Heatmap Korelasi Antar Fitur Numerik')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("korelasi_fitur_target.png", bbox_inches='tight')  # simpan dulu
plt.show()

"""# Data Preparation

## Mengatasi Missing Values

Berikut beberapa penanganan missing values pada data ini:
1. Fitur numerik seperti trestbps, chol, thalach, dan oldpeak diimputasi menggunakan nilai median.
2. Fitur kategorikal seperti fbs, restecg, exang, slope, dan thal diimputasi menggunakan nilai modus.
3. Fitur 'ca' dihapus karena memiliki terlalu banyak missing value.
"""

# Imputasi median (fitur numerik) menggunakan fillna pada DataFrame
# Menggunakan dictionary mapping column names to their fill values
heart.fillna({'trestbps': heart['trestbps'].median(),
              'chol': heart['chol'].median(),
              'thalach': heart['thalach'].median(),
              'oldpeak': heart['oldpeak'].median()}, inplace=True)

# Imputasi modus (fitur kategorikal)
for col in ['fbs', 'restecg', 'exang', 'slope', 'thal']:
    # Cek apakah kolom tersebut ada sebelum imputasi (optional, tapi bagus)
    if col in heart.columns:
        # Menggunakan penugasan langsung untuk menghindari warning
        heart[col] = heart[col].fillna(heart[col].mode()[0])

# Drop kolom ca karena missing value terlalu banyak
# Cek apakah kolom 'ca' ada sebelum di-drop untuk menghindari KeyError
if 'ca' in heart.columns:
    heart.drop(columns=['ca'], inplace=True)

# Mengecek missing values (output ini akan tetap muncul kecuali diakhiri dengan ; )
heart.isna().sum()

"""## Encoding Fitur Kategorik

Encoding dilakukan menggunakan one-hot encoding untuk fitur kategorikal seperti sex, cp, fbs, restecg, exang, slope, thal, dan dataset.

Hal ini penting karena sebagian besar algoritma machine learning membutuhkan data numerik sebagai input.
"""

# Add categorical encoding here
# Identify categorical columns to encode
categorical_cols_to_encode = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'dataset'] # Added 'dataset' as it is an object type

# Perform one-hot encoding
heart_encoded = pd.get_dummies(heart, columns=categorical_cols_to_encode, drop_first=True)

# Convert boolean columns created by get_dummies to integer type (1s and 0s)
bool_cols = heart_encoded.select_dtypes(include='bool').columns
heart_encoded[bool_cols] = heart_encoded[bool_cols].astype(int)

heart_encoded.head()

"""## Standarisasi

Standarisasi ini dilakukan untuk memastikan skala fitur seragam dan mempercepat proses pelatihan model. Fitur numerik seperti *age*, *trestbps*, *chol*, *thalach*, dan *oldpeak* dinormalisasi menggunakan StandardScaler.


"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
heart_encoded[numerical_cols] = scaler.fit_transform(heart_encoded[numerical_cols])
heart_encoded[numerical_cols].head()

"""## Data Spliting

Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan fungsi train_test_split.
"""

# Pisahkan fitur dan target
# Remove 'dataset' from the list of columns to drop since it was one-hot encoded
X = heart_encoded.drop(['target', 'num', 'id'], axis=1)
y = heart_encoded['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# Modeling

Tiga model machine learning digunakan untuk memprediksi apakah seseorang berisiko terkena penyakit jantung adalah:




1. K-Nearest Neighbors (KNN):

  Mengklasifikasikan berdasarkan kedekatan dengan data tetangga terdekat. Model KNN dibangun dengan parameter default n_neighbors=5.

2. Random Forest:

  Kombinasi dari banyak decision tree, memberikan hasil yang lebih stabil dan minim overfitting. Model Random Forest dibangun dengan n_estimators=100.

3. XGBoost:

  Metode boosting yang sangat powerful dan efisien dalam menangani dataset kompleks. Model XGBoost dibangun dengan parameter dasar, yaitu *eval_metric='logloss'* dan *use_label_encoder=False*
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===================
# 1. K-NEAREST NEIGHBOR
# ===================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("K-Nearest Neighbor")
print(classification_report(y_test, y_pred_knn))
print("Akurasi:", round(accuracy_score(y_test, y_pred_knn), 3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("-" * 40)

# ===================
# 2. RANDOM FOREST
# ===================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest")
print(classification_report(y_test, y_pred_rf))
print("Akurasi:", round(accuracy_score(y_test, y_pred_rf), 3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("-" * 40)

# ===================
# 3. XGBOOST (Boosting)
# ===================
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost")
print(classification_report(y_test, y_pred_xgb))
print("Akurasi:", round(accuracy_score(y_test, y_pred_xgb), 3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("-" * 40)

# Commented out IPython magic to ensure Python compatibility.
# %%writefile requirements.txt
# pandas~=2.2
# numpy~=1.26
# scikit-learn~=1.4
# xgboost~=2.0
# matplotlib~=3.8
# seaborn~=0.13