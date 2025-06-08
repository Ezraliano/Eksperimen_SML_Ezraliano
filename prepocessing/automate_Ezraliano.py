import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_dataset(csv_path, test_size=0.2, random_state=42):
    """
    Fungsi otomatis untuk melakukan preprocessing dataset Crop Recommendation.
    Tahapan:
    1. Load data
    2. Cek & hapus duplikat (jika ada)
    3. Cek & tangani missing values (jika ada)
    4. Standarisasi fitur numerik
    5. Split data train/test
    6. Kembalikan data siap latih (fitur sudah distandarisasi dan label)
    """
    # 1. Load data
    df = pd.read_csv(csv_path)

    # 2. Hapus duplikat jika ada
    df = df.drop_duplicates()

    # 3. Tangani missing values (jika ada)
    df = df.dropna()

    # 4. Pisahkan fitur dan target
    X = df.drop('label', axis=1)
    y = df['label']

    # 5. Standarisasi fitur numerik
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Split data train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 7. Kembalikan hasil dalam bentuk DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df['label'] = y.values
    train_df = pd.DataFrame(X_train, columns=X.columns)
    train_df['label'] = y_train.values
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df['label'] = y_test.values

    return {
        'full_preprocessed': X_scaled_df,
        'train': train_df,
        'test': test_df,
        'scaler': scaler
    }

