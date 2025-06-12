import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_dataset(csv_path, test_size=0.2, random_state=42):
  
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

# Tambahkan bagian ini untuk menjalankan fungsi dan menyimpan hasilnya
if __name__ == "__main__":
    original_csv_path = 'Crop_recommendation.csv' 
    

    processed_data = load_and_preprocess_dataset(original_csv_path)

    # Simpan dataset yang sudah diproses penuh ke CSV
    # Nama file ini harus cocok dengan 'path' di upload-artifact workflow
    output_filename = "Crop_recommendation_prepocessing.csv"
    processed_data['full_preprocessed'].to_csv(output_filename, index=False)
    print(f"Dataset preprocessed saved to {output_filename}")