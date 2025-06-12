import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import json
import os
import pathlib

# Menggunakan pathlib untuk membuat URI yang benar untuk MLflow di Windows
mlruns_path = pathlib.Path('mlruns').resolve().as_uri()
mlflow.set_tracking_uri("http://localhost:5000")
print("MLflow tracking URI diatur ke: http://localhost:5000")

# Load data hasil preprocessing
DATA_PATH = 'prepocessing/Crop_recommendation_prepocessing.csv'
df = pd.read_csv(DATA_PATH)

# Pisahkan fitur dan target
X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Lakukan scaling pada data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning dengan GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200, 500, 1000]
}

model = LogisticRegression()
gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
gs.fit(X_train_scaled, y_train)

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)



output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


report_path = os.path.join(output_dir, 'classification_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

# Manual logging ke MLflow
with mlflow.start_run():
    # Log parameter terbaik
    mlflow.log_params(gs.best_params_)
    # Log metrik utama
    mlflow.log_metric('accuracy', acc)

    
    mlflow.log_artifacts(output_dir)

    # Log model
    mlflow.sklearn.log_model(best_model, 'model')

    print('\n--- Hasil Model Terbaik ---')
    print(f"Parameter Terbaik: {gs.best_params_}")
    print(f"Akurasi: {acc}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

