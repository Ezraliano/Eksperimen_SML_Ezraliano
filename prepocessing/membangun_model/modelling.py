import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.sklearn.autolog()

DATA_PATH = 'prepocessing/membangun_model/Crop_recommendation_prepocessing.csv'
df = pd.read_csv(DATA_PATH, index_col=0)



X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)

with mlflow.start_run() as run:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Akurasi:', acc)
    print(classification_report(y_test, y_pred))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")