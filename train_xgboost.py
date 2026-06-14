import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

df = pd.read_csv('tiago_collision_dataset_17dof_1M.csv')

X = df.drop(columns=['collision', 'num_contacts'])
y = df['collision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

start_time = time.perf_counter()
y_pred = model.predict(X_test)
end_time = time.perf_counter()

y_proba = model.predict_proba(X_test)[:, 1]

total_inference_time = end_time - start_time
num_samples = len(X_test)
avg_time_per_sample_ms = (total_inference_time / num_samples) * 1000
samples_per_second = num_samples / total_inference_time

print("--- Model Assessment ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Total Inference Time: {total_inference_time:.6f} seconds")
print(f"Average Time per Sample: {avg_time_per_sample_ms:.4f} ms")
print(f"Samples per Second: {samples_per_second:.2f}")