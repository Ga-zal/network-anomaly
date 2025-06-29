most_frequent_protocol = X_train[y_train == 0]['protocol_type'].mode()[0]
print(f"The most frequent protocol type in normal training data is: {most_frequent_protocol}")
if 'normal_stats' in locals():
    print("normal_stats is defined.")
else:
    print("normal_stats is NOT defined.")
  import pandas as pd

if 'X_train' in locals() and 'y_train' in locals():

    normal_stats = X_train[y_train == 0].describe()
    normal_stats = normal_stats.fillna(0) 
    print("normal_stats has been successfully created.")
else:
    print("Error: X_train or y_train is not defined. Please run the data loading, preprocessing, and scaling steps first.")
# --- Generate Synthetic Anomalous Data ---
n_synthetic_anomalies = 500
synthetic_anomalies = pd.DataFrame(index=range(n_synthetic_anomalies), columns=X_train.columns)

# Assign 'duration' for all synthetic anomalies with high values
synthetic_anomalies['duration'] = np.random.uniform(normal_stats.loc['max', 'duration'] * 5, normal_stats.loc['max', 'duration'] * 10, n_synthetic_anomalies)

# Assign 'src_bytes' for all synthetic anomalies with high values
synthetic_anomalies['src_bytes'] = np.random.uniform(normal_stats.loc['max', 'src_bytes'] * 5, normal_stats.loc['max', 'src_bytes'] * 10, n_synthetic_anomalies)

# Fill other features with values within the normal range
for col in X_train.columns:
    if col not in ['duration', 'src_bytes']:
        synthetic_anomalies[col] = np.random.uniform(normal_stats.loc['min', col], normal_stats.loc['max', col], n_synthetic_anomalies)

synthetic_anomalies['label_synthetic'] = 1  # Mark as anomaly

# --- Generate Synthetic Normal Data ---
n_synthetic_normal = 500
synthetic_normal = pd.DataFrame(index=range(n_synthetic_normal), columns=X_train.columns)
for col in X_train.columns:
    synthetic_normal[col] = np.random.uniform(normal_stats.loc['min', col], normal_stats.loc['max', col], n_synthetic_normal)
synthetic_normal['label_synthetic'] = 0  # Mark as normal

# Combine synthetic data
synthetic_data = pd.concat([synthetic_normal, synthetic_anomalies], ignore_index=True)
X_synthetic = synthetic_data.drop('label_synthetic', axis=1)
y_synthetic = synthetic_data['label_synthetic']

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Assuming X_synthetic is already defined

wcss_synthetic = []
range_n_clusters_synthetic = range(1, 11) # Let's try numbers of clusters from 1 to 10

for n_clusters in range_n_clusters_synthetic:
    kmeans_synthetic_elbow = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans_synthetic_elbow.fit(X_synthetic)
    wcss_synthetic.append(kmeans_synthetic_elbow.inertia_) # inertia_ is the WCSS

# Plot the elbow curve for synthetic data
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters_synthetic, wcss_synthetic, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal Number of Clusters (Synthetic Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range_n_clusters_synthetic)
plt.grid(True)
plt.show()
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming X_synthetic and y_synthetic are already defined

# --- Predict on Synthetic Data (K-Means with n_clusters=2) ---
kmeans_n2 = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_n2.fit(X_synthetic)
y_pred_kmeans_n2 = kmeans_n2.predict(X_synthetic)

# --- Evaluate K-Means (n_clusters=2) ---
precision_km_n2 = precision_score(y_synthetic, y_pred_kmeans_n2)
recall_km_n2 = recall_score(y_synthetic, y_pred_kmeans_n2)
f1_km_n2 = f1_score(y_synthetic, y_pred_kmeans_n2)

# --- Predict on Synthetic Data (Isolation Forest - Using Original Model with Adjusted Threshold) ---
scores_synthetic = iso_forest.decision_function(X_synthetic)
threshold_synthetic = np.percentile(scores_synthetic, 100 * (1 - 0.1)) # Threshold based on 50% contamination in synthetic data
y_pred_iso_synthetic_adjusted = np.where(scores_synthetic <= threshold_synthetic, 1, 0) # 1 for anomaly

precision_iso = precision_score(y_synthetic, y_pred_iso_synthetic_adjusted)
recall_iso = recall_score(y_synthetic, y_pred_iso_synthetic_adjusted)
f1_iso = f1_score(y_synthetic, y_pred_iso_synthetic_adjusted)

# --- Evaluate on Synthetic Data ---
print("\n📊 K-Means Performance (Synthetic Data - n_clusters=2):")
print(f"Precision: {precision_km_n2:.4f}")
print(f"Recall:   {recall_km_n2:.4f}")
print(f"F1 Score:  {f1_km_n2:.4f}\n")

print("📊 Isolation Forest Performance (Synthetic Data - Original Model, Adjusted Threshold):")
print(f"Precision: {precision_iso:.4f}")
print(f"Recall:   {recall_iso:.4f}")
print(f"F1 Score:  {f1_iso:.4f}\n")

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

plot_conf_matrix(y_synthetic, y_pred_kmeans_n2, "K-Means Confusion Matrix (Synthetic Data - n=2)")
plot_conf_matrix(y_synthetic, y_pred_iso_synthetic_adjusted, "Isolation Forest Confusion Matrix (Synthetic Data - Adjusted Threshold)")import matplotlib.pyplot as plt
import numpy as np

metrics = ['Precision', 'Recall', 'F1 Score']
kmeans_scores = [precision_km_n2, recall_km_n2, f1_km_n2]
iso_scores = [precision_iso, recall_iso, f1_iso]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, kmeans_scores, width, label='K-Means')
plt.bar(x + width/2, iso_scores, width, label='Isolation Forest')

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Model Performance Comparison on Synthetic Data')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
