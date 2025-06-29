from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


wcss = []
range_n_clusters = range(1, 11) 

for n_clusters in range_n_clusters:
    kmeans_elbow = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans_elbow.fit(X_train)
    wcss.append(kmeans_elbow.inertia_) # inertia_ is the WCSS

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range_n_clusters)
plt.grid(True)
plt.show()


# ---- K-Means ----
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
y_pred_kmeans = kmeans.predict(X_test)

# ----  Isolation Forest ----
X_train_if = X_train[y_train == 0]  # Train only on normal (label=0)

# Initialize Isolation Forest with better parameters
iso_forest = IsolationForest(
    n_estimators=400,         # more trees
    max_samples=0.6,          # use 60% of training samples
    contamination=0.1,        # based on actual label ratio
    random_state=42,
    verbose=0
)

# Fit model
iso_forest.fit(X_train_if)

# Predict on test set
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convert -1 to 1 (anomaly)

