# 1. Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df_scaled)
plt.title("Normal vs Anomalous Traffic")
plt.xticks([0, 1], ['Normal', 'Anomaly'])
plt.xlabel("Traffic Type")
plt.ylabel("Count")
plt.show()
# 2. K-Means Cluster Scatter Plot
plt.figure(figsize=(6, 4))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred_kmeans, cmap='viridis', s=5)
plt.title("K-Means Cluster Assignments")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
# 3. Isolation Forest Anomaly Scatter Plot
plt.figure(figsize=(6, 4))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred_iso, cmap='coolwarm', s=5)
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
# 4. Confusion Matrix Function
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# Confusion Matrices
plot_conf_matrix(y_test, y_pred_kmeans, "K-Means Confusion Matrix")
plot_conf_matrix(y_test, y_pred_iso, "Isolation Forest Confusion Matrix")
# 5. Performance bar plot
metrics = ['Precision', 'Recall', 'F1 Score']
kmeans_scores = [precision_kmeans, recall_kmeans, f1_kmeans]
iso_scores = [precision_iso, recall_iso, f1_iso]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, kmeans_scores, width, label='K-Means')
plt.bar(x + width/2, iso_scores, width, label='Isolation Forest')

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
