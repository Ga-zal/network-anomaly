# K-Means Scores
precision_kmeans = precision_score(y_test, y_pred_kmeans)
recall_kmeans = recall_score(y_test, y_pred_kmeans)
f1_kmeans = f1_score(y_test, y_pred_kmeans)

# Isolation Forest Scores
precision_iso = precision_score(y_test, y_pred_iso)
recall_iso = recall_score(y_test, y_pred_iso)
f1_iso = f1_score(y_test, y_pred_iso)

# Print results
print("📊 K-Means Performance:")
print(f"Precision: {precision_kmeans:.4f}")
print(f"Recall:    {recall_kmeans:.4f}")
print(f"F1 Score:  {f1_kmeans:.4f}\n")

print("📊 Isolation Forest Performance:")
print(f"Precision: {precision_iso:.4f}")
print(f"Recall:    {recall_iso:.4f}")
print(f"F1 Score:  {f1_iso:.4f}\n")
