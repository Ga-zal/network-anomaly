X = df_scaled.drop('label', axis=1)
y = df_scaled['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
