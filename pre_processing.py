#df.isnull()
# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

df.drop('num_outbound_cmds', axis=1, inplace=True)
print(df.columns.tolist())from sklearn.preprocessing import LabelEncoder
import pandas as pd

le_protocol = LabelEncoder()
le_protocol.fit(df['protocol_type'])
print(le_protocol.classes_)
from sklearn.preprocessing import LabelEncoder
import pandas as pd

le_service = LabelEncoder()
le_service.fit(df['service']) 
print("Service Encoding Mapping:")
for i, service_name in enumerate(le_service.classes_):
    print(f"{service_name} -> {i}")

# Encode categorical features: protocol_type, service, flag
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Convert label to binary: 0 for normal, 1 for anomaly
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# Feature Scaling: Normalize all feature columns (excluding label)
scaler = StandardScaler()
features = df.drop(['label'], axis=1)
scaled_features = scaler.fit_transform(features)

# Create a new DataFrame with scaled features and label
df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
df_scaled['label'] = df['label'].values

