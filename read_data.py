# Define column names (based on KDD Cup documentation)
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

# Reload the dataset with column names
df = pd.read_csv("/content/kddcup.data_10_percent_corrected", names=column_names, header=None)

# Show first 5 rows with column names
#print(df.head(100))
df
print("Dataset Shape:", df.shape)
print("\nColumn Data Types:")
print(df.dtypes)

#  Summary statistics
print("\nDescriptive Statistics:")
print(df.describe(include="all"))

df.info()
df.to_csv("saved_dataset.csv")
