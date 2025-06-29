# plots display in notebook
%matplotlib inline

# 1. Class distribution (Normal vs. Attack)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', hue='label', palette='Set2', legend=False)
plt.title('Class Distribution: Normal vs Attack')
plt.xticks(rotation=45)
plt.show()
# 2. Correlation Heatmap
plt.figure(figsize=(16, 12))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()
# 3. Protocol type distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='protocol_type', hue='protocol_type', palette='pastel', legend=False)
plt.title('Protocol Type Distribution')
plt.show()
# 4. Top 10 most frequent services
plt.figure(figsize=(10, 4))
top_services = df['service'].value_counts().nlargest(10)
sns.barplot(x=top_services.index, y=top_services.values, palette='mako', hue=top_services.index)
plt.title('Top 10 Services Used')
plt.xticks(rotation=45)
plt.show()
