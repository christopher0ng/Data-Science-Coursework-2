import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure dataset in same directory 
df = pd.read_csv('dataset/heart_statlog_cleveland_hungary_final.csv')

# Plot the proportions of targets with labels 0 and 1
sns.countplot(x='target',data=df,hue='target',palette='muted')
plt.savefig('figures/target_distribution.png')
plt.title('Distribution of targets')

# Plot the distribution of targets across the different categorical features
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

categorical_features = ['sex','chest pain type', 'fasting blood sugar', 'resting ecg','exercise angina','ST slope']

for i in range(6):
    sns.countplot(ax=axes[i], x=categorical_features[i], data=df,hue='target',palette='muted')

# Show the plots
plt.suptitle('Distribution of targets across categorical features')
plt.tight_layout()
plt.savefig('figures/target_distribution_categorical.png')

# Plot a heatmap to show the correlation between the different continuous features
plt.figure(figsize=(10,8))
sns.heatmap(df[['age','resting bp s','cholesterol','max heart rate','oldpeak']].corr(),annot=True,cmap='viridis')
plt.title('Correlation between continuous predictors')
plt.savefig('figures/features_heatmap.png')

# Plot the correlation of each of the continuous features with the target
plt.figure(figsize=(10,8))
df[['age','resting bp s','cholesterol','max heart rate','oldpeak','target']].corr()['target'].sort_values().drop('target').plot(kind='bar',)
plt.title('Correlation between continuous predictors and the target')
plt.savefig('figures/features_target_correlation.png')