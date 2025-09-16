#!/usr/bin/env python3
"""
Simple script to generate all Iris EDA plots
Run this after installing: pip install pandas numpy matplotlib seaborn scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os
import warnings
warnings.filterwarnings('ignore')

# Create plots directory
os.makedirs('plots', exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
print("Loading Iris dataset...")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Save dataset
df.to_csv('iris_dataset.csv', index=False)
print("✓ Dataset saved as iris_dataset.csv")

# 1. Pairplot
print("Creating pairplot...")
plt.figure(figsize=(12, 10))
g = sns.pairplot(df, hue='species', diag_kind='kde', height=2.5)
g.fig.suptitle('Pairplot of Iris Dataset Features by Species', y=1.02)
plt.savefig('plots/01_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Pairplot saved")

# 2. Boxplot - Sepal Length
print("Creating boxplot...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='species', y='sepal_length', palette='Set2')
plt.title('Distribution of Sepal Length by Species', fontsize=16, pad=20)
plt.xlabel('Species', fontsize=14)
plt.ylabel('Sepal Length (cm)', fontsize=14)
plt.savefig('plots/02_boxplot_sepal_length.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Boxplot saved")

# 3. Violin Plot - Petal Length
print("Creating violin plot...")
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='species', y='petal_length', palette='viridis')
plt.title('Distribution of Petal Length by Species', fontsize=16, pad=20)
plt.xlabel('Species', fontsize=14)
plt.ylabel('Petal Length (cm)', fontsize=14)
plt.savefig('plots/03_violin_petal_length.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Violin plot saved")

# 4. KDE Plot - Sepal Width
print("Creating KDE plot...")
plt.figure(figsize=(10, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    sns.kdeplot(data=subset, x='sepal_width', label=species, fill=True, alpha=0.6)
plt.title('KDE Plot of Sepal Width by Species', fontsize=16, pad=20)
plt.xlabel('Sepal Width (cm)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(title='Species')
plt.savefig('plots/04_kde_sepal_width.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ KDE plot saved")

# 5. Swarm Plot - Petal Length
print("Creating swarm plot...")
plt.figure(figsize=(10, 6))
sns.swarmplot(data=df, x='species', y='petal_length', palette='husl', size=6)
plt.title('Swarm Plot of Petal Length by Species', fontsize=16, pad=20)
plt.xlabel('Species', fontsize=14)
plt.ylabel('Petal Length (cm)', fontsize=14)
plt.savefig('plots/05_swarm_petal_length.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Swarm plot saved")

# 6. Correlation Heatmap
print("Creating correlation heatmap...")
plt.figure(figsize=(10, 8))
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Iris Features', fontsize=16, pad=20)
plt.savefig('plots/06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Correlation heatmap saved")

# 7. Count Plot
print("Creating count plot...")
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x='species', palette='pastel')
plt.title('Number of Samples per Species', fontsize=16, pad=20)
plt.xlabel('Species', fontsize=14)
plt.ylabel('Count', fontsize=14)
# Add count labels
for i, v in enumerate(df['species'].value_counts().values):
    ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.savefig('plots/07_count_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Count plot saved")

print("\n🎉 All plots generated successfully!")
print("Files created:")
print("- iris_dataset.csv")
print("- plots/01_pairplot.png")
print("- plots/02_boxplot_sepal_length.png") 
print("- plots/03_violin_petal_length.png")
print("- plots/04_kde_sepal_width.png")
print("- plots/05_swarm_petal_length.png")
print("- plots/06_correlation_heatmap.png")
print("- plots/07_count_plot.png")
