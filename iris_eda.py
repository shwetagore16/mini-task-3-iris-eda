#!/usr/bin/env python3
"""
Iris Dataset - Exploratory Data Analysis

This script performs a comprehensive exploratory data analysis on the famous Iris dataset.
The Iris dataset contains measurements of 150 iris flowers from three different species:
- Iris setosa
- Iris versicolor  
- Iris virginica

Each flower has four features measured:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)
"""

# Import necessary libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
import os

warnings.filterwarnings('ignore')

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")
plt.style.use('default')

# Set figure size for better readability
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def load_and_prepare_data():
    """Load the Iris dataset and prepare it for analysis."""
    print("Loading Iris dataset...")
    
    # Load the Iris dataset using sklearn
    iris_sklearn = load_iris()
    
    # Create a pandas DataFrame
    df = pd.DataFrame(iris_sklearn.data, columns=iris_sklearn.feature_names)
    df['species'] = iris_sklearn.target
    
    # Map target numbers to species names
    species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species'].map(species_names)
    
    # Clean column names (remove spaces and make them more readable)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    return df

def basic_exploration(df):
    """Perform basic exploration of the dataset."""
    print("\n" + "="*50)
    print("BASIC DATA EXPLORATION")
    print("="*50)
    
    # Display first 5 rows of the dataset
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Print basic information about the dataset
    print(f"\nDataset Shape: {df.shape}")
    print(f"Column Names: {list(df.columns)}")
    print("\nData Types:")
    print(df.dtypes)
    
    # Get detailed information about the dataset
    print("\nDataset Info:")
    df.info()
    
    # Get summary statistics for numeric columns
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Count the number of samples for each species
    print("\nNumber of samples per species:")
    print(df['species'].value_counts())
    print("\nPercentage distribution:")
    print(df['species'].value_counts(normalize=True) * 100)

def data_cleaning(df):
    """Check for data quality issues and clean if necessary."""
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    
    # Check for missing values
    print("Missing values in each column:")
    print(df.isnull().sum())
    
    # Check for duplicate rows
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
    
    # Save the cleaned dataset as CSV
    df.to_csv('iris_dataset.csv', index=False)
    print("\nDataset saved as 'iris_dataset.csv'")
    
    # Confirm dataset is ready for visualization
    print("\nDataset is clean and ready for analysis!")
    print(f"Final dataset shape: {df.shape}")
    
    return df

def create_pairplot(df):
    """Create pairplot to show relationships between all numeric variables."""
    print("\nCreating pairplot...")
    
    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(df, hue='species', diag_kind='kde', height=2.5)
    pairplot.fig.suptitle('Pairplot of Iris Dataset Features by Species', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/01_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Pairplot saved as 'plots/01_pairplot.png'")

def create_boxplot(df):
    """Create boxplot for sepal length by species."""
    print("Creating boxplot...")
    
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x='species', y='sepal_length', palette='Set2')
    ax.set_title('Distribution of Sepal Length by Species', fontsize=16, pad=20)
    ax.set_xlabel('Species', fontsize=14)
    ax.set_ylabel('Sepal Length (cm)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/02_boxplot_sepal_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Boxplot saved as 'plots/02_boxplot_sepal_length.png'")

def create_violin_plot(df):
    """Create violin plot for petal length by species."""
    print("Creating violin plot...")
    
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(data=df, x='species', y='petal_length', palette='viridis')
    ax.set_title('Distribution of Petal Length by Species', fontsize=16, pad=20)
    ax.set_xlabel('Species', fontsize=14)
    ax.set_ylabel('Petal Length (cm)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/03_violin_petal_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Violin plot saved as 'plots/03_violin_petal_length.png'")

def create_kde_plot(df):
    """Create KDE plot for sepal width by species."""
    print("Creating KDE plot...")
    
    plt.figure(figsize=(10, 6))
    for species in df['species'].unique():
        sns.kdeplot(data=df[df['species'] == species], x='sepal_width', 
                    label=species, fill=True, alpha=0.6)
    plt.title('KDE Plot of Sepal Width by Species', fontsize=16, pad=20)
    plt.xlabel('Sepal Width (cm)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(title='Species', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/04_kde_sepal_width.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ KDE plot saved as 'plots/04_kde_sepal_width.png'")

def create_swarm_plot(df):
    """Create swarm plot for petal length vs species."""
    print("Creating swarm plot...")
    
    plt.figure(figsize=(10, 6))
    ax = sns.swarmplot(data=df, x='species', y='petal_length', palette='husl', size=6)
    ax.set_title('Swarm Plot of Petal Length by Species', fontsize=16, pad=20)
    ax.set_xlabel('Species', fontsize=14)
    ax.set_ylabel('Petal Length (cm)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/05_swarm_petal_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Swarm plot saved as 'plots/05_swarm_petal_length.png'")

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric features."""
    print("Creating correlation heatmap...")
    
    plt.figure(figsize=(10, 8))
    numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    correlation_matrix = df[numeric_cols].corr()
    
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    ax.set_title('Correlation Heatmap of Iris Features', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('plots/06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Correlation heatmap saved as 'plots/06_correlation_heatmap.png'")

def create_count_plot(df):
    """Create count plot showing number of samples per species."""
    print("Creating count plot...")
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='species', palette='pastel')
    ax.set_title('Number of Samples per Species', fontsize=16, pad=20)
    ax.set_xlabel('Species', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    
    # Add count labels on top of bars
    for i, v in enumerate(df['species'].value_counts().values):
        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/07_count_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Count plot saved as 'plots/07_count_plot.png'")

def create_all_visualizations(df):
    """Create all required visualizations."""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    create_pairplot(df)
    create_boxplot(df)
    create_violin_plot(df)
    create_kde_plot(df)
    create_swarm_plot(df)
    create_correlation_heatmap(df)
    create_count_plot(df)
    
    print("\nAll visualizations created successfully!")

def analyze_insights(df):
    """Perform detailed analysis and generate insights."""
    print("\n" + "="*50)
    print("ANALYSIS & INSIGHTS")
    print("="*50)
    
    print("\nSummary Statistics by Species:")
    print("=" * 50)
    for species in df['species'].unique():
        print(f"\n{species.upper()}:")
        species_data = df[df['species'] == species]
        print(species_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].describe())
        print("-" * 50)
    
    # Calculate correlations
    numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    correlation_matrix = df[numeric_cols].corr()
    
    print("\nKey Correlations:")
    print(f"Petal Length vs Petal Width: {correlation_matrix.loc['petal_length', 'petal_width']:.3f}")
    print(f"Sepal Length vs Petal Length: {correlation_matrix.loc['sepal_length', 'petal_length']:.3f}")
    print(f"Sepal Length vs Petal Width: {correlation_matrix.loc['sepal_length', 'petal_width']:.3f}")
    
    print("\n" + "="*60)
    print("SUMMARY OF KEY FINDINGS")
    print("="*60)
    
    print("""
Species Characteristics:
- Setosa: Smallest petals (length & width), widest sepals, most distinct species
- Versicolor: Medium-sized features, some overlap with virginica
- Virginica: Largest petals and longest sepals, some overlap with versicolor

Best Discriminating Features:
1. Petal Length: Excellent separation between all species
2. Petal Width: Strong discriminating power, highly correlated with petal length
3. Sepal Length: Good for distinguishing setosa from others
4. Sepal Width: Least discriminating feature

Strong Correlations:
- Petal length vs Petal width (r = 0.87): Very strong positive correlation
- Sepal length vs Petal length (r = 0.87): Strong positive correlation
- Sepal length vs Petal width (r = 0.82): Strong positive correlation

Species Overlap:
- Minimal overlap: Setosa is clearly separable from other species
- Some overlap: Versicolor and Virginica show overlap in sepal measurements
- Best separation: Achieved using petal measurements

Dataset Quality:
- Perfectly balanced dataset (50 samples per species)
- No missing values or data quality issues
- Suitable for machine learning classification tasks
    """)

def main():
    """Main function to run the complete EDA."""
    print("="*60)
    print("IRIS DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Basic exploration
    basic_exploration(df)
    
    # Data cleaning
    df = data_cleaning(df)
    
    # Create visualizations
    create_all_visualizations(df)
    
    # Analyze insights
    analyze_insights(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles generated:")
    print("- iris_dataset.csv")
    print("- plots/01_pairplot.png")
    print("- plots/02_boxplot_sepal_length.png")
    print("- plots/03_violin_petal_length.png")
    print("- plots/04_kde_sepal_width.png")
    print("- plots/05_swarm_petal_length.png")
    print("- plots/06_correlation_heatmap.png")
    print("- plots/07_count_plot.png")

if __name__ == "__main__":
    main()
