# scripts/credit_risk_eda.py
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Define the CreditRiskEDA class
class CreditRiskEDA:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EDA class with the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be analyzed.
        """
        self.df = df

    def data_overview(self):
        """Provide an overview of the dataset including shape, data types, and first few rows."""
        print("Data Overview:")
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")
        print("\nColumn Data Types:")
        print(self.df.dtypes)
        print("\nFirst Five Rows:")
        display(self.df.head())
        print("\nMissing Values Overview:")
        print(self.df.isnull().sum())
        
    def summary_statistics(self):
        """
        Function to compute summary statistics like mean, median, std, skewness, etc.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the dataset to be analyzed.
        
        Returns:
        --------
        summary_stats : pandas.DataFrame
            DataFrame containing the summary statistics for numeric columns.
        """
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include='number')
        
        # Calculate basic summary statistics
        summary_stats = numeric_df.describe().T
        summary_stats['median'] = numeric_df.median()
        summary_stats['mode'] = numeric_df.mode().iloc[0]
        summary_stats['skewness'] = numeric_df.skew()
        summary_stats['kurtosis'] = numeric_df.kurtosis()
        
        # Print summary statistics
        # Sprint("Summary Statistics:\n", summary_stats)
        
        return summary_stats
    
    def plot_numerical_distribution(self, cols):
        """
        Function to plot multiple histograms in a grid layout.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the dataset.
        cols : list
            List of numeric columns to plot.
        n_rows : int
            Number of rows in the grid.
        n_cols : int
            Number of columns in the grid.
        """

        # Select numeric columns
        n_cols = len(cols)

        # Automatically determine grid size (square root method)
        n_rows = math.ceil(n_cols**0.5)
        n_cols = math.ceil(n_cols / n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.histplot(self.df[col], bins=15, kde=True, color='skyblue', edgecolor='black', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].axvline(self.df[col].mean(), color='red', linestyle='dashed', linewidth=1)
            axes[i].axvline(self.df[col].median(), color='green', linestyle='dashed', linewidth=1)
            axes[i].legend({'Mean': self.df[col].mean(), 'Median': self.df[col].median()})

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        
    # Function to plot skewness for each numerical feature
    def plot_skewness(self):
        df = self.df.select_dtypes(include='number')
        skewness = df.skew().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 4))
        sns.barplot(x=skewness.index, y=skewness.values, hue=skewness.index, legend=False, palette="coolwarm")
        plt.title("Skewness of Numerical Features", fontsize=16)
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Skewness", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        
    def plot_categorical_distribution(self):
        """Plot the distribution of categorical features."""
        cat_cols = self.df.select_dtypes(include=[object, 'category']).columns
        for col in cat_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=self.df[col], palette='Set2')
            plt.title(f'Distribution of {col}', fontsize=14)
            plt.xticks(rotation=45)
            plt.show()

    def correlation_analysis(self):
        """Generate and visualize the correlation matrix."""
        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=16)
        plt.show()
    
    def check_missing_values(self):
        """Check for missing values and visualize the missing data pattern."""
        missing_values = self.df.isnull().sum()
        print("\nMissing Values in Each Column:")
        print(missing_values[missing_values > 0])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap', fontsize=14)
        plt.show()

    def detect_outliers(self):
        """Detect outliers using box plots for numerical features."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(num_cols, 1):
            plt.subplot(3, 3, i)
            sns.boxplot(y=self.df[col], color='orange')
            plt.title(f'Boxplot of {col}', fontsize=12)
        plt.tight_layout()
        plt.show()

# Function to run the entire EDA process
def run_full_eda(file_path: str):
    """
    Run the full EDA process on the dataset by loading the data and performing 
    the necessary analysis and visualizations.

    Parameters:
    -----------
    file_path : str
        The path to the dataset file.
    """
    df = load_data(file_path)
    
    if not df.empty:
        eda = CreditRiskEDA(df)
        
        # Perform EDA tasks
        print("\n--- Data Overview ---")
        eda.data_overview()
        
        print("\n--- Summary Statistics ---")
        eda.summary_statistics()
        
        print("\n--- Numerical Feature Distributions ---")
        eda.plot_numerical_distribution()
        
        print("\n--- Categorical Feature Distributions ---")
        eda.plot_categorical_distribution()
        
        print("\n--- Correlation Analysis ---")
        eda.correlation_analysis()
        
        print("\n--- Missing Values ---")
        eda.check_missing_values()
        
        print("\n--- Outlier Detection ---")
        eda.detect_outliers()

