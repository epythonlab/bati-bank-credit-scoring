# scripts/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureEngineering:
    """
    A class to perform feature engineering tasks for a given DataFrame.

    Methods
    -------
    create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame
        Creates aggregate features such as total, average, count, and standard deviation of transaction amounts.

    create_transaction_features(df: pd.DataFrame) -> pd.DataFrame
        Creates features based on transaction types (debit/credit) and their ratios.

    extract_time_features(df: pd.DataFrame) -> pd.DataFrame
        Extracts time-related features from the TransactionStartTime column.

    encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame
        Encodes categorical variables into numerical format using One-Hot or Label Encoding.

    handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame
        Handles missing values by imputation or removal.

    normalize_numerical_features(df: pd.DataFrame, numerical_cols: list, method: str = 'standardize') -> pd.DataFrame
        Normalizes or standardizes numerical features.
    """

    @staticmethod
    def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        agg_features = df.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount', 'sum'),
            Average_Transaction_Amount=('Amount', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Std_Transaction_Amount=('Amount', 'std')
        ).reset_index()
        
        df = df.merge(agg_features, on='CustomerId', how='left')
        return df

    @staticmethod
    def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features based on transaction types (debit/credit) and their ratios.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing transaction data.

        Returns
        -------
        pd.DataFrame
            The DataFrame with new transaction-based features.
        """
        # Group by 'CustomerId' to calculate net transaction amount
        transaction_features = df.groupby('CustomerId').agg(
            Net_Transaction_Amount=('Amount', 'sum'),
            Debit_Count=('Amount', lambda x: (x > 0).sum()),
            Credit_Count=('Amount', lambda x: (x < 0).sum())
        ).reset_index()

        # Calculate the debit/credit ratio and handle division by zero
        transaction_features['Debit_Credit_Ratio'] = transaction_features['Debit_Count'] / (
            transaction_features['Credit_Count'] + 1)  # Adding 1 to avoid division by zero

        # Merge the new features back to the original DataFrame on 'CustomerId'
        df = pd.merge(df, transaction_features, on='CustomerId', how='left')

        return df


    @staticmethod
    def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
        df['Transaction_Day'] = df['TransactionStartTime'].dt.day
        df['Transaction_Month'] = df['TransactionStartTime'].dt.month
        df['Transaction_Year'] = df['TransactionStartTime'].dt.year
        
        return df

    @staticmethod
    def encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """
        Encodes categorical variables into numerical format using One-Hot Encoding.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing categorical data.
        categorical_cols : list
            List of categorical columns to encode.

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical features.
        """
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        for col in categorical_cols:
            # Fit and transform the column, then convert to DataFrame
            encoded_col = encoder.fit_transform(df[[col]].astype(str))  # Use double brackets for a DataFrame
            # Get the category names from the encoder
            category_names = encoder.get_feature_names_out(input_features=[col])
            # Create a DataFrame with the actual category names
            encoded_df = pd.DataFrame(encoded_col, columns=category_names)
            
            # Concatenate the new DataFrame with the original DataFrame and drop the original column
            df = pd.concat([df, encoded_df], axis=1)
            df.drop(columns=[col], inplace=True)  # Drop the original column

        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        if strategy in ['mean', 'median', 'mode']:
            imputer = SimpleImputer(strategy=strategy)
            df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))
        elif strategy == 'remove':
            df.dropna(inplace=True)
        return df

    @staticmethod
    def normalize_numerical_features(df: pd.DataFrame, numerical_cols: list, method: str = 'standardize') -> pd.DataFrame:
        if method == 'standardize':
            scaler = StandardScaler()
        elif method == 'normalize':
            scaler = MinMaxScaler()

        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        df.set_index('TransactionId', inplace=True)
        return df
