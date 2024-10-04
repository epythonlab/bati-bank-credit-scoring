import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class CreditScoringModel:
    """
    A class to perform RFMS-based default estimator and WoE binning.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_rfms(self):
        """
        Calculate Recency, Frequency, and Monetary value for RFMS formalism.

        Returns
        -------
        pd.DataFrame
            DataFrame with added RFMS features.
        """
        # Recency: Days since the last transaction
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        max_date = self.df['TransactionStartTime'].max()
        self.df['Recency'] = (max_date - self.df['TransactionStartTime']).dt.days
        
        # Frequency: Count of transactions per customer
        freq_df = self.df.groupby('CustomerId').agg(Frequency=('TransactionId', 'count')).reset_index()
        
        # Monetary Value: Sum of transaction amounts per customer
        mon_value_df = self.df.groupby('CustomerId').agg(Monetary=('Amount', 'sum')).reset_index()
        
        # Merge the RFMS features
        rfms_df = pd.merge(freq_df, mon_value_df, on='CustomerId')
        self.df = pd.merge(self.df, rfms_df, on='CustomerId', how='left')
        
        return self.df
    
    def visualize_rfms(self):
        """
        Visualize RFMS space using clustering.
        """
        # Select the RFMS columns
        rfms = self.df[['Recency', 'Frequency', 'Monetary']].dropna()

        # KMeans Clustering to separate users
        kmeans = KMeans(n_clusters=2)
        self.df['RFMS_Label'] = kmeans.fit_predict(rfms)

        # Plot the clusters
        plt.scatter(rfms['Recency'], rfms['Frequency'], c=self.df['RFMS_Label'])
        plt.xlabel('Recency')
        plt.ylabel('Frequency')
        plt.title('Clustering in RFMS Space')
        plt.show()
        
        return self.df
    
    def assign_good_bad_labels(self):
        """
        Assign 'good' and 'bad' labels based on RFMS score.

        Returns
        -------
        pd.DataFrame
            DataFrame with good and bad labels.
        """
        # Assuming RFMS_Label 0 is good, 1 is bad
        self.df['User_Label'] = np.where(self.df['RFMS_Label'] == 0, 'Good', 'Bad')
        return self.df
    
    def perform_woe_binning(self):
        """
        Perform WoE binning based on the RFMS score.

        Returns
        -------
        pd.DataFrame
            DataFrame with WoE-transformed RFMS scores.
        """
        def calculate_woe(df, target_col, feature_col):
            bins = pd.cut(df[feature_col], bins=5)  # You can adjust bin size
            woe_df = df.groupby(bins).agg({target_col: ['count', 'sum']})
            woe_df.columns = ['Total', 'Bad']
            woe_df['Good'] = woe_df['Total'] - woe_df['Bad']
            woe_df['Bad_Rate'] = woe_df['Bad'] / woe_df['Total']
            woe_df['WoE'] = np.log((woe_df['Good'] / woe_df['Good'].sum()) / (woe_df['Bad'] / woe_df['Bad'].sum()))
            return woe_df
        
        # Perform WoE binning on Recency, Frequency, and Monetary
        woe_recency = calculate_woe(self.df, 'FraudResult', 'Recency')
        woe_frequency = calculate_woe(self.df, 'FraudResult', 'Frequency')
        woe_monetary = calculate_woe(self.df, 'FraudResult', 'Monetary')
        
        return woe_recency, woe_frequency, woe_monetary


