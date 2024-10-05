import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CreditScoreRFMS:
    """
    A class to calculate Recency, Frequency, Monetary values and perform Weight of Evidence (WoE) binning,
    as well as merging these features into the original feature-engineered dataset.

    Attributes:
    -----------
    rfms_data : pd.DataFrame
        The dataset containing the transaction information for RFMS calculation.

    Methods:
    --------
    calculate_rfms(current_date):
        Calculates Recency, Frequency, and Monetary values for each customer in the dataset.
    
    visualize_rfms():
        Plots a 3D scatter plot of the Recency, Frequency, and Monetary values.

    calculate_rfms_score():
        Calculates an RFMS score based on Recency, Frequency, and Monetary values.

    assign_good_bad_labels():
        Assigns users into "Good" and "Bad" categories based on the RFMS score threshold.

    calc_woe_iv(feature, target):
        Performs WoE binning for a given feature and returns the WoE-transformed feature along with IV.

    merge_with_feature_data(feature_data):
        Merges the RFMS dataset with an external feature-engineered dataset using TransactionId and CustomerId.
    """

    def __init__(self, rfms_data):
        """
        Initializes the RFMS class with the provided dataset.
        
        Parameters:
        -----------
        rfms_data : pd.DataFrame
            The input dataset containing transaction data.
        """
        self.rfms_data = rfms_data

    def calculate_rfms(self):
        """
        Calculates Recency, Frequency, and Monetary values for each customer using the last transaction date as the end date.
        """
        # Convert 'TransactionStartTime' to datetime if not already
        self.rfms_data['TransactionStartTime'] = pd.to_datetime(self.rfms_data['TransactionStartTime'])

        # Determine the end date as the maximum TransactionStartTime
        end_date = self.rfms_data['TransactionStartTime'].max()

        # Calculate the last access date (most recent transaction) for each customer
        self.rfms_data['Last_Access_Date'] = self.rfms_data.groupby('CustomerId')['TransactionStartTime'].transform('max')

        # Calculate Recency: Time in days since the last transaction before the statistical deadline
        self.rfms_data['Recency'] = (end_date - self.rfms_data['Last_Access_Date']).dt.days

        # Frequency: Number of transactions per customer
        self.rfms_data['Frequency'] = self.rfms_data.groupby('CustomerId')['TransactionId'].transform('count')

        # Monetary: Sum of transaction values per customer
        self.rfms_data['Monetary'] = self.rfms_data.groupby('CustomerId')['Amount'].transform('sum')

        return self.rfms_data

    def visualize_rfms(self):
        """
        Plots a 3D scatter plot of Recency, Frequency, and Monetary values for visual analysis.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.rfms_data['Recency'], self.rfms_data['Frequency'], self.rfms_data['Monetary'], c='b', marker='o')
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        plt.title('RFMS Space Visualization')
        plt.show()

    def calculate_rfms_score(self):
        """
        Calculates the RFMS score for each customer by combining Recency, Frequency, and Monetary values.
        The weightings can be adjusted as necessary.
        """
        self.rfms_data['RFMS_Score'] = self.rfms_data['Frequency'] * 0.5 + self.rfms_data['Monetary'] * 0.4 - self.rfms_data['Recency'] * 0.1

    def assign_good_bad_labels(self):
        """
        Assigns users into 'Good' or 'Bad' risk categories based on the median RFMS score.
        """
        threshold = self.rfms_data['RFMS_Score'].median()
        self.rfms_data['Risk_Label'] = self.rfms_data['RFMS_Score'].apply(lambda x: 'Good' if x >= threshold else 'Bad')

    def calc_woe_iv(self, feature, target):
        """
        Calculates Weight of Evidence (WoE) and Information Value (IV) for a given feature and target.

        Parameters:
        -----------
        feature : str
            The feature for which to calculate WoE and IV.
        
        target : str
            The target column (usually binary) indicating good/bad or fraud status.

        Returns:
        --------
        pd.DataFrame:
            WoE and IV values for each bin of the feature.
        
        float:
            The total Information Value (IV) for the feature.
        """
        # Create quantile bins for the feature
        self.rfms_data['bin'] = pd.qcut(self.rfms_data[feature], q=10, duplicates='drop')
        
        # Group by bins and calculate good/bad counts
        grouped = self.rfms_data.groupby('bin')[target].agg(['count', 'sum'])
        grouped['good'] = grouped['count'] - grouped['sum']
        total_good = grouped['good'].sum()
        total_bad = grouped['sum'].sum()
        
        # Calculate WoE and IV
        grouped['WoE'] = np.log((grouped['good'] / total_good) / (grouped['sum'] / total_bad))
        grouped['IV'] = ((grouped['good'] / total_good) - (grouped['sum'] / total_bad)) * grouped['WoE']
        iv = grouped['IV'].sum()
        
        return grouped[['WoE', 'IV']], iv

    def merge_with_feature_data(self, feature_data):
        """
        Merges the RFMS dataset with the external feature-engineered dataset using 'TransactionId' and 'CustomerId'.

        Parameters:
        -----------
        feature_data : pd.DataFrame
            The feature-engineered dataset to merge with.
        
        Returns:
        --------
        pd.DataFrame:
            The merged dataset containing both the original features and the RFMS features.
        """
        # Merge datasets on 'TransactionId' and 'CustomerId'
        merged_data = feature_data.merge(
            self.rfms_data[['TransactionId', 'CustomerId', 'Recency_WoE', 'Frequency_WoE', 'Monetary_WoE', 'RFMS_Score', 'Risk_Label']],
            how='left',
            on=['TransactionId', 'CustomerId']
        )
        return merged_data
