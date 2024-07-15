import pandas as pd
class MissingValueHandler:
    """
    A class for handling missing values in a dataset.
    
    Attributes:
        data (DataFrame): The input DataFrame 
    """
    
    def __init__(self, data):
        """
        Initializes the MissingValueHandler with the input DataFrame.
        
        Parameters:
            data (DataFrame): The input DataFrame.
        """
        self.__data = data
    def get_data(self):
        """
        Get the stored DataFrame containing the data.

        Returns:
            DataFrame: The DataFrame containing the data.
        """

        return self.__data

    def drop_missing(self):
        """
        Drop rows with any missing values across the entire DataFrame.
        """
        self.__data .dropna(how='all', inplace=True)

        # Drop columns where all values are missing
        self.__data .dropna(axis=1, how='any', inplace=True)
        #self.__data = self.__data.dropna(how='any').reset_index(drop=True)
        return self.__data

    def drop_columns(self, cols=None):
        """
        Drops specified columns from the DataFrame.
        
        Parameters:
            cols (list): A list of column names to drop.
        """
        self.__data = self.__data.drop(cols, axis=1, errors='ignore')
        return self.__data

    def fill_missing(self, cols, strategy='mean'):
        """
        Fills missing values in the specified columns using the specified strategy.
        
        Parameters:
            cols (list or str): The column name or list of column names in which missing values should be filled.
            strategy (str): The strategy to fill missing values.
                Options: 'mean', 'median', 'mode', 'missing'.
        """
        if isinstance(cols, str):
            cols = [cols]  # Convert to list if a single column name is passed

        for col in cols:
            if strategy == 'mean':
                self.__data[col].fillna(self.__data[col].mean(), inplace=True)
            elif strategy == 'median':
                self.__data[col].fillna(self.__data[col].median(), inplace=True)
            elif strategy == 'mode':
                self.__data[col].fillna(self.__data[col].mode().iloc[0], inplace=True)
            elif strategy == 'missing':
                self.__data[col].fillna('Missing', inplace=True)
        return self.__data