import pandas as pd
import numpy as np

def clean_data(df):
    """
    Performs data cleaning: drops irrelevant columns and handles missing values.
    """
    # Drop columns with no predictive value based on the analysis report
    df = df.drop(columns=['id', 'dataset', 'num'], errors='ignore')
    
    # Numeric imputation with median; Categorical with mode
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

def handle_outliers(df):
    """
    Removes outliers using the 1.5x IQR method for continuous variables.
    """
    continuous_features = ['trestbps', 'chol', 'thalch', 'oldpeak']
    for feature in continuous_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter the dataframe to remove records outside the bounds
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df