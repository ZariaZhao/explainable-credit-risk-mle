import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_and_split_data(filepath: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data, rename columns, and split into training and testing sets.
    """
    df = pd.read_csv(filepath)
    
    # Rename target column to standard naming convention
    if 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'target'})
    
    # Remove ID column as it has no predictive value
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    print(f"Data loaded. Shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Split data with stratification to maintain target distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test