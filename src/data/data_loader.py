"""Data loading utilities for FraudGuard."""

import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

from ..utils import logger, settings


def load_credit_card_fraud_data(data_path: Path | None = None) -> pd.DataFrame:
    """
    Load credit card fraud dataset.
    
    Dataset: Kaggle Credit Card Fraud Detection
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
    Args:
        data_path: Path to CSV file. If None, uses settings.data_path/creditcard.csv
        
    Returns:
        DataFrame with fraud transactions
    """
    if data_path is None:
        data_path = settings.data_path / "raw" / "creditcard.csv"
    
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df):,} transactions")
        logger.info(f"Fraud rate: {df['Class'].mean():.4%}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        raise


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_column: str = "Class"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Fraction for test set
        val_size: Fraction of train set for validation
        random_state: Random seed
        stratify_column: Column to stratify on
        
    Returns:
        Tuple of (train, val, test) DataFrames
    """
    logger.info("Splitting data...")
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_column]
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val[stratify_column]
    )
    
    logger.info(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    logger.info(f"Fraud - Train: {train['Class'].mean():.4%} | Val: {val['Class'].mean():.4%} | Test: {test['Class'].mean():.4%}")
    
    return train, val, test


def get_feature_target_split(
    df: pd.DataFrame,
    target_column: str = "Class",
    drop_columns: list[str] | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        drop_columns: Additional columns to drop
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    drop_cols = [target_column]
    if drop_columns:
        drop_cols.extend(drop_columns)
    
    X = df.drop(columns=drop_cols)
    y = df[target_column]
    
    return X, y
