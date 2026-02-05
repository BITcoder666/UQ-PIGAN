# data_preprocessing.py

import pandas as pd
import torch
import numpy as np
import os


class DataPreprocessor:
    """Data Preprocessor Class for normalization and missing value handling."""

    def __init__(self, min_max_path='min_max_values.csv'):
        """
        Initialize the preprocessor.

        Args:
            min_max_path: Path to CSV file for saving/loading min-max values.
        """
        self.min_max_path = min_max_path
        self.min_values = None
        self.max_values = None
        self.median_values = None  # Stores median values for NaN filling
        self.numerical_features = None

    def fit(self, train_data, test_data):
        """
        Fit the preprocessor, calculate and save normalization parameters.
        Important: Statistics are calculated using ONLY the training set to avoid data leakage.

        Args:
            train_data: Training set DataFrame.
            test_data: Test set DataFrame.

        Returns:
            Tuple of tensors: (train_features, train_labels, test_features, test_labels)
        """
        # ===== Use only training features for statistics =====
        train_features = train_data.iloc[:, :-1]
        test_features = test_data.iloc[:, :-1]

        # Identify numerical features
        self.numerical_features = train_features.dtypes[train_features.dtypes != 'object'].index

        # ===== Check for NaNs =====
        nan_counts = train_features[self.numerical_features].isnull().sum()
        if nan_counts.sum() > 0:
            print("NaN detected in training set:")
            print(nan_counts[nan_counts > 0])

        # ===== Calculate Median (Train Set Only) =====
        self.median_values = train_features[self.numerical_features].median(skipna=True)

        # Check if any column is all NaN
        nan_medians = self.median_values.isnull()
        if nan_medians.any():
            print(f"Warning: Columns {list(nan_medians[nan_medians].index)} are all NaN. Filling with 0.")
            self.median_values.fillna(0, inplace=True)

        # ===== Fill NaNs =====
        # Fill training set
        train_features_filled = train_features.copy()
        train_features_filled[self.numerical_features] = train_features_filled[
            self.numerical_features
        ].fillna(self.median_values)

        # Fill test set (using training set medians)
        test_features_filled = test_features.copy()
        test_features_filled[self.numerical_features] = test_features_filled[
            self.numerical_features
        ].fillna(self.median_values)

        # ===== Calculate Min/Max (Based on filled training set) =====
        self.min_values = train_features_filled[self.numerical_features].min(axis=0)
        self.max_values = train_features_filled[self.numerical_features].max(axis=0)

        # ===== Save Parameters =====
        params_df = pd.DataFrame({
            'min': self.min_values,
            'max': self.max_values,
            'median': self.median_values
        })
        params_df.to_csv(self.min_max_path, index=True, encoding='gbk')
        print(f"Normalization parameters saved to: {self.min_max_path}")

        # ===== Min-Max Normalization to [0, 1] =====
        train_features_norm = pd.DataFrame(index=train_features_filled.index)
        test_features_norm = pd.DataFrame(index=test_features_filled.index)

        for col in self.numerical_features:
            min_val = self.min_values[col]
            max_val = self.max_values[col]
            denominator = max_val - min_val

            if denominator < 1e-8:
                # Avoid division by zero for constant columns
                train_features_norm[col] = train_features_filled[col]
                test_features_norm[col] = test_features_filled[col]
            else:
                train_features_norm[col] = (train_features_filled[col] - min_val) / denominator
                test_features_norm[col] = (test_features_filled[col] - min_val) / denominator

        # ===== Final NaN Check =====
        if train_features_norm.isnull().sum().sum() > 0:
            print("Warning: NaN detected in training set after normalization!")
        if test_features_norm.isnull().sum().sum() > 0:
            print("Warning: NaN detected in test set after normalization!")

        # Convert to tensors
        train_features_tensor = torch.tensor(train_features_norm.values, dtype=torch.float)
        test_features_tensor = torch.tensor(test_features_norm.values, dtype=torch.float)
        train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float).view(-1, 1)
        test_labels = torch.tensor(test_data.iloc[:, -1].values, dtype=torch.float).view(-1, 1)

        return train_features_tensor, train_labels, test_features_tensor, test_labels

    def load_min_max(self):
        """
        Load saved normalization parameters.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not os.path.exists(self.min_max_path):
            print(f"Warning: Normalization parameter file {self.min_max_path} not found.")
            return False

        params_df = pd.read_csv(self.min_max_path, index_col=0, encoding='gbk')
        self.min_values = params_df['min']
        self.max_values = params_df['max']

        # Compatibility for older versions without 'median'
        if 'median' in params_df.columns:
            self.median_values = params_df['median']
        else:
            self.median_values = pd.Series(0, index=self.min_values.index)

        self.numerical_features = self.min_values.index
        print(f"Loaded normalization parameters: {self.min_max_path}")
        return True

    def transform(self, data):
        """
        Transform new data using saved parameters.

        Args:
            data: DataFrame to transform.

        Returns:
            tensor: Transformed feature tensor.
        """
        if self.min_values is None or self.max_values is None:
            raise ValueError("Please call fit() or load_min_max() first.")

        # Extract features (remove label column if present)
        if '侵彻深度' in data.columns:
            features = data.iloc[:, :-1].copy()
        else:
            features = data.copy()

        # Fill NaNs with saved medians
        features[self.numerical_features] = features[self.numerical_features].fillna(self.median_values)

        # Fallback fill for safety
        if features[self.numerical_features].isnull().any().any():
            features[self.numerical_features] = features[self.numerical_features].fillna(0)

        # Normalize
        features_norm = pd.DataFrame(index=features.index)
        for col in self.numerical_features:
            min_val = self.min_values[col]
            max_val = self.max_values[col]
            denominator = max_val - min_val

            if denominator < 1e-8:
                features_norm[col] = features[col]
            else:
                features_norm[col] = (features[col] - min_val) / denominator

        return torch.tensor(features_norm.values, dtype=torch.float)

    def inverse_transform(self, normalized_value, feature_name='侵彻深度'):
        """
        Inverse transform normalized values to original scale.

        Args:
            normalized_value: Normalized value.
            feature_name: Name of the feature.

        Returns:
            Original value.
        """
        if self.min_values is None or self.max_values is None:
            raise ValueError("Please call fit() or load_min_max() first.")

        min_val = self.min_values[feature_name]
        max_val = self.max_values[feature_name]
        return normalized_value * (max_val - min_val) + min_val


def load_data(train_path, test_path, encoding='gbk'):
    """
    Load CSV data files.

    Args:
        train_path: Path to training CSV.
        test_path: Path to test CSV.
        encoding: File encoding.

    Returns:
        train_data, test_data: Pandas DataFrames.
    """
    train_data = pd.read_csv(train_path, encoding=encoding, dtype={'弹体长度': float})
    test_data = pd.read_csv(test_path, encoding=encoding, dtype={'弹体长度': float})

    print(f"Train set size: {train_data.shape}")
    print(f"Test set size: {test_data.shape}")

    return train_data, test_data


def create_data_loaders(train_features, train_labels, test_features, test_labels,
                        train_batch_size=None, test_batch_size=1, shuffle_train=True):
    """
    Create PyTorch DataLoaders.

    Args:
        train_features: Training feature tensor.
        train_labels: Training label tensor.
        test_features: Test feature tensor.
        test_labels: Test label tensor.
        train_batch_size: Batch size for training (default: all).
        test_batch_size: Batch size for testing.
        shuffle_train: Whether to shuffle training data.

    Returns:
        train_loader, test_loader
    """
    if train_batch_size is None:
        train_batch_size = len(train_features)

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=shuffle_train
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader


def preprocess_pipeline(train_path, test_path, min_max_path='min_max_values.csv',
                        train_batch_size=None, test_batch_size=1):
    """
    Full data preprocessing pipeline.

    Returns:
        train_loader, test_loader, preprocessor
    """
    # 1. Load Data
    train_data, test_data = load_data(train_path, test_path)

    # 2. Init and Fit Preprocessor
    preprocessor = DataPreprocessor(min_max_path)
    train_features, train_labels, test_features, test_labels = preprocessor.fit(train_data, test_data)

    # 3. Create Loaders
    train_loader, test_loader = create_data_loaders(
        train_features, train_labels, test_features, test_labels,
        train_batch_size, test_batch_size
    )

    print(f"Data preprocessing completed.")
    return train_loader, test_loader, preprocessor


if __name__ == "__main__":
    # Test execution
    train_path = './train1/train_data1.csv'
    test_path = './train1/test_data1.csv'

    train_loader, test_loader, preprocessor = preprocess_pipeline(
        train_path, test_path, train_batch_size=32
    )
