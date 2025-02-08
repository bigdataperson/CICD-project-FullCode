# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--train_data", type=str, help="Path to save train data")
    parser.add_argument("--test_data", type=str, help="Path to save test data")
    args = parser.parse_args()

    # Start MLflow Run
    mlflow.start_run()

    # Log arguments
    logging.info(f"Input data path: {args.data}")
    logging.info(f"Test-train ratio: {args.test_train_ratio}")

    # Read data
    df = pd.read_csv(args.data)

    # Split data
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)

    # Log completion
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("test_size", len(test_df))
    mlflow.end_run()

if __name__ == "__main__":
    main()
