# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

#trigger a build

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.start_run() # Starting the mlflow experiment run

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the trained model")
    #parser.add_argument("--test_data", type=str, help="Path to the test dataset")
    #parser.add_argument("--predict_result", type=str, help="Output path for predictions")
    # parser.add_argument("--registered_model_name", type=str, help="Name for registering the model")
    args = parser.parse_args()

    # Load the model
    model = mlflow.sklearn.load_model(Path(args.model))
    

    print("Registering the best trained model")
    
    # Logging the model as a registered model with mlflow
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="used_cars_price_prediction_model",
        artifact_path="random_forest_price_regressor"
    )

    # End MLflow run
    mlflow.end_run()
if __name__ == "__main__":
    main()

