# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

# Required imports for training 
import mlflow
import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sys

# End any active runs before starting a new one
if mlflow.active_run() is not None:
    mlflow.end_run()

mlflow.start_run()  # Starting the mlflow experiment run

os.makedirs("./outputs", exist_ok=True)  # Create the "outputs" directory if it doesn't exist

def select_first_file(path):
    """Selects the first file in a folder, assuming there's only one file.
    Args:
        path (str): Path to the directory or file to choose.
    Returns:
        str: Full path of the selected file.
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

def main(train_data, test_data, n_estimators, max_depth, model_output):
    # Print the current working directory
    print("Current Working Directory:", os.getcwd())
    
    # Check if train_data and test_data are directories
    if os.path.isdir(train_data):
        train_file = select_first_file(train_data)
        print(f"Selected train file: {train_file}")
    else:
        train_file = train_data
        print(f"Provided train file: {train_file}")

    if os.path.isdir(test_data):
        test_file = select_first_file(test_data)
        print(f"Selected test file: {test_file}")
    else:
        test_file = test_data
        print(f"Provided test file: {test_file}")

    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Print the first few rows to verify the data
    print("First few rows of the training data:")
    print(train_df.head())
    print("First few rows of the testing data:")
    print(test_df.head())

    # Check and print column names to verify
    print("Train Data Columns:", train_df.columns)
    print("Test Data Columns:", test_df.columns)

    # Dropping the label column and assigning it to y_train
    y_train = train_df["Price"].values

    # Dropping the 'Price' column from train_df to get the features
    X_train = train_df.drop("Price", axis=1)

    # Dropping the label column and assigning it to y_test
    y_test = test_df["Price"].values

    # Dropping the 'Price' column from test_df to get the features
    X_test = test_df.drop("Price", axis=1)

    # Preprocess categorical and numerical features
    numeric_features = ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
    categorical_features = ["Segment"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OrdinalEncoder(), categorical_features)
        ]
    )

    # Create a pipeline with preprocessing and model training
    rf_model = make_pipeline(
        preprocessor,
        RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    )

    # Train the model
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Compute and log Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, rf_predictions)
    print('MSE of Random Forest Regressor on test set: {:.2f}'.format(mse))
    # Logging the MSE as a metric
    #mlflow.log_metric("MSE", float(mse))
    mlflow.log_metric("Mean Squared Error", float(mse))  # Ensure metric name matches

    
    # Output the model
    mlflow.sklearn.save_model(rf_model, model_output)
    print(f"Model saved to: {model_output}")  # Log the model output path
    mlflow.end_run()  # Ending the mlflow experiment run

if __name__ == "__main__":
    if 'ipykernel_launcher' in sys.argv[0]:
        # If running in an interactive environment, set default values for arguments
        main(
            train_data="./data/used_cars.csv",
            test_data="./data/used_cars.csv",
            n_estimators=100,
            max_depth=None,
            model_output="./outputs/model"
        )
    else:
        # If running from the command line, parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--train_data", type=str, help="Path to train data")
        parser.add_argument("--test_data", type=str, help="Path to test data")
        parser.add_argument('--n_estimators', type=int, default=100, help='The number of trees in the forest')
        parser.add_argument('--max_depth', type=int, default=None, help='The maximum depth of the trees')
        parser.add_argument("--model_output", type=str, help="Path of output model")
        args = parser.parse_args()
        main(
            train_data=args.train_data,
            test_data=args.test_data,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            model_output=args.model_output
        )
