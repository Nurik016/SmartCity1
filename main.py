import time
import logging
import pandas as pd  # Add this line to import pandas
import matplotlib.pyplot as plt
from joblib import dump, load
from models.preprocessing import preprocess_data
from models.training import train_model, train_stacking_model
from models.prediction import predict_usage
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Load data
    start_time = time.time()
    df = pd.read_csv("data/municipality_bus_utilization.csv")
    logging.info("Data loaded successfully.")

    # Preprocess data
    processed_df, scaler = preprocess_data(df)
    logging.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds.")

    # Load or train model
    try:
        model = load('model.joblib')
        logging.info("Model loaded from file.")
    except FileNotFoundError:
        logging.warning("Model file not found. A new model will be trained.")
        model, X_train, X_test, y_train, y_test = train_model(processed_df)
        dump(model, 'model.joblib')
        logging.info("Model training completed and saved.")

    # Example prediction
    example_timestamp = pd.to_datetime("2017-06-04 11:00:00")
    example_input = pd.DataFrame({
        'timestamp_unix': [example_timestamp.timestamp()],
        'hour': [example_timestamp.hour],
        'minute': [example_timestamp.minute],
        'dayofweek': [example_timestamp.dayofweek],
        'season': [2],
        'municipality_id': [0],
        'usage_lag_1': [0],  # Fill with recent known values
        'usage_lag_24': [0]  # Fill with recent known values
    })
    example_input = pd.get_dummies(example_input, columns=['municipality_id'], prefix='municipality')
    missing_cols = set(X_train.columns) - set(example_input.columns)
    for col in missing_cols:
        example_input[col] = 0
    example_input = example_input[X_train.columns]

    predicted_usage = predict_usage(model, example_input, scaler)
    logging.info(f"Predicted usage for {example_timestamp}: {predicted_usage[0]:.2f}")

    # Model evaluation
    predictions = predict_usage(model, X_test, scaler)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    logging.info(f"Model evaluation completed: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Real Values', color='blue')
    plt.plot(predictions, label='Predicted Values', color='red')
    plt.title('Comparison of Real and Predicted Values')
    plt.xlabel('Observation')
    plt.ylabel('Usage')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
