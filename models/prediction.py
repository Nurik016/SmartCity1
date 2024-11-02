def predict_usage(model, input_features, scaler):
    """Predicts bus usage."""
    numerical_features = ['timestamp_unix', 'hour', 'minute', 'dayofweek', 'season', 'usage_lag_1', 'usage_lag_24']
    input_features = input_features.copy()
    input_features[numerical_features] = scaler.transform(input_features[numerical_features])
    return model.predict(input_features)
