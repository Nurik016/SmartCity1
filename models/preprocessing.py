import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Preprocesses the bus utilization data."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['season'] = df['timestamp'].dt.month % 12 // 3 + 1  # 1: весна, 2: лето, 3: осень, 4: зима
    df['timestamp_unix'] = df['timestamp'].astype('int64') // 1e9

    # Lag features (e.g., usage 1 hour ago)
    df['usage_lag_1'] = df['usage'].shift(1)
    df['usage_lag_24'] = df['usage'].shift(24)

    df = pd.get_dummies(df, columns=['municipality_id'], prefix='municipality')

    # Fill NA values created by lag features
    df.fillna(0, inplace=True)

    # Scaling numerical features
    numerical_features = ['timestamp_unix', 'hour', 'minute', 'dayofweek', 'season', 'usage_lag_1', 'usage_lag_24']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, scaler