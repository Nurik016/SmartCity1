import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

def train_model(df):
    """Trains a gradient boosting regression model."""
    X = df.drop(['usage', 'total_capacity', 'timestamp'], axis=1)
    y = df['usage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    model = GradientBoostingRegressor()
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 7, 5]
    }
    grid_search = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model, X_train, X_test, y_train, y_test

def train_stacking_model(X_train, y_train):
    """Trains a stacking ensemble model."""
    base_learners = [
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10))
    ]
    meta_learner = GradientBoostingRegressor()
    stack_model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner)
    stack_model.fit(X_train, y_train)
    return stack_model
