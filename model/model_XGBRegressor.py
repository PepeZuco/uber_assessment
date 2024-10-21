import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# Configuração do log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log: Step 1 - Loading datasets
logging.info("Loading datasets...")
train_df = pd.read_csv('data_final/train_df_cleaned_full.csv')
test_df = pd.read_csv('data_final/test_df_cleaned_full.csv')

# Log: Step 2 - Identifying numerical columns for scaling
logging.info("Identifying numerical columns for scaling...")
numerical_cols = ['pick_lat', 'pick_lng', 'dropoff_lat', 'dropoff_lng', 'eta',
                  'trip_distance', 'surge_multiplier', 'driver_rating', 'lifetime_trips',
                  'trip_distance_haversine', 'distance_from_center', 'rain_last_hour', 'rain_last_6_hours']

# Log: Step 3 - Normalizing numerical columns
logging.info("Normalizing numerical columns...")
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])

# Log: Step 4 - One-Hot Encoding categorical columns
logging.info("Applying One-Hot Encoding to categorical columns...")
categorical_cols = ['pickup_airport_code', 'dropoff_airport_code', 'pickup_day_of_the_week',
                    'pickup_hour_cat', 'rain_category']
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)

# Log: Step 5 - Handling outliers in the target (trip_duration) using IQR
logging.info("Handling outliers in the target variable...")
Q1 = train_df['trip_duration'].quantile(0.25)
Q3 = train_df['trip_duration'].quantile(0.75)
IQR = Q3 - Q1

# Removing outliers based on the 1.5*IQR rule
train_df = train_df[~((train_df['trip_duration'] < (Q1 - 1.5 * IQR)) | (train_df['trip_duration'] > (Q3 + 1.5 * IQR)))]

# Log: Step 6 - Filling missing values
logging.info("Filling missing values...")
train_df['rain_last_hour'].fillna(train_df['rain_last_hour'].median(), inplace=True)
train_df['rain_last_6_hours'].fillna(train_df['rain_last_6_hours'].median(), inplace=True)

# Log: Step 7 - Applying log transformation to the target variable (trip_duration)
logging.info("Applying log transformation to the target variable...")
train_df['log_trip_duration'] = np.log1p(train_df['trip_duration'])  # log1p avoids log(0) issues

# Log: Step 8 - Separating target and features
logging.info("Separating the log-transformed target variable and features...")
train_df.to_csv('this.csv', index=False)
X = train_df.drop(columns=['trip_duration', 'log_trip_duration', 'pickup_ts', 'dropoff_ts'])
y = train_df['log_trip_duration']

# Log: Step 9 - Splitting the dataset into training and validation sets
logging.info("Splitting dataset into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Log: Step 10 - Hyperparameter tuning using GridSearchCV with fixed parameters
logging.info("Performing hyperparameter tuning using GridSearchCV...")
param_grid = {
    'n_estimators': [250, 500, 1000],  # Adjusted number of trees
    'learning_rate': [0.1],            # Fixed learning rate
    'max_depth': [5],                  # Fixed maximum depth of trees
    'subsample': [1.0],                # Fixed subsample ratio
    'colsample_bytree': [1.0]          # Fixed column subsample ratio
}

xgb_model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)

# Log: Step 11 - Training the XGBRegressor with best parameters from GridSearch
best_params = grid_search.best_params_
logging.info(f"Best parameters from GridSearch: {best_params}")

xgb_model = XGBRegressor(**best_params, random_state=42)
xgb_model.fit(X_train, y_train)

# Log: Step 12 - Making predictions with XGBRegressor
logging.info("Making predictions with XGBRegressor...")
y_xgb_pred = xgb_model.predict(X_val)

# Reverse log transformation to get predictions in original scale
y_xgb_pred_orig = np.expm1(y_xgb_pred)  # Inverse of log1p transformation
y_val_orig = np.expm1(y_val)

# Log: Step 13 - Calculating XGBRegressor performance metrics
xgb_mae = mean_absolute_error(y_val_orig, y_xgb_pred_orig)
xgb_r2 = r2_score(y_val_orig, y_xgb_pred_orig)
logging.info(f"XGBRegressor MAE: {xgb_mae}")
logging.info(f"XGBRegressor R² Score: {xgb_r2}")

# Log: Step 14 - Generating comparison plot for XGBRegressor
logging.info("Generating comparison plot for XGBRegressor...")
plt.figure(figsize=(8, 6))
plt.scatter(y_val_orig, y_xgb_pred_orig, alpha=0.5)
plt.plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], '--', color='red')
plt.title('Previsão vs Valores Reais (XGBRegressor)')
plt.xlabel('Valores Reais (trip_duration)')
plt.ylabel('Previsões (trip_duration)')
plt.savefig(f'xgb_model_result_{best_params["n_estimators"]}_{train_df.shape[0]}_rows.png')

logging.info("Process complete!")
