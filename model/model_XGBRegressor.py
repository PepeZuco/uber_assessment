import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings
from scipy.stats import mstats

remove_outliers = True

warnings.filterwarnings("ignore")

# Configuração do log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(f"results_{'removing_outliers' if remove_outliers else ''}.txt"), logging.StreamHandler()])

# Log: Step 1 - Loading datasets
logging.info("Loading datasets...")
train_df = pd.read_csv('data_final/train_df_cleaned_full.csv')
test_df = pd.read_csv('data_final/test_df_cleaned_full.csv')

# Log: Step 2 - Identifying numerical columns for scaling
logging.info("Identifying numerical columns for scaling...")
numerical_cols = ['pick_lat', 'pick_lng', 'dropoff_lat', 'dropoff_lng', 'eta',
                  'trip_distance', 'surge_multiplier', 'driver_rating', 'lifetime_trips',
                  'trip_distance_haversine', 'distance_from_center',
                  'rain_last_hour_interlagos', 'rain_last_6_hours_interlagos',
                  'rain_last_hour_mirante', 'rain_last_6_hours_mirante']

# Log: Step 3 - Normalizing numerical columns
logging.info("Normalizing numerical columns...")
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])

# Log: Step 4 - One-Hot Encoding categorical columns
logging.info("Applying One-Hot Encoding to categorical columns...")
categorical_cols = ['pickup_airport_code', 'dropoff_airport_code', 'pickup_day_of_the_week',
                    'pickup_hour_cat', 'rain_category_interlagos', 'rain_category_mirante']
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)

if remove_outliers:
    logging.info("Handling outliers in the target variable...")
    Q1 = train_df['trip_duration'].quantile(0.25)
    Q3 = train_df['trip_duration'].quantile(0.75)
    IQR = Q3 - Q1

    # Removing outliers based on the 1.5*IQR rule
    train_df = train_df[~((train_df['trip_duration'] < (Q1 - 1.5 * IQR)) | (train_df['trip_duration'] > (Q3 + 1.5 * IQR)))]

# Log: Step 5 - Handling outliers in the target (trip_duration) using Winsorization for long trips
logging.info("Handling outliers in the target variable for long trips using Winsorization...")
long_trip_df = train_df[train_df['trip_duration'] > 1000]
long_trip_df['trip_duration'] = mstats.winsorize(long_trip_df['trip_duration'], limits=[0.05, 0.05])

# Log: Step 6 - Filling missing values
logging.info("Filling missing values...")
train_df['rain_last_hour_interlagos'].fillna(train_df['rain_last_hour_interlagos'].median(), inplace=True)
train_df['rain_last_6_hours_interlagos'].fillna(train_df['rain_last_6_hours_interlagos'].median(), inplace=True)
train_df['rain_last_hour_mirante'].fillna(train_df['rain_last_hour_mirante'].median(), inplace=True)
train_df['rain_last_6_hours_mirante'].fillna(train_df['rain_last_6_hours_mirante'].median(), inplace=True)

# Log: Step 7 - Feature Engineering (new features for long trips)
logging.info("Adding new features for long trips...")
train_df['trip_first_mile'] = np.minimum(train_df['trip_distance'], 1.0)
train_df['trip_last_mile'] = np.maximum(train_df['trip_distance'] - 1.0, 0)
train_df['rush_hour_weighted_duration'] = train_df['trip_duration'] * train_df['is_rush_hour']
train_df['duration_rush_hour_interaction'] = train_df['trip_duration'] * train_df['is_rush_hour']

# Log: Step 8 - Applying log transformation to the target variable (trip_duration)
logging.info("Applying log transformation to the target variable...")
train_df['log_trip_duration'] = np.log1p(train_df['trip_duration'])  # log1p avoids log(0) issues

# Log: Step 9 - Splitting dataset into short and long trips
logging.info("Splitting the dataset into short and long trips based on trip duration...")
short_trip_df = train_df[train_df['trip_duration'] <= 1000]  # Example threshold for short trips
long_trip_df = train_df[train_df['trip_duration'] > 1000]    # Example threshold for long trips

# Log: Step 10 - Splitting short and long trips into training and validation sets
logging.info("Splitting short and long trips into training and validation sets...")
X_train_short, X_val_short, y_train_short, y_val_short = train_test_split(
    short_trip_df.drop(columns=['trip_duration', 'log_trip_duration', 'pickup_ts', 'dropoff_ts']),
    short_trip_df['log_trip_duration'],
    test_size=0.2, random_state=42
)

X_train_long, X_val_long, y_train_long, y_val_long = train_test_split(
    long_trip_df.drop(columns=['trip_duration', 'log_trip_duration', 'pickup_ts', 'dropoff_ts']),
    long_trip_df['log_trip_duration'],
    test_size=0.2, random_state=42
)

# Log: Step 11 - Training XGBRegressor for short trips
logging.info("Training XGBRegressor for short trips...")
xgb_short = XGBRegressor(n_estimators=1000, random_state=42, verbosity=1)
xgb_short.fit(X_train_short, y_train_short, eval_set=[(X_val_short, y_val_short)], verbose=True)

# Log: Step 12 - Training XGBRegressor for long trips with fine-tuned parameters
logging.info("Training XGBRegressor for long trips with fine-tuned hyperparameters...")
xgb_long = XGBRegressor(n_estimators=2000, learning_rate=0.01, random_state=42, verbosity=1)
xgb_long.fit(X_train_long, y_train_long, eval_set=[(X_val_long, y_val_long)], verbose=True)

# Log: Step 13 - Making predictions for short and long trips
logging.info("Making predictions for short trips...")
y_pred_short = xgb_short.predict(X_val_short)
y_pred_short_orig = np.expm1(y_pred_short)  # Inverse log transformation

logging.info("Making predictions for long trips...")
y_pred_long = xgb_long.predict(X_val_long)
y_pred_long_orig = np.expm1(y_pred_long)  # Inverse log transformation

# Log: Step 14 - Calculating performance metrics for both models
logging.info("Calculating performance metrics for short trips...")
mae_short = mean_absolute_error(np.expm1(y_val_short), y_pred_short_orig)
r2_short = r2_score(np.expm1(y_val_short), y_pred_short_orig)
logging.info(f"Short Trip MAE: {mae_short}, R²: {r2_short}")

logging.info("Calculating performance metrics for long trips...")
mae_long = mean_absolute_error(np.expm1(y_val_long), y_pred_long_orig)
r2_long = r2_score(np.expm1(y_val_long), y_pred_long_orig)
logging.info(f"Long Trip MAE: {mae_long}, R²: {r2_long}")

# Log: Step 15 - Generating comparison plots for short and long trips
logging.info("Generating comparison plot for short trips...")
plt.figure(figsize=(8, 6))
plt.style.use('dark_background')  # Set the background to black
plt.scatter(np.expm1(y_val_short), y_pred_short_orig, color='white', alpha=0.5)  # White dots
plt.plot([np.expm1(y_val_short).min(), np.expm1(y_val_short).max()],
         [np.expm1(y_val_short).min(), np.expm1(y_val_short).max()], '--', color='green')  # Green line
plt.title('Previsão vs Valores Reais - Short Trips (XGBRegressor)', color='white')
plt.xlabel('Valores Reais (trip_duration)', color='white')
plt.ylabel('Previsões (trip_duration)', color='white')
plt.savefig(f"model/Graphs/short_trips_predictions_{'removing_outliers' if remove_outliers else ''}.png")

logging.info("Generating comparison plot for long trips...")
plt.figure(figsize=(8, 6))
plt.style.use('dark_background')  # Set the background to black
plt.scatter(np.expm1(y_val_long), y_pred_long_orig, color='white', alpha=0.5)  # White dots
plt.plot([np.expm1(y_val_long).min(), np.expm1(y_val_long).max()],
         [np.expm1(y_val_long).min(), np.expm1(y_val_long).max()], '--', color='green')  # Green line
plt.title('Previsão vs Valores Reais - Long Trips (XGBRegressor)', color='white')
plt.xlabel('Valores Reais (trip_duration)', color='white')
plt.ylabel('Previsões (trip_duration)', color='white')
plt.savefig(f"model/Graphs/long_trips_predictions_{'removing_outliers' if remove_outliers else ''}.png")

logging.info("Process complete!")
