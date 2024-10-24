import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("model/results.txt"), logging.StreamHandler()])

# Step 1: Loading datasets
logging.info("Loading datasets...")
train_df = pd.read_csv('data_final/train_df_cleaned_full.csv')
test_df = pd.read_csv('data_final/test_df_cleaned_full.csv')

# Step 2: Identifying numerical columns for scaling
logging.info("Identifying numerical columns for scaling...")
numerical_cols = ['pick_lat', 'pick_lng', 'dropoff_lat', 'dropoff_lng', 'eta',
                  'trip_distance', 'surge_multiplier', 'driver_rating', 'lifetime_trips',
                  'trip_distance_haversine', 'distance_from_center']

# Step 3: Normalizing numerical columns
logging.info("Normalizing numerical columns...")
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])  # Applying the same scaling to the test set

# Step 4: One-Hot Encoding categorical columns
logging.info("Applying One-Hot Encoding to categorical columns...")
categorical_cols = ['pickup_airport_code', 'dropoff_airport_code', 'pickup_day_of_the_week',
                    'pickup_hour_cat']
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Ensure that test_df has the same columns as train_df
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# Step 5: Feature Engineering (adding new features)
logging.info("Adding new features...")
train_df['trip_first_mile'] = np.minimum(train_df['trip_distance'], 1.0)
train_df['trip_last_mile'] = np.maximum(train_df['trip_distance'] - 1.0, 0)
train_df['rush_hour_weighted_duration'] = train_df['trip_duration'] * train_df['is_rush_hour']
train_df['duration_rush_hour_interaction'] = train_df['trip_duration'] * train_df['is_rush_hour']

test_df['trip_first_mile'] = np.minimum(test_df['trip_distance'], 1.0)
test_df['trip_last_mile'] = np.maximum(test_df['trip_distance'] - 1.0, 0)
test_df['rush_hour_weighted_duration'] = test_df['trip_duration'] * test_df['is_rush_hour']
test_df['duration_rush_hour_interaction'] = test_df['trip_duration'] * test_df['is_rush_hour']

# Step 6: Applying log transformation to the target variable (trip_duration)
logging.info("Applying log transformation to the target variable...")
train_df['log_trip_duration'] = np.log1p(train_df['trip_duration'])  # Log-transform to handle skewness

# Step 7: Splitting dataset into training and validation sets
logging.info("Splitting dataset into training and validation sets...")
X = train_df.drop(columns=['trip_duration', 'log_trip_duration', 'pickup_ts', 'dropoff_ts'])
y = train_df['log_trip_duration']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Training the XGBRegressor model
logging.info("Training XGBRegressor model...")
xgb_model = XGBRegressor(n_estimators=1000, random_state=42, verbosity=1)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Step 9: Making predictions on the validation dataset
logging.info("Making predictions on validation dataset...")
y_pred_val = xgb_model.predict(X_val)
y_pred_val_orig = np.expm1(y_pred_val)  # Inverse log transformation

# Step 10: Calculating validation performance metrics
logging.info("Calculating validation performance metrics...")
mae_val = mean_absolute_error(np.expm1(y_val), y_pred_val_orig)
r2_val = r2_score(np.expm1(y_val), y_pred_val_orig)
logging.info(f"Validation MAE: {mae_val / 60:.2f} minutes, R²: {r2_val * 100:.2f}%")

# Step 11: Making predictions on the test dataset
logging.info("Making predictions on test dataset...")
X_test = test_df.drop(columns=['trip_duration', 'pickup_ts', 'dropoff_ts'])
y_test = test_df['trip_duration']
y_pred_test = xgb_model.predict(X_test)
y_pred_test_orig = np.expm1(y_pred_test)  # Inverse log transformation

# Step 12: Calculating test performance metrics
logging.info("Calculating test performance metrics...")
mae_test = mean_absolute_error(y_test, y_pred_test_orig)
r2_test = r2_score(y_test, y_pred_test_orig)
logging.info(f"Test MAE: {mae_test / 60:.2f} minutes, R²: {r2_test * 100:.2f}%")

# Step 13: Generating comparison plots for test dataset
logging.info("Generating comparison plots for test dataset...")

plt.figure(figsize=(8, 6))
plt.style.use('dark_background')
plt.scatter(y_test, y_pred_test_orig, color='white', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='green')
plt.title('Predictions vs Actual Values - Test (XGBRegressor)', color='white')
plt.xlabel('Actual Values (trip_duration)', color='white')
plt.ylabel('Predictions (trip_duration)', color='white')
plt.savefig("model/Graphs/test_predictions.png")

# Step 14: Creating a DataFrame with actual and predicted results
logging.info("Creating DataFrame with actual, predicted results, and difference percentage...")


# Function to convert seconds to m:ss format
def format_to_minutes_seconds(duration):
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    return f"{minutes}.{seconds:02d}"


# Apply the function to both actual and predicted trip durations
y_test_formatted = y_test.apply(format_to_minutes_seconds)
y_pred_test_formatted = pd.Series(y_pred_test_orig).apply(format_to_minutes_seconds)

# Calculate the percentage difference between actual and predicted values
diff_percent = ((y_test - y_pred_test_orig) / y_test * 100).abs()

# Creating a DataFrame to store the actual, predicted trip durations and the percentage difference
results_df = pd.DataFrame({
    'Actual (m:ss)': y_test_formatted,
    'Predicted (m:ss)': y_pred_test_formatted,
    'Actual (seconds)': y_test,
    'Predicted (seconds)': y_pred_test_orig,
    'Diff (%)': diff_percent
})

# Saving the DataFrame to a CSV file for further analysis if needed
results_df.to_csv('model/Graphs/test_predictions_with_diff_formatted.csv', index=False)
logging.info("Process complete!")
