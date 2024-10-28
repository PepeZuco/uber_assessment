import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")


class ModelPipeline:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.scaler = StandardScaler()
        self.y_pred_test_orig = None

    def load_data(self):
        logging.info("Loading datasets...")
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

    def process_features(self):
        logging.info("Identifying numerical columns for scaling...")
        numerical_cols = ['pick_lat', 'pick_lng', 'dropoff_lat', 'dropoff_lng', 'eta',
                          'trip_distance', 'surge_multiplier', 'driver_rating', 'lifetime_trips',
                          'trip_distance_haversine', 'distance_from_center', 'direction_of_travel']

        categorical_cols = ['pickup_airport_code', 'dropoff_airport_code', 'pickup_day_of_the_week',
                            'pickup_hour_cat', 'pickup_month']

        # Creating interaction feature between 'pickup_hour' and 'pickup_day_of_the_week'
        logging.info("Creating 'hour_day_of_week_interaction' feature...")
        self.train_df['hour_day_of_week'] = self.train_df['pickup_hour'].astype(str) + "_" + self.train_df['pickup_day_of_the_week'].astype(str)
        self.test_df['hour_day_of_week'] = self.test_df['pickup_hour'].astype(str) + "_" + self.test_df['pickup_day_of_the_week'].astype(str)
        categorical_cols.append('hour_day_of_week')

        # Normalizing numerical columns
        logging.info("Normalizing numerical columns...")
        self.train_df[numerical_cols] = self.scaler.fit_transform(self.train_df[numerical_cols])
        self.test_df[numerical_cols] = self.scaler.transform(self.test_df[numerical_cols])

        # One-Hot Encoding categorical columns
        logging.info("Applying One-Hot Encoding to categorical columns...")
        self.train_df = pd.get_dummies(self.train_df, columns=categorical_cols, drop_first=True)
        self.test_df = pd.get_dummies(self.test_df, columns=categorical_cols, drop_first=True)

        # Ensure that test_df has the same columns as train_df
        self.test_df = self.test_df.reindex(columns=self.train_df.columns, fill_value=0)

    def feature_engineering(self):
        logging.info("Adding new features...")
        for df in [self.train_df, self.test_df]:
            df['trip_first_mile'] = np.minimum(df['trip_distance'], 1.0)
            df['trip_last_mile'] = np.maximum(df['trip_distance'] - 1.0, 0)
            df['rush_hour_weighted_duration'] = df['trip_duration'] * df['is_rush_hour']
            df['duration_rush_hour_interaction'] = df['trip_duration'] * df['is_rush_hour']

    def apply_log_transform(self):
        # Applying log transformation to the target variable (trip_duration)
        logging.info("Applying log transformation to the target variable...")
        self.train_df['log_trip_duration'] = np.log1p(self.train_df['trip_duration'])

    def train_model(self):
        # Splitting dataset into training and validation sets
        logging.info("Splitting dataset into training and validation sets...")
        X = self.train_df.drop(columns=['trip_duration', 'log_trip_duration', 'pickup_ts', 'dropoff_ts'])
        y = self.train_df['log_trip_duration']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the XGBRegressor model
        logging.info("Training XGBRegressor model...")
        xgb_model = XGBRegressor(n_estimators=1000, random_state=42, verbosity=1)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        # Making predictions on the validation dataset
        logging.info("Making predictions on validation dataset...")
        y_pred_val = xgb_model.predict(X_val)
        y_pred_val_orig = np.expm1(y_pred_val)  # Inverse log transformation

        # Calculating validation performance metrics
        logging.info("Calculating validation performance metrics...")
        mae_val = mean_absolute_error(np.expm1(y_val), y_pred_val_orig)
        r2_val = r2_score(np.expm1(y_val), y_pred_val_orig)
        logging.info(f"Validation MAE: {mae_val / 60:.2f} minutes, R²: {r2_val * 100:.2f}%")

        return xgb_model

    def evaluate_model(self, model):
        # Making predictions on the test dataset
        logging.info("Making predictions on test dataset...")
        X_test = self.test_df.drop(columns=['trip_duration', 'pickup_ts', 'dropoff_ts'])
        y_test = self.test_df['trip_duration']
        y_pred_test = model.predict(X_test)
        self.y_pred_test_orig = np.expm1(y_pred_test)  # Inverse log transformation

        # Calculating test performance metrics
        logging.info("Calculating test performance metrics...")
        mae_test = mean_absolute_error(y_test, self.y_pred_test_orig)
        r2_test = r2_score(y_test, self.y_pred_test_orig)
        logging.info(f"Test MAE: {mae_test / 60:.2f} minutes, R²: {r2_test * 100:.2f}%")

        # Generating comparison plots for test dataset
        logging.info("Generating comparison plots for test dataset...")

        plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')
        plt.scatter(y_test, self.y_pred_test_orig, color='white', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='green')
        plt.title('Predictions vs Actual Values - Test (XGBRegressor)', color='white')
        plt.xlabel('Actual Values (trip_duration)', color='white')
        plt.ylabel('Predictions (trip_duration)', color='white')
        plt.savefig("model/Graphs/test_predictions.png")

    def save_results(self):
        # Creating a DataFrame with actual and predicted results
        logging.info("Creating DataFrame with actual, predicted results, and difference percentage...")

        # Function to convert seconds to m:ss format
        def format_to_minutes_seconds(duration):
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}.{seconds:02d}"

        # Apply the function to both actual and predicted trip durations
        y_test_formatted = self.test_df['trip_duration'].apply(format_to_minutes_seconds)
        y_pred_test_formatted = pd.Series(self.y_pred_test_orig).apply(format_to_minutes_seconds)

        # Calculate the percentage difference between actual and predicted values
        diff_percent = ((self.test_df['trip_duration'] - self.y_pred_test_orig) / self.test_df['trip_duration'] * 100).abs()

        # Creating a DataFrame to store the actual, predicted trip durations and the percentage difference
        results_df = pd.DataFrame({
            'Actual (m:ss)': y_test_formatted,
            'Predicted (m:ss)': y_pred_test_formatted,
            'Actual (seconds)': self.test_df['trip_duration'],
            'Predicted (seconds)': self.y_pred_test_orig,
            'Diff (%)': diff_percent
        })

        # Saving the DataFrame to a CSV file for further analysis if needed
        results_df.to_csv('model/Graphs/test_predictions_with_diff_formatted.csv', index=False)
        logging.info("Process complete!")

    def save_model(self, model, model_path="xgb_model.joblib"):
        logging.info(f"Saving model to {model_path}...")
        joblib.dump(model, model_path)
        logging.info("Model saved successfully!")

    def run_pipeline(self):
        self.load_data()
        self.process_features()
        self.feature_engineering()
        self.apply_log_transform()
        model = self.train_model()
        self.evaluate_model(model)
        self.save_model(model, "model/trained_model/xgb_model.joblib")  # Save the model
        self.save_results()
