import pandas as pd
import numpy as np
import warnings
import logging

warnings.filterwarnings("ignore")


class DataProcessor:
    def __init__(self, df_type: str, generate_sample: bool = False, sample_amount: int = None):
        self.data_path = f'data_original/latam_aa_{df_type}_data_mlops.csv'
        self.holiday_path = 'data_extra/feriados_cidade_de_sao_paulo_2022-2023.xlsx'
        self.df_type = df_type
        self.df = None
        self.generate_sample = generate_sample
        self.sample_amount = sample_amount

        # Validation for generate_sample and sample_amount
        if self.generate_sample:
            if not isinstance(self.sample_amount, int) or self.sample_amount <= 0:
                raise ValueError("When generate_sample is True, sample_amount must be a positive integer.")

        self.output_path = f'data_final/{df_type}_df_cleaned_{str(self.sample_amount) if self.sample_amount else "full"}.csv'

        self.SP_LAT_MIN = -23.68
        self.SP_LAT_MAX = -23.35
        self.SP_LNG_MIN = -46.83
        self.SP_LNG_MAX = -46.40

        self.CITY_CENTER_LAT = -23.55052
        self.CITY_CENTER_LNG = -46.633308

    def load_data(self):
        logging.info(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        if self.generate_sample:
            logging.info(f"Generating a sample of {self.sample_amount} rows from the dataset...")
            self.df = self.df.sample(self.sample_amount)

    def process_time_features(self):
        logging.info("Processing time features...")
        self.df['pickup_ts'] = pd.to_datetime(self.df['pickup_ts'])
        self.df['pickup_hour'] = self.df['pickup_ts'].dt.hour
        self.df['pickup_day_of_the_week'] = self.df['pickup_ts'].dt.day_name()
        self.df['pickup_month'] = self.df['pickup_ts'].dt.month

        # pickup_hour into categories (e.g., morning, afternoon, evening, night)
        def categorize_hour(hour):
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'
        self.df['pickup_hour_cat'] = self.df['pickup_hour'].apply(categorize_hour)

    def process_lat_long_data(self):
        logging.info("Processing Latitude and Longitude data")

        self.df = self.df[(self.df['pick_lat'] < 90) & (self.df['dropoff_lat'] < 90) &
                          (self.df['pick_lat'] > - 90) & (self.df['dropoff_lat'] > - 90) &
                          (self.df['pick_lng'] < 180) & (self.df['dropoff_lng'] < 180) &
                          (self.df['pick_lng'] > - 180) & (self.df['dropoff_lng'] > - 180)]

    def process_airport_codes(self):
        logging.info("Filling missing values for airport codes...")
        self.df['pickup_airport_code'] = self.df['pickup_airport_code'].fillna('Unknown').astype('category')
        self.df['dropoff_airport_code'] = self.df['dropoff_airport_code'].fillna('Unknown').astype('category')

    def process_driver_rating(self):
        logging.info("Filtering invalid 'driver_rating' values...")
        self.df = self.df[self.df['driver_rating'] != '\\N']
        logging.info("Converting 'driver_rating' to float...")
        self.df['driver_rating'] = self.df['driver_rating'].astype(float)

    def process_flags(self):
        logging.info("Applying 'is_weekend' and geographic boundary flags...")

        def is_outside_sao_paulo(lat, lng):
            return 1 if lat < self.SP_LAT_MIN or lat > self.SP_LAT_MAX or lng < self.SP_LNG_MIN or lng > self.SP_LNG_MAX else 0

        self.df['is_weekend'] = self.df['pickup_day_of_the_week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
        self.df['pickup_outside_sp'] = self.df.apply(lambda row: is_outside_sao_paulo(row['pick_lat'], row['pick_lng']), axis=1)
        self.df['dropoff_outside_sp'] = self.df.apply(lambda row: is_outside_sao_paulo(row['dropoff_lat'], row['dropoff_lng']), axis=1)
        # Interaction between pickup_hour and is_weekend
        self.df['hour_weekend_interaction'] = self.df['pickup_hour'] * self.df['is_weekend']

    def process_holidays(self):
        logging.info("Loading holiday data and applying 'is_holiday'...")
        holidays_df = pd.read_excel(self.holiday_path)
        holidays_df['date'] = pd.to_datetime(holidays_df['Data'])
        self.df['pickup_date'] = pd.to_datetime(self.df['pickup_ts']).dt.date
        self.df['is_holiday'] = self.df['pickup_date'].isin(holidays_df['date'].dt.date).astype(int)

    def process_brearing(self):
        logging.info("Adding brearing direction...")

        def calculate_bearing(lat1, lon1, lat2, lon2):
            """
            Calculate the bearing between two points on the Earth (specified in decimal degrees).
            """
            # Convert latitude and longitude from degrees to radians
            lat1 = np.radians(lat1)
            lon1 = np.radians(lon1)
            lat2 = np.radians(lat2)
            lon2 = np.radians(lon2)

            # Difference in the longitudes
            dlon = lon2 - lon1

            # Bearing calculation
            x = np.sin(dlon) * np.cos(lat2)
            y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

            initial_bearing = np.arctan2(x, y)

            # Convert from radians to degrees and normalize to 0-360
            initial_bearing = np.degrees(initial_bearing)
            compass_bearing = (initial_bearing + 360) % 360

            return compass_bearing

        self.df['direction_of_travel'] = self.df.apply(
            lambda row: calculate_bearing(row['pick_lat'], row['pick_lng'], row['dropoff_lat'], row['dropoff_lng']), axis=1)

    def process_haversine_distance(self):
        logging.info("Adding haversine distance...")

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Radius of the earth in kilometers
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) *
                 np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2))
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = R * c
            return distance

        self.df['trip_distance_haversine'] = haversine(self.df['pick_lat'], self.df['pick_lng'],
                                                       self.df['dropoff_lat'], self.df['dropoff_lng'])
        # Calculate distance from city center
        self.df['distance_from_center'] = self.df.apply(lambda row: haversine(row['pick_lat'], row['pick_lng'],
                                                                              self.CITY_CENTER_LAT, self.CITY_CENTER_LNG), axis=1)

    def process_rush_hour(self):
        logging.info("Applying 'is_rush_hour'...")

        def is_morning_rush(hour):
            return 1 if 6 <= hour <= 9 else 0

        def is_evening_rush(hour):
            return 1 if 16 <= hour <= 19 else 0

        # Step 4: Apply the new features
        self.df['is_morning_rush_hour'] = self.df['pickup_hour'].apply(is_morning_rush)
        self.df['is_evening_rush_hour'] = self.df['pickup_hour'].apply(is_evening_rush)

        # Step 5: (Optional) Keep original rush hour feature if needed
        def is_rush_hour(hour):
            return 1 if (6 <= hour <= 9) or (17 <= hour <= 20) else 0
        self.df['is_rush_hour'] = self.df['pickup_hour'].apply(is_rush_hour)

        logging.info("Rush hour features applied: 'is_morning_rush_hour', 'is_evening_rush_hour' and 'is_rush_hour'.")

    def clean_and_save(self):
        logging.info("Saving final dataset to CSV...")

        self.df = self.df[['pickup_ts', 'pick_lat', 'pick_lng',
                           'dropoff_ts', 'dropoff_lat', 'dropoff_lng',
                           'eta', 'trip_distance', 'trip_duration',
                           'is_airport', 'pickup_airport_code', 'dropoff_airport_code',
                           'is_surged', 'surge_multiplier', 'driver_rating', 'lifetime_trips',
                           'pickup_hour', 'pickup_day_of_the_week', 'pickup_hour_cat', 'is_weekend', 'pickup_month',
                           'pickup_outside_sp', 'dropoff_outside_sp', 'hour_weekend_interaction',
                           'is_holiday', 'trip_distance_haversine', 'direction_of_travel',
                           'distance_from_center', 'is_rush_hour', 'is_morning_rush_hour', 'is_evening_rush_hour'
                           ]]

        self.df.to_csv(self.output_path, index=False)

    def process_data(self):
        """
        Este método orquestra todo o processo de processamento de dados e salva o resultado final.
        """
        # Carregar os dados brutos
        self.load_data()

        # Processar recursos de tempo (data e hora)
        self.process_time_features()

        # Ajustar pontos de latitude e longitude
        self.process_lat_long_data()

        # Preencher valores faltantes para os códigos de aeroportos
        self.process_airport_codes()

        # Processar a coluna de avaliação do motorista
        self.process_driver_rating()

        # Aplicar flags de fim de semana e limites geográficos
        self.process_flags()

        # Aplicar a flag de feriado com base no dataset de feriados
        self.process_holidays()

        # Aplicar bearing direction
        self.process_brearing()

        # Aplicar distancia de haversine
        self.process_haversine_distance()

        # Aplicar a flag de horário de pico
        self.process_rush_hour()

        # Limpar o dataset e salvar o resultado final em um arquivo CSV
        self.clean_and_save()

        logging.info(f"Data processing completed. Output saved to {self.output_path}")
