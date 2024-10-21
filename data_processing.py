import pandas as pd
import numpy as np
import warnings
import logging

warnings.filterwarnings("ignore")


class DataProcessor:
    def __init__(self, df_type: str, generate_sample: bool = False, sample_amount: int = None):
        self.data_path = f'data_original/latam_aa_{df_type}_data_mlops.csv'
        self.holiday_path = 'data_extra/feriados_cidade_de_sao_paulo_2022-2023.xlsx'
        self.weather_2022_path = 'data_extra/dados_meteorologicos_sp_2022.xlsx'
        self.weather_2023_path = 'data_extra/dados_meteorologicos_sp_2023.xlsx'
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

        def is_rush_hour(hour):
            return 1 if (6 <= hour <= 9) or (17 <= hour <= 20) else 0
        self.df['is_rush_hour'] = self.df['pickup_hour'].apply(is_rush_hour)

    def merge_weather_data(self):
        logging.info("Merging with weather data...")

        # Get the minimum and maximum dates from the main dataset
        min_date = self.df['pickup_ts'].min() - pd.DateOffset(hours=3)
        max_date = self.df['pickup_ts'].max() - pd.DateOffset(hours=3)

        # Load the weather data
        weather_2022_df = pd.read_excel(self.weather_2022_path)
        weather_2023_df = pd.read_excel(self.weather_2023_path)

        # Convert and filter weather data based on the min/max dates from the main dataset
        weather_2022_df['Hora UTC'] = weather_2022_df['Hora UTC'].str[:2].astype(int)
        weather_2023_df['Hora UTC'] = weather_2023_df['Hora UTC'].str[:2].astype(int)

        weather_2022_df['Hora UTC'] = pd.to_timedelta(weather_2022_df['Hora UTC'], unit='h')
        weather_2023_df['Hora UTC'] = pd.to_timedelta(weather_2023_df['Hora UTC'], unit='h')

        weather_2022_df['datetime_utc'] = weather_2022_df['Data'] + weather_2022_df['Hora UTC']
        weather_2023_df['datetime_utc'] = weather_2023_df['Data'] + weather_2023_df['Hora UTC']

        # Filter weather data to only include rows between min_date and max_date
        weather_2022_filtered = weather_2022_df[(weather_2022_df['datetime_utc'] >= min_date) & (weather_2022_df['datetime_utc'] <= max_date)]
        weather_2023_filtered = weather_2023_df[(weather_2023_df['datetime_utc'] >= min_date) & (weather_2023_df['datetime_utc'] <= max_date)]

        # Concatenate filtered weather data
        weather_combined_df = pd.concat([weather_2022_filtered, weather_2023_filtered], ignore_index=True)

        # Create the 'date_utc' column for merging
        weather_combined_df['date_utc'] = weather_combined_df['datetime_utc'].dt.floor('H')

        # Convert the pickup timestamps to UTC and floor them to the nearest hour
        self.df['pickup_ts_utc'] = self.df['pickup_ts'] - pd.DateOffset(hours=3)
        self.df['date_utc'] = self.df['pickup_ts_utc'].dt.floor('H')

        # Merge the main dataset with the filtered weather data
        self.df = pd.merge(self.df, weather_combined_df, on='date_utc', how='left')

    def process_rain_features(self):
        logging.info("Categorizing rain intensity...")

        def categorize_rain(precipitation):
            if precipitation == 0:
                return 'No rain'
            elif 0 < precipitation <= 2.5:
                return 'Light rain'
            elif 2.5 < precipitation <= 7.6:
                return 'Moderate rain'
            else:
                return 'Heavy rain'

        self.df['rain_category'] = self.df['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].apply(categorize_rain)

        logging.info("Calculating rain accumulation and intensity features...")
        self.df['rain_last_hour'] = self.df['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].rolling(window=2, min_periods=1).sum()
        self.df['rain_last_6_hours'] = self.df['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].rolling(window=6, min_periods=1).sum()

    def clean_and_save(self):
        logging.info("Saving final dataset to CSV...")

        self.df = self.df[['pickup_ts', 'pick_lat', 'pick_lng',
                           'dropoff_ts', 'dropoff_lat', 'dropoff_lng',
                           'eta', 'trip_distance', 'trip_duration',
                           'is_airport', 'pickup_airport_code', 'dropoff_airport_code',
                           'is_surged', 'surge_multiplier', 'driver_rating', 'lifetime_trips',
                           'pickup_hour', 'pickup_day_of_the_week', 'pickup_hour_cat', 'is_weekend',
                           'pickup_outside_sp', 'dropoff_outside_sp', 'hour_weekend_interaction',
                           'is_holiday', 'trip_distance_haversine', 'distance_from_center',
                           'is_rush_hour', 'rain_category', 'rain_last_hour',
                           'rain_last_6_hours']]

        self.df.to_csv(self.output_path, index=False)

    def process_data(self):
        """
        Este método orquestra todo o processo de processamento de dados e salva o resultado final.
        """
        # Carregar os dados brutos
        self.load_data()

        # Processar recursos de tempo (data e hora)
        self.process_time_features()

        # Preencher valores faltantes para os códigos de aeroportos
        self.process_airport_codes()

        # Processar a coluna de avaliação do motorista
        self.process_driver_rating()

        # Aplicar flags de fim de semana e limites geográficos
        self.process_flags()

        # Aplicar a flag de feriado com base no dataset de feriados
        self.process_holidays()

        # Aplicar distancia de haversine
        self.process_haversine_distance()

        # Aplicar a flag de horário de pico
        self.process_rush_hour()

        # Mesclar dados meteorológicos (2022 e 2023)
        self.merge_weather_data()

        # Processar categorias de chuva e outras variáveis relacionadas ao clima
        self.process_rain_features()

        # Limpar o dataset e salvar o resultado final em um arquivo CSV
        self.clean_and_save()

        logging.info(f"Data processing completed. Output saved to {self.output_path}")
