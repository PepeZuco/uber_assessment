import logging
from data_processing import DataProcessor

if __name__ == "__main__":
    # Configurar o logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Criar uma instância da classe DataProcessor
    processor = DataProcessor(df_type='test')

    # Processar os dados e salvar o arquivo final
    processor.process_data()
