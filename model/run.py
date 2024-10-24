import logging
from model_pipeline import ModelPipeline

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("model/results.txt"), logging.StreamHandler()])

if __name__ == "__main__":
    # Running the pipeline
    pipeline = ModelPipeline(train_path='data_final/train_df_cleaned_full.csv',
                             test_path='data_final/test_df_cleaned_full.csv')
    pipeline.run_pipeline()
