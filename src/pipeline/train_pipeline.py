# train_pipeline.py
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer

if __name__ == "__main__":
    # 1. Ingest Data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. Transform Data
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

    # 3. Train Model
    trainer = ModelTrainer()
    print(f"Model R2 Score: {trainer.initiate_model_trainer(train_arr, test_arr)}")