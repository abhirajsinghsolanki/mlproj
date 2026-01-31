from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer

def run_training():
    try:
        # Step 1: Ingest Data
        # This creates train.csv, test.csv, and the artifacts folder
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        # Step 2: Transform Data
        # This creates preprocessor.pkl and returns cleaned arrays
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Step 3: Train Model
        # This creates model.pkl and returns the accuracy score
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"Success! Model trained with R2 Score: {r2_score}")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    run_training()