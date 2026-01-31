import os
import pandas as pd
from src.utils import load_object
from src.exception import CustomException
import sys

ARTIFACT_DIR = os.path.join(os.getcwd(), 'artifacts')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pkl')
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, 'preprocessor.pkl')


def predict(input_df: pd.DataFrame):
    """Load preprocessor + model and predict on input dataframe.

    Args:
        input_df: pd.DataFrame with the same columns as training features.

    Returns:
        numpy array of predictions
    """
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError('Model or preprocessor artifact not found in artifacts/')

        preprocessor = load_object(PREPROCESSOR_PATH)
        model = load_object(MODEL_PATH)

        # Try to coerce numeric columns
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except Exception:
                pass

        X = preprocessor.transform(input_df)
        preds = model.predict(X)
        return preds

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    # simple CLI to test prediction: read CSV from argv[1]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Path to CSV file containing input rows')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(predict(df))