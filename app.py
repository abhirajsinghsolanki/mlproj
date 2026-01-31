from flask import Flask, request, jsonify, render_template
import os
import pickle
import pandas as pd

app = Flask(__name__, template_folder="templates", static_folder="static")

DATA_DIR = "artifacts"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")


def load_artifacts():
    preprocessor = None
    model = None
    if os.path.exists(PREPROCESSOR_PATH):
        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = pickle.load(f)
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    return preprocessor, model


preprocessor, model = load_artifacts()


def get_feature_names():
    if os.path.exists(TRAIN_CSV):
        df = pd.read_csv(TRAIN_CSV, nrows=1)
        return list(df.columns)
    return []


@app.route("/")
def index():
    fields = get_feature_names()
    return render_template("index.html", fields=fields, model_available=bool(model))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    fields = get_feature_names()
    if request.method == "POST":
        data = {}
        for key in request.form:
            data[key] = request.form.get(key)

        if model is not None and preprocessor is not None:
            df = pd.DataFrame([data])
            # try to convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass
            X = preprocessor.transform(df)
            pred = model.predict(X)
            return render_template("result.html", prediction=pred[0], inputs=data)
        else:
            return render_template("result.html", prediction=None, inputs=data, message="Model not trained yet.")

    return render_template("predict.html", fields=fields, model_available=bool(model))


if __name__ == "__main__":
    app.run(debug=True)

