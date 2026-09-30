import joblib
import pandas as pd
from pathlib import Path
from feature_engineering import get_model_feature_cols

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "V1"

MODEL_NAMES = {
    "Breakfast": ("breakfast", "breakfast"),
    "Lunch": ("lunch", "lunch"),
    "Dinner": ("dinner", "dinner"),
    "Evening_Snacks": ("snacks", "snacks"),
}

def load_models(meal):
    folder, prefix = MODEL_NAMES[meal]
    models = {}
    for name in ["xgb", "lgbm", "cb"]:
        path = MODEL_DIR / folder / f"{prefix}_{name}_v1.joblib"
        models[name] = joblib.load(path)
    return models

def predict_rows(meal_df, dates, models):
    """Ensemble predictions for the given dates. meal_df is the output of build_meal_features."""
    feature_cols = get_model_feature_cols(meal_df)
    rows = meal_df[meal_df["Date"].isin(dates)]
    X = rows[feature_cols]
    out = pd.DataFrame({"Date": rows["Date"].values})
    for name, model in models.items():
        out[name] = model.predict(X)
    out["ensemble"] = out[["xgb", "lgbm", "cb"]].mean(axis=1)
    return out