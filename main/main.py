import pandas as pd
from feature_engineering import build_meal_features
from feature_engineering import get_model_feature_cols
from predicter import load_models
from feature_engineering import get_prediction_row

def predict_for(meal, df_master, target_date):
    if target_date not in set(df_master["Date"]):
        return {"status": "blocked", "reason": f"no row for {target_date.date()}"}

    cat = df_master.loc[df_master["Date"] == target_date, "Category"].iloc[0]
    if pd.isna(cat):
        return {"status": "blocked", "reason": f"calendar has no entry for {target_date.date()}"}

    meal_df = build_meal_features(df_master, meal)
    feature_cols = get_model_feature_cols(meal_df)

    try:
        X = get_prediction_row(meal_df, target_date, feature_cols)
    except ValueError as e:
        return {"status": "blocked", "reason": str(e)}

    models = load_models(meal)
    xgb = models["xgb"].predict(X)[0]
    lgbm = models["lgbm"].predict(X)[0]
    cb = models["cb"].predict(X)[0]
    ensemble = (xgb + lgbm + cb) / 3

    return {
        "target_date": str(target_date.date()),
        "meal": meal,
        "xgb": round(float(xgb), 1),
        "lgbm": round(float(lgbm), 1),
        "cb": round(float(cb), 1),
        "ensemble": round(float(ensemble), 1),
    }