"""
Permanent, single-entry-point regression test for the whole pipeline.
Run this any time feature_engineering.py, predicter.py, or the saved
models change. Two independent stages -- if Stage 1 fails, Stage 2's
results can't be trusted either, since predictions are built on features.
"""

import pandas as pd
import numpy as np
from feature_engineering import build_master, build_meal_features
from predicter import load_models, predict_rows

DATA = "../data/processed"
MEALS = ["Breakfast", "Lunch", "Evening_Snacks", "Dinner"]

EXPECTED_MAE = {
    "Dinner":          {"xgb": 38.14, "lgbm": 37.45, "cb": 37.01, "ensemble": 37.10},
    "Lunch":           {"xgb": 53.70, "lgbm": 52.91, "cb": 55.71, "ensemble": 53.66},
    "Breakfast":       {"xgb": 20.56, "lgbm": 20.55, "cb": 22.21, "ensemble": 20.66},
    "Evening_Snacks":  {"xgb": 22.97, "lgbm": 22.29, "cb": 24.54, "ensemble": 22.38},
}
TOLERANCE = 0.10


def stage_1_features():
    print("=" * 50)
    print("STAGE 1: feature_engineering.py")
    print("=" * 50)

    df_raw = pd.read_csv(f"{DATA}/2024-2025.csv", parse_dates=["Date"])
    df_calendar_raw = pd.read_excel(f"{DATA}/academic_calendar_2426.xlsx")
    df_master = build_master(df_raw, df_calendar_raw)

    passed = True
    for meal in MEALS:
        print(f"--- {meal} ---")
        rebuilt = build_meal_features(df_master, meal)
        existing = pd.read_csv(f"{DATA}/{meal}_feature_2024_2025.csv", parse_dates=["Date"])
        rebuilt_slice = rebuilt[rebuilt["Date"] <= existing["Date"].max()].reset_index(drop=True)
        existing = existing.reset_index(drop=True)

        if rebuilt_slice.shape != existing.shape:
            print(f"  FAIL shape: rebuilt={rebuilt_slice.shape} existing={existing.shape}")
            passed = False
            continue

        mismatched = [c for c in existing.columns if c not in ("Date", "DayOfWeek")
                      and not np.allclose(existing[c].values, rebuilt_slice[c].values,
                                          equal_nan=True, atol=1e-6)]
        if mismatched:
            print(f"  FAIL columns: {mismatched}")
            passed = False
        else:
            print(f"  OK")

    print(f"\nStage 1: {'PASSED' if passed else 'FAILED'}\n")
    return passed


def stage_2_predictions():
    print("=" * 50)
    print("STAGE 2: predicter.py")
    print("=" * 50)

    df_2425 = pd.read_csv(f"{DATA}/2024-2025.csv", parse_dates=["Date"])
    df_2026 = pd.read_csv(f"{DATA}/2026.csv", parse_dates=["Date"])
    df_raw = pd.concat([df_2425, df_2026], ignore_index=True).sort_values("Date").reset_index(drop=True)
    df_calendar_raw = pd.read_excel(f"{DATA}/academic_calendar_2426.xlsx")
    df_master = build_master(df_raw, df_calendar_raw)
    holdout_dates = df_2026["Date"]

    passed = True
    for meal, expected in EXPECTED_MAE.items():
        print(f"--- {meal} ---")
        meal_df = build_meal_features(df_master, meal)

        try:
            models = load_models(meal)
        except FileNotFoundError as e:
            print(f"  FAIL model load: {e}")
            passed = False
            continue

        preds = predict_rows(meal_df, holdout_dates, models)
        merged = preds.merge(meal_df[["Date", "Target_Headcount"]], on="Date")

        if len(merged) != 151:
            print(f"  FAIL rows: {len(merged)} (expected 151)")
            passed = False

        for col, exp_val in expected.items():
            mae = (merged[col] - merged["Target_Headcount"]).abs().mean()
            diff = abs(mae - exp_val)
            ok = diff <= TOLERANCE
            print(f"  {col:9s} MAE={mae:.2f} expected={exp_val:.2f} diff={diff:.3f} {'OK' if ok else 'FAIL'}")
            if not ok:
                passed = False

    print(f"\nStage 2: {'PASSED' if passed else 'FAILED'}\n")
    return passed


if __name__ == "__main__":
    s1 = stage_1_features()

    if not s1:
        print("=" * 50)
        print("Stage 1 failed -- skipping Stage 2 (predictions built on features can't be trusted).")
        print("OVERALL: FAILED")
    else:
        s2 = stage_2_predictions()
        print("=" * 50)
        print("OVERALL:", "ALL PASSED" if (s1 and s2) else "FAILED")