import pandas as pd
import numpy as np

MEALS = ['Breakfast', 'Lunch', 'Evening_Snacks', 'Dinner']
MESS_CLOSED_RANGES = [
    ("2024-06-01", "2024-06-30"),
    ("2024-12-08", "2024-12-23"),
]
FEST_KEYWORDS = 'odyssey|astra|esya|e summit|riise'
BASE_FEATURES = [
    'Date', 'DayOfWeek', 'DayOfMonth', 'Month', 'Is_Start_Of_Month',
    'Is_End_Of_Month', 'Is_Weekend', 'Is_Holiday', 'Is_Vacation',
    'Is_Exam_Period', 'Is_Fest', 'Days_Until_Holiday',
    'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos',
    'Is_Broke_Weekend', 'Is_Rich_Weekend', 'WeekOfYear', 'DayOfYear'
]

def build_master(df_raw, df_calendar_raw):
    df_calendar = df_calendar_raw.copy()
    df_calendar.columns = ["Date", "Category", "Notes"]
    df_calendar["Date"] = pd.to_datetime(df_calendar["Date"])
    df_calendar["Category"] = df_calendar["Category"].replace({
        "winter vacation": "winter_vacation",
        "holdiay": "holiday"
    })

    df_master = pd.merge(df_raw, df_calendar, on="Date", how="left")
    df_master = df_master.set_index("Date").asfreq("D").reset_index()
    df_master["DayOfWeek"] = df_master["Date"].dt.day_name()

    for start, end in MESS_CLOSED_RANGES:
        df_master.loc[df_master["Date"].between(start, end), "Category"] = "mess_closed"

    df_master['DayOfMonth'] = df_master['Date'].dt.day
    df_master['DayOfWeek_Num'] = df_master['Date'].dt.dayofweek
    df_master['DayOfYear'] = df_master['Date'].dt.dayofyear
    df_master['WeekOfYear'] = df_master['Date'].dt.isocalendar().week
    df_master['Month'] = df_master['Date'].dt.month
    df_master['Is_Start_Of_Month'] = (df_master['DayOfMonth'] <= 7).astype(int)
    df_master['Is_End_Of_Month'] = (df_master['DayOfMonth'] >= 24).astype(int)
    df_master['Is_Weekend'] = (df_master['DayOfWeek_Num'] >= 5).astype(int)

    df_master['Is_Holiday'] = (df_master['Category'] == 'holiday').astype(int)
    df_master['Is_Vacation'] = df_master['Category'].fillna('').str.contains('vacation', case=False).astype(int)
    df_master['Is_Exam_Period'] = df_master['Category'].isin(['mid_sem_exams', 'end_sem_exams']).astype(int)
    df_master['Is_Fest'] = df_master['Notes'].fillna('').str.contains(FEST_KEYWORDS, case=False).astype(int)

    df_master['Next_Holiday_Date'] = np.where(df_master['Is_Holiday'] == 1, df_master['Date'], pd.NaT)
    df_master['Next_Holiday_Date'] = pd.to_datetime(df_master['Next_Holiday_Date']).bfill()
    df_master['Days_Until_Holiday'] = (df_master['Next_Holiday_Date'] - df_master['Date']).dt.days.fillna(99)
    df_master.drop(columns=['Next_Holiday_Date'], inplace=True)

    df_master['DayOfWeek_Sin'] = np.sin(2 * np.pi * df_master['DayOfWeek_Num'] / 7)
    df_master['DayOfWeek_Cos'] = np.cos(2 * np.pi * df_master['DayOfWeek_Num'] / 7)
    df_master['Month_Sin'] = np.sin(2 * np.pi * df_master['Month'] / 12)
    df_master['Month_Cos'] = np.cos(2 * np.pi * df_master['Month'] / 12)

    df_master['Is_Broke_Weekend'] = df_master['Is_End_Of_Month'] * df_master['Is_Weekend']
    df_master['Is_Rich_Weekend'] = df_master['Is_Start_Of_Month'] * df_master['Is_Weekend']

    return df_master


def build_meal_features(df_master, meal):
    meal_df = df_master[BASE_FEATURES + [meal]].copy()
    meal_df.rename(columns={meal: 'Target_Headcount'}, inplace=True)

    for lag in [1, 2, 3, 7, 14, 21]:
        meal_df[f'Target_{lag}_Days_Ago'] = meal_df['Target_Headcount'].shift(lag)

    for w in [3, 7, 14, 28]:
        meal_df[f'Target_{w}_Day_Avg'] = meal_df['Target_Headcount'].shift(1).rolling(window=w).mean()
        meal_df[f'Target_{w}_Day_Std'] = meal_df['Target_Headcount'].shift(1).rolling(window=w).std()
        meal_df[f'Target_{w}_Day_Max'] = meal_df['Target_Headcount'].shift(1).rolling(window=w).max()
        meal_df[f'Target_{w}_Day_Min'] = meal_df['Target_Headcount'].shift(1).rolling(window=w).min()

    meal_df['Target_EWMA_7'] = meal_df['Target_Headcount'].shift(1).ewm(span=7, adjust=False).mean()
    meal_df['Target_EWMA_14'] = meal_df['Target_Headcount'].shift(1).ewm(span=14, adjust=False).mean()
    meal_df['Trend_7_vs_28'] = meal_df['Target_7_Day_Avg'] / (meal_df['Target_28_Day_Avg'] + 1)

    return meal_df


def get_model_feature_cols(meal_df):
    exclude = {"Date", "Target_Headcount", "Category", "Notes", "DayOfWeek"}
    return [c for c in meal_df.columns if c not in exclude]


def get_prediction_row(meal_df, target_date, feature_cols):
    row = meal_df[meal_df["Date"] == pd.Timestamp(target_date)]
    if row.empty:
        raise ValueError(f"{target_date} not found -- did you add it to df_master first?")
    X = row[feature_cols]
    if X.isna().any(axis=None):
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN values for {target_date} in: {bad_cols} -- not enough history yet.")
    return X