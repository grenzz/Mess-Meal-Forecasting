import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])


def get_raw_history_df():
    raw_df = pd.read_sql('SELECT * FROM "08_2026" ORDER BY "Date"', engine, parse_dates=["Date"])
    return raw_df

def get_calendar_df():
    calendar = pd.read_sql('SELECT "Date", "Category", "Notes" FROM academic_calendar_2426', engine, parse_dates=["Date"])
    return calendar