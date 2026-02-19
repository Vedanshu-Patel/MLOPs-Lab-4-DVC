"""
preprocess.py
-------------
Loads raw CC_GENERAL.csv, cleans and preprocesses it,
saves the result to data/processed/CC_PROCESSED.csv.

Run: python preprocess.py
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/CC_GENERAL.csv"
PROCESSED_PATH = "data/processed/CC_PROCESSED.csv"


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Raw shape: {df.shape}")

    # ── 1. Drop identifier column
    df = df.drop(columns=["CUST_ID"], errors="ignore")

    # ── 2. Fill missing values
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())

    # ── 3. Feature engineering
    df["MONTHLY_AVG_PURCHASE"] = df["PURCHASES"] / df["TENURE"]
    df["MONTHLY_AVG_CASH_ADVANCE"] = df["CASH_ADVANCE"] / df["TENURE"]
    df["PURCHASE_TO_LIMIT_RATIO"] = df["PURCHASES"] / (df["CREDIT_LIMIT"] + 1)
    df["BALANCE_TO_LIMIT_RATIO"] = df["BALANCE"] / (df["CREDIT_LIMIT"] + 1)

    # ── 4. Clip outliers at 99th percentile for key numeric columns
    cols_to_clip = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS", "MINIMUM_PAYMENTS"]
    for col in cols_to_clip:
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed shape: {df.shape}")
    print(f"Saved to: {output_path}")
    return df


if __name__ == "__main__":
    df = preprocess(RAW_PATH, PROCESSED_PATH)
    print("\nColumn list:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}  (nulls: {df[col].isna().sum()})")