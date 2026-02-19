"""
generate_data.py
----------------
Generates a synthetic credit card customer dataset (mimicking CC_GENERAL.csv).
Run this to create v1 and v2 of the dataset to practice DVC versioning.

Usage:
    python generate_data.py --version 1   # Creates initial dataset
    python generate_data.py --version 2   # Creates updated dataset (simulates new data)
"""

import argparse
import numpy as np
import pandas as pd
import os

def generate_credit_card_data(n_samples=8950, seed=42):
    np.random.seed(seed)

    data = {
        "CUST_ID": [f"C1{str(i).zfill(4)}" for i in range(1, n_samples + 1)],
        "BALANCE": np.random.exponential(scale=1500, size=n_samples).round(6),
        "BALANCE_FREQUENCY": np.random.uniform(0, 1, n_samples).round(6),
        "PURCHASES": np.random.exponential(scale=1000, size=n_samples).round(6),
        "ONEOFF_PURCHASES": np.random.exponential(scale=500, size=n_samples).round(6),
        "INSTALLMENTS_PURCHASES": np.random.exponential(scale=400, size=n_samples).round(6),
        "CASH_ADVANCE": np.random.exponential(scale=900, size=n_samples).round(6),
        "PURCHASES_FREQUENCY": np.random.uniform(0, 1, n_samples).round(6),
        "ONEOFF_PURCHASES_FREQUENCY": np.random.uniform(0, 1, n_samples).round(6),
        "PURCHASES_INSTALLMENTS_FREQUENCY": np.random.uniform(0, 1, n_samples).round(6),
        "CASH_ADVANCE_FREQUENCY": np.random.uniform(0, 0.5, n_samples).round(6),
        "CASH_ADVANCE_TRX": np.random.randint(0, 20, n_samples),
        "PURCHASES_TRX": np.random.randint(0, 100, n_samples),
        "CREDIT_LIMIT": np.random.choice(
            [500, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000],
            size=n_samples
        ).astype(float),
        "PAYMENTS": np.random.exponential(scale=1500, size=n_samples).round(6),
        "MINIMUM_PAYMENTS": np.random.exponential(scale=400, size=n_samples).round(6),
        "PRC_FULL_PAYMENT": np.random.uniform(0, 1, n_samples).round(6),
        "TENURE": np.random.randint(6, 13, n_samples),
    }

    df = pd.DataFrame(data)

    # Introduce some NaN values (realistic)
    nan_idx_credit = np.random.choice(df.index, size=1, replace=False)
    nan_idx_min = np.random.choice(df.index, size=313, replace=False)
    df.loc[nan_idx_credit, "CREDIT_LIMIT"] = np.nan
    df.loc[nan_idx_min, "MINIMUM_PAYMENTS"] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic credit card dataset")
    parser.add_argument("--version", type=int, choices=[1, 2], default=1,
                        help="Dataset version: 1=initial (8950 rows), 2=updated (9500 rows)")
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/CC_GENERAL.csv"

    if args.version == 1:
        df = generate_credit_card_data(n_samples=8950, seed=42)
        print(f"Generated dataset v1: {len(df)} rows")
    else:
        # Version 2: more rows + slightly different distribution (simulates new data collection)
        df = generate_credit_card_data(n_samples=9500, seed=99)
        print(f"Generated dataset v2: {len(df)} rows (updated)")

    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nSample:\n{df.head(3).to_string()}")


if __name__ == "__main__":
    main()