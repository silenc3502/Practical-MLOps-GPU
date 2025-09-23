import os
import pandas as pd


def save_gold_table(silver_path="./data/silver_table.parquet",
                    output_path="./data/gold_table.parquet"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_parquet(silver_path)
    gold_df = df.assign(review_clean=df["review"].str.lower())
    if os.path.exists(output_path):
        os.remove(output_path)
    gold_df.to_parquet(output_path, index=False)
    print(f"Gold table saved at {output_path}, rows: {len(gold_df)}")
    return gold_df

save_gold_table()