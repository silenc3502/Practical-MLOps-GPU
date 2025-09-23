import os
import pandas as pd


def save_silver_table(bronze_path: str = "./data/bronze_table.parquet",
                      output_path: str = "./data/silver_table.parquet"):
    df = pd.read_parquet(bronze_path)
    silver_df = df.dropna(subset=["review"])
    if os.path.exists(output_path):
        os.remove(output_path)
    silver_df.to_parquet(output_path, index=False)
    print(f"Silver table saved at {output_path}, rows: {len(silver_df)}")
    return silver_df

save_silver_table()