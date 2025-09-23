import pandas as pd

df = pd.read_parquet("./data/bronze_table.parquet")

# 내용 확인
print(df.head())
