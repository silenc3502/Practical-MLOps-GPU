import os

from create_dummy_review_data import get_dummy_data

def save_bronze_table(output_path: str = "./data/bronze_table.parquet", n_rows: int = 1000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 데이터 생성
    df = get_dummy_data(n_rows)

    # 기존 파일 삭제
    if os.path.exists(output_path):
        os.remove(output_path)

    # parquet 저장 (항상 하나만)
    df.to_parquet(output_path, index=False)
    print(f"Bronze table saved at {output_path}, rows: {len(df)}")

save_bronze_table()