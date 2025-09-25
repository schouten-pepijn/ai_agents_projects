import pandas as pd
from paths import DATA_PATH


def convert_csv_to_parquet(file_path: str):
    df = pd.read_csv(file_path)
    parquet_file_path = file_path.with_suffix(".parquet")
    df.to_parquet(parquet_file_path, index=False)


if __name__ == "__main__":

    files = [
        DATA_PATH / "transactions.csv",
        DATA_PATH / "invoices.csv",
        DATA_PATH / "users.csv",
        DATA_PATH / "counterparties.csv",
    ]

    for file in files:
        convert_csv_to_parquet(file)
