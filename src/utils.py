
import pandas as pd
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return df

def basic_checks(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "nulls": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()}
    }
