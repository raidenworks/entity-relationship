from __future__ import annotations

from typing import Dict
import pandas as pd

from .types import infer_sql_type


def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        null_rate = s.isna().mean()
        nunique = s.nunique(dropna=True)
        uniq_ratio = nunique / max(1, n)
        ex_val = next((str(v) for v in s.dropna().head(1).tolist()), "")
        sqlt = infer_sql_type(s)
        maxlen = int(s.dropna().astype(str).str.len().max()) if s.dtype == object or pd.api.types.is_string_dtype(s) else None
        rows.append(dict(
            column=c,
            sql_type=str(sqlt),
            null_rate=float(round(null_rate, 3)),
            unique_ratio=float(round(uniq_ratio, 3)),
            nunique=int(nunique),
            example=ex_val,
            maxlen=(int(maxlen) if maxlen is not None else None),
        ))
    return pd.DataFrame(rows)


def build_profiles(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {t: profile_table(df) for t, df in tables.items()}

