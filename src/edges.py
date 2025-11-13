from pathlib import Path
from typing import List, Union, Tuple, Any, Dict

import numpy as np
import pandas as pd


ColSpec = Union[str, List[str], tuple]


def _normalize_cols(df: pd.DataFrame, cols: ColSpec) -> List[str]:
    if cols is None:
        return []
    if isinstance(cols, (list, tuple)):
        return [c for c in cols if isinstance(c, str) and c in df.columns]
    return [cols] if (isinstance(cols, str) and cols in df.columns) else []


def _as_tuple_str(df: pd.DataFrame, cols: ColSpec) -> pd.Series:
    col_list = _normalize_cols(df, cols)
    if not col_list:
        return pd.Series([], dtype=str)
    if len(col_list) == 1:
        return df[col_list[0]].dropna().astype(str)
    return (
        df[col_list]
        .dropna()
        .astype(str)
        .apply(tuple, axis=1)
        .astype(str)
    )


def _source_guess(child_col: str, parent_t: str, parent_pk_cols) -> str:
    base = parent_t[:-1] if parent_t.endswith("s") else parent_t
    pk_names = set(parent_pk_cols if isinstance(parent_pk_cols, (list, tuple)) else [parent_pk_cols])
    if child_col in pk_names or child_col == base:
        return "name_match"
    if child_col == f"{base}_id" or (child_col.endswith("_id") and base in child_col):
        return "pattern"
    return "domain_coverage"


def _cardinality(child_vals: pd.Series, parent_vals: pd.Series) -> str:
    common = set(child_vals) & set(parent_vals)
    if not common:
        return "N:1"
    child_on = child_vals[child_vals.isin(common)]
    if child_on.empty:
        return "N:1"
    counts = child_on.value_counts()
    if counts.max() == 1:
        return "1:1" if len(child_on) == child_on.nunique() else "N:1"
    return "N:1"


def generate_edge_list(tables: dict, inferred_fk: dict, inferred_pk: dict, cfg: dict, logger) -> pd.DataFrame:
    edges = []
    for child_t, fks in inferred_fk.items():
        for _, (child_cols, parent_t, parent_pk_cols, cov) in fks.items():
            child_cols_list = list(child_cols) if isinstance(child_cols, (list, tuple)) else [child_cols]
            parent_pk_list = list(parent_pk_cols) if isinstance(parent_pk_cols, (list, tuple)) else [parent_pk_cols]

            child_df = tables[child_t]
            parent_df = tables[parent_t]

            child_series = _as_tuple_str(child_df, child_cols_list)
            parent_series = _as_tuple_str(parent_df, parent_pk_list)

            child_set = set(child_series)
            parent_set = set(parent_series)
            child_unique = len(child_set)
            matches_unique = len(child_set & parent_set)
            coverage = float(matches_unique / child_unique) if child_unique else float("nan")

            card = _cardinality(child_series, parent_series)

            edges.append({
                "child_table": child_t,
                "child_columns": child_cols_list,
                "parent_table": parent_t,
                "parent_pk": parent_pk_list,
                "coverage": round(coverage, 3) if not np.isnan(coverage) else None,
                "child_unique": child_unique,
                "matches_unique": matches_unique,
                "cardinality": card,
                "composite": (len(child_cols_list) > 1) or (len(parent_pk_list) > 1),
            })

    edge_df = pd.DataFrame(edges)

    def _to_sort_key(v):
        if isinstance(v, list):
            return tuple(v)
        if isinstance(v, tuple):
            return v
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return tuple()
        return (str(v),)

    if "child_columns" in edge_df.columns:
        edge_df["child_columns_key"] = edge_df["child_columns"].map(_to_sort_key)
    if "parent_pk" in edge_df.columns:
        edge_df["parent_pk_key"] = edge_df["parent_pk"].map(_to_sort_key)

    sort_cols = ["child_table", "parent_table"]
    if "child_columns_key" in edge_df.columns:
        sort_cols.append("child_columns_key")

    edge_df = edge_df.sort_values(sort_cols).drop(columns=[c for c in ["child_columns_key", "parent_pk_key"] if c in edge_df.columns])

    out_path = Path(cfg["output_dir"]) / cfg["edges_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(out_path, index=False)
    logger.info(f"Edges saved -> {out_path} ({len(edge_df)} links)")
    return edge_df

