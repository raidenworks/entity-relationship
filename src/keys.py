from typing import List, Tuple, Any
import itertools

import numpy as np
import pandas as pd


def likely_pk_names(tname: str, cols: List[str], id_aliases: set[str]) -> List[str]:
    base = tname[:-1] if tname.endswith("s") else tname
    patterns = {"id", f"{base}_id", f"{tname}_id"}
    return [c for c in cols if c in patterns or c in id_aliases or c.endswith("_id")]


def to_float_scalar(x: Any) -> float:
    if isinstance(x, np.generic):
        return float(np.asarray(x).item())
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        y = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
        return float(y) if pd.notna(y) else float("nan")


def pick_pk(
    df: pd.DataFrame,
    tname: str,
    profiles: dict | None,
    cfg: dict,
    logger,
) -> Tuple[List[str], List[str]]:
    H = cfg["heuristics"]
    id_aliases = set([x.lower() for x in cfg.get("id_aliases", [])])
    logs: List[str] = []
    cols = list(df.columns)

    if profiles is not None and tname in profiles:
        base_prof = profiles[tname].set_index("column")
        prof_t = base_prof.assign(
            null_rate=pd.to_numeric(base_prof.get("null_rate"), errors="coerce"),
            unique_ratio=pd.to_numeric(base_prof.get("unique_ratio"), errors="coerce"),
        )
    else:
        n = max(1, len(df))
        prof_t = pd.DataFrame({
            "null_rate": df.isna().mean(),
            "unique_ratio": df.nunique(dropna=True) / n,
        })

    def metric(col: str, name: str) -> float:
        return to_float_scalar(prof_t.at[col, name])

    def is_unique_not_null_local(df_: pd.DataFrame, pk_cols: List[str]) -> bool:
        if not pk_cols:
            return False
        sub = df_[pk_cols]
        if sub.isna().any().any():
            return False
        return sub.drop_duplicates().shape[0] == len(df_)

    hinted = list(dict.fromkeys([c for c in likely_pk_names(tname, cols, id_aliases) if c in prof_t.index]))
    for c in hinted:
        nr, ur = metric(c, "null_rate"), metric(c, "unique_ratio")
        logs.append(f"[{tname}] try hinted '{c}': null={nr:.3f}, uniq={ur:.3f}")
        if (nr <= H["PK_NULL_RATE_MAX"]) and (ur >= H["PK_UNIQUENESS_MIN"]):
            logs.append(f"[{tname}] PK=['{c}'] (hinted single)")
            return [c], logs

    for c in cols:
        if c not in prof_t.index:
            continue
        nr, ur = metric(c, "null_rate"), metric(c, "unique_ratio")
        if (nr <= H["PK_NULL_RATE_MAX"]) and (ur >= H["PK_UNIQUENESS_MIN"]):
            logs.append(f"[{tname}] PK=['{c}'] (unique single)")
            return [c], logs
        else:
            logs.append(f"[{tname}] reject '{c}' as PK: null={nr:.3f}, uniq={ur:.3f}")

    mask = (prof_t["null_rate"] <= H["PK_NULL_RATE_MAX"]) & (prof_t["unique_ratio"] >= H["MIN_CARDINALITY_FOR_PK"])
    composite_candidates: List[str] = [c for c in cols if (c in prof_t.index) and bool(mask.get(c, False))]
    max_k = min(H["MAX_COMPOSITE_PK_SIZE"], len(composite_candidates))
    for k in range(2, max_k + 1):
        for combo in itertools.combinations(composite_candidates, k):
            if is_unique_not_null_local(df, list(combo)):
                logs.append(f"[{tname}] PK={list(combo)} (composite)")
                return list(combo), logs
            else:
                logs.append(f"[{tname}] combo not unique: {combo}")

    if hinted:
        best = max(hinted, key=lambda c: (metric(c, "unique_ratio"), -metric(c, "null_rate")))
        logs.append(f"[{tname}] fallback PK=['{best}']")
        return [best], logs

    logs.append(f"[{tname}] no PK found")
    return [], logs


def parent_domain(df: pd.DataFrame, pk_cols: list[str]) -> set:
    if len(pk_cols) == 1:
        return set(df[pk_cols[0]].dropna().astype(str))
    return set(df[pk_cols].dropna().astype(str).apply(tuple, axis=1))


def singular(name: str) -> str:
    return name[:-1] if name.endswith("s") else name


def fk_cov(series: pd.Series, pset: set) -> float:
    s = series.dropna().astype(str)
    if len(s) == 0:
        return np.nan
    return s.isin(pset).mean()


def infer_foreign_keys(tables: dict, inferred_pk: dict, cfg: dict, logger) -> dict:
    H = cfg["heuristics"]
    id_aliases = set([x.lower() for x in cfg.get("id_aliases", [])])
    parent_sets = {t: parent_domain(df, inferred_pk[t]) for t, df in tables.items()}
    inferred_fk = {t: {} for t in tables}
    for parent_t, pdf in tables.items():
        base = singular(parent_t)
        pks = inferred_pk[parent_t]
        pset = parent_sets[parent_t]
        for child_t, cdf in tables.items():
            if child_t == parent_t:
                continue
            candidates = []
            for col in cdf.columns:
                if col == base or col == f"{base}_id" or col in pks:
                    candidates.append(col)
                if col.endswith("_id") and base in col:
                    candidates.append(col)
                if col in id_aliases and base in col:
                    candidates.append(col)
            candidates = list(dict.fromkeys(candidates))
            for c in candidates:
                cov = fk_cov(cdf[c], pset)
                if pd.notna(cov) and cov >= H["FK_COVERAGE_MIN"]:
                    key = f"{c}->{parent_t}"
                    inferred_fk[child_t][key] = (c, parent_t, pks, cov)
    total = sum(len(v) for v in inferred_fk.values())
    logger.info(f"Inferred {total} FK edges with coverage >= {H['FK_COVERAGE_MIN']:.0%}")
    return inferred_fk
