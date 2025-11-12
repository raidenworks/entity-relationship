import os
import re
import json
import math
import itertools
from pathlib import Path
from typing import List, Tuple, Any, Union
import numpy as np
import pandas as pd
import yaml
import html as _html
import fnmatch
from slugify import slugify
from graphviz import Digraph
import logging
import time
# -------------------------------------------------------------
# Harmonized ERD renderer using unique-based coverage and qualified labels
# -------------------------------------------------------------
def render_erd_v2(tables: dict, profiles: dict, inferred_pk: dict, inferred_fk: dict, cfg: dict, logger: logging.Logger) -> Path:
    spline_type = cfg.get("spline_type", "spline")
    font = cfg.get("font", "Helvetica")
    dot = Digraph("ERD", graph_attr={
        "rankdir": cfg.get("rankdir", "LR"),
        "splines": spline_type,
        "fontname": font,
        "fontsize": "10",
        "bgcolor": "white",
    })
    dot.attr("node", shape="plaintext", fontname=font, fontsize="10")
    dot.attr("edge", arrowsize="0.7", color="black")
    for t, df in tables.items():
        dot.node(t, label=f'''<
{node_html(t, df, inferred_pk, inferred_fk, profiles)}
>''')
    def _fmt_side(table: str, cols) -> str:
        if isinstance(cols, (list, tuple)):
            cols = list(cols)
            if len(cols) == 0:
                return table
            if len(cols) == 1:
                return f"{table}.{cols[0]}"
            return f"{table}.({','.join(map(str, cols))})"
        return f"{table}.{cols}" if cols else table
    for child_t, fks in inferred_fk.items():
        for _, (child_cols, parent_t, parent_pk, _cov_unused) in fks.items():
            child_df = tables[child_t]
            parent_df = tables[parent_t]
            child_cols_list = list(child_cols) if isinstance(child_cols, (list, tuple)) else [child_cols]
            parent_pk_list  = list(parent_pk) if isinstance(parent_pk, (list, tuple)) else [parent_pk]
            child_series = _as_tuple_str(child_df, child_cols_list)
            parent_series = _as_tuple_str(parent_df, parent_pk_list)
            child_set = set(child_series)
            parent_set = set(parent_series)
            child_unique = len(child_set)
            matches_unique = len(child_set & parent_set)
            cov = float(matches_unique / child_unique) if child_unique else float("nan")
            base = f"{_fmt_side(child_t, child_cols_list)} -> {_fmt_side(parent_t, parent_pk_list)}"
            label = base if (cov != cov) else f"{base} ({cov:.2f})"
            dot.edge(child_t, parent_t, label=label)
    out_stem = Path(cfg["output_dir"]) / Path(cfg["erd_output"]).with_suffix("").name
    out_stem = Path(str(out_stem))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    dot.format = "svg"
    svg_path = Path(dot.render(str(out_stem), cleanup=True))
    logger.info(f"ERD rendered -> {svg_path}")
    return svg_path
# =============================================================
# 0) Config + Logging
# =============================================================
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def setup_logging(cfg: dict) -> logging.Logger:
    log_path = Path(cfg.get("output_dir", ".")) / cfg.get("log_file", "pipeline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("erd_pipeline")
    logger.info(f"Logging initialized -> {log_path}")
    return logger
# =============================================================
DATE_LIKE_THRESHOLD: float = 0.5  # defaults overridden by config.yml at runtime
COLUMN_SAMPLE_SIZE: int = 50  # defaults overridden by config.yml at runtime
INFERENCE_THRESHOLD: float = 0.98  # defaults overridden by config.yml at runtime
SAMPLE_RANDOM_SEED: int | None = None  # defaults overridden by config.yml at runtime
def set_heuristics_from_config(cfg: dict) -> None:
    global DATE_LIKE_THRESHOLD, COLUMN_SAMPLE_SIZE, INFERENCE_THRESHOLD, SAMPLE_RANDOM_SEED
    try:
        DATE_LIKE_THRESHOLD = float(cfg.get("heuristics", {}).get("DATE_LIKE_THRESHOLD", DATE_LIKE_THRESHOLD))
    except Exception:
        pass
    try:
        COLUMN_SAMPLE_SIZE = int(cfg.get("heuristics", {}).get("COLUMN_SAMPLE_SIZE", COLUMN_SAMPLE_SIZE))
    except Exception:
        pass
    try:
        INFERENCE_THRESHOLD = float(cfg.get("heuristics", {}).get("INFERENCE_THRESHOLD", INFERENCE_THRESHOLD))
    except Exception:
        pass
    try:
        seed_val = cfg.get("heuristics", {}).get("SAMPLE_RANDOM_SEED", None)
        SAMPLE_RANDOM_SEED = None if seed_val in (None, "", "null") else int(seed_val)
    except Exception:
        SAMPLE_RANDOM_SEED = None
def norm_name(s: str) -> str:
    s = s.strip().replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    return slugify(s, separator="_").lower()
def _sample_nonnull(s: pd.Series, max_n: int) -> pd.Series:
    nonnull = s.dropna()
    if len(nonnull) <= max_n:
        return nonnull
    # Use uniform random sampling for better coverage of anomalies
    return nonnull.sample(n=max_n, random_state=SAMPLE_RANDOM_SEED)
def is_likely_datetime(s: pd.Series, threshold: float | None = None) -> bool:
    thr = DATE_LIKE_THRESHOLD if threshold is None else threshold
    nonnull = s.dropna()
    if len(nonnull) == 0:
        return False
    # Use random sampling governed by COLUMN_SAMPLE_SIZE and SAMPLE_RANDOM_SEED
    sample = _sample_nonnull(s, COLUMN_SAMPLE_SIZE).astype(str)
    has_date_sep = sample.str.contains(r"[-/]", na=False).mean()
    has_time_sep = sample.str.contains(r":", na=False).mean()
    has_iso_t   = sample.str.contains(r"[Tt]", na=False).mean()
    has_month   = sample.str.contains(r"(?i)jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", na=False).mean()
    is_digits   = sample.str.fullmatch(r"\d+", na=False).mean()
    len8        = sample.str.len().eq(8).mean()
    len10       = sample.str.len().eq(10).mean()
    len13       = sample.str.len().eq(13).mean()
    len14       = sample.str.len().eq(14).mean()
    strong = max(has_date_sep, has_time_sep, has_iso_t, has_month, len8, len10, len13, len14, is_digits)
    combo = min(has_date_sep, has_time_sep)
    return (strong >= thr) or (combo >= thr)
def _try_parse_with_format(sample: pd.Series, fmt: str, threshold: float = 0.98) -> bool:
    try:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce", cache=True)
        return parsed.notna().mean() >= threshold
    except Exception:
        return False
def _pick_known_datetime_format(sample: pd.Series) -> Union[str, None]:
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y%m%d",
        "%Y%m%d%H%M%S",
    ]
    for fmt in fmts:
        if _try_parse_with_format(sample, fmt):
            return fmt
    # Try ISO8601 fast path if supported
    try:
        parsed = pd.to_datetime(sample, format="ISO8601", errors="coerce", cache=True)
        if parsed.notna().mean() >= 0.98:
            return "ISO8601"
    except Exception:
        pass
    return None
def _pick_epoch_unit(sample: pd.Series) -> Union[str, None]:
    units = ["s", "ms", "us", "ns"]
    for u in units:
        try:
            parsed = pd.to_datetime(sample, unit=u, errors="coerce")
            ok = parsed.notna()
            if ok.mean() < 0.98:
                continue
            years = parsed[ok].dt.year
            if (years.between(1970, 2100).mean() >= 0.98):
                return u
        except Exception:
            continue
    return None
def to_datetime_fast(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    nonnull = s.dropna()
    if len(nonnull) == 0:
        return pd.to_datetime(s, errors="coerce", cache=True)
    sample = nonnull.head(2000)
    sample_str = sample.astype(str)
    fmt = _pick_known_datetime_format(sample_str)
    if fmt:
        try:
            return pd.to_datetime(s, format=fmt, errors="coerce", cache=True)
        except Exception:
            pass
    is_digits = sample_str.str.fullmatch(r"\d+").mean() >= 0.98
    if is_digits:
        if sample_str.str.len().eq(8).mean() >= 0.98:
            try:
                return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce", cache=True)
            except Exception:
                pass
        unit = _pick_epoch_unit(pd.to_numeric(sample_str, errors="coerce"))
        if unit:
            try:
                return pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit=unit, errors="coerce")
            except Exception:
                pass
    try:
        test = pd.to_datetime(sample_str, format="mixed", errors="coerce")
        if test.notna().mean() >= 0.98:
            return pd.to_datetime(s.astype(str), format="mixed", errors="coerce")
    except Exception:
        pass
    # Fallback to default; detect slow path and switch to chunked with progress
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UserWarning)
        _ = pd.to_datetime(sample_str, errors="coerce", cache=True)
        if any("Could not infer format" in str(m.message) for m in w):
            col = getattr(s, "name", None)
            desc = f"datetime[{col}]" if col else "datetime"
            return to_datetime_chunked(s.astype(str), desc=f"Parsing {desc}")
    return pd.to_datetime(s, errors="coerce", cache=True)
def to_datetime_chunked(
    s: pd.Series,
    fmt: Union[str, None] = None,
    unit: Union[str, None] = None,
    chunksize: int = 200_000,
    desc: str = "Parsing datetimes",
) -> pd.Series:
    """Parse datetimes in chunks with a progress bar.
    - Progress total equals the count of non-null inputs (values to infer)
    - Uses tqdm when available (leave=False to avoid repeated 100% lines)
    - Falls back to coarse console updates offline
    """
    # Harden numeric args to integers for safe slicing
    try:
        n = int(len(s))
    except Exception:
        n = len(s)
    try:
        chunksize = int(chunksize)
    except Exception:
        chunksize = 200_000
    if n == 0:
        return pd.to_datetime(s, format=fmt, unit=unit, errors="coerce")
    import numpy as np
    parts: list[np.ndarray] = []
    total = int(s.notna().sum())
    processed = 0
    last_bucket = -1
    bar = None
    try:
        from tqdm.auto import tqdm  # type: ignore
        bar = tqdm(total=total, desc=desc, unit="rows", leave=False)
    except Exception:
        bar = None
    log = logging.getLogger("erd_pipeline")
    for i in range(0, n, chunksize):
        j = min(i + chunksize, n)
        ii = int(i)
        jj = int(j)
        chunk = s.iloc[ii:jj]
        try:
            parsed_chunk = pd.to_datetime(chunk, format=fmt, unit=unit, errors="coerce")
            parts.append(parsed_chunk.values)
        except TypeError as te:
            # Log rich context to help diagnose float index or bad params
            log.error(
                "Datetime chunk parse TypeError: %s | i=%s j=%s n=%s chunksize=%s fmt=%s unit=%s s.dtype=%s chunk.dtype=%s",
                te,
                ii,
                jj,
                n,
                chunksize,
                fmt,
                unit,
                getattr(s, "dtype", None),
                getattr(chunk, "dtype", None),
            )
            log.error("Chunk head non-null sample: %s", chunk.dropna().astype(str).head(3).tolist())
            # Fallback: force string parse for this chunk
            try:
                parsed_chunk = pd.to_datetime(chunk.astype(str), errors="coerce")
                parts.append(parsed_chunk.values)
            except Exception as te2:
                log.error("Fallback string parse failed: %s", te2)
                # Last resort: fill with NaT for this segment
                parts.append(np.full(int(jj - ii), np.datetime64("NaT"), dtype="datetime64[ns]"))
        except Exception as ex:
            log.error(
                "Datetime chunk parse error: %s | i=%s j=%s fmt=%s unit=%s", ex, ii, jj, fmt, unit
            )
            # Fallback to safe coercion
            parsed_chunk = pd.to_datetime(chunk, errors="coerce")
            parts.append(parsed_chunk.values)

        inc = int(chunk.notna().sum())
        if bar is not None:
            bar.update(inc)
        else:
            processed += inc
            if total > 0:
                pct = int((processed / total) * 100)
                bucket = (pct // 10) * 10
                if bucket != last_bucket:
                    print(f"{desc}: {processed}/{total} ({bucket}%)")
                    last_bucket = bucket
    if bar is not None:
        bar.close()
    out_vals = np.concatenate(parts) if parts else np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    return pd.Series(out_vals, index=s.index)
def read_csv_fast(path: Union[str, Path], nrows: Union[int, None] = None, sep: Union[str, None] = None) -> pd.DataFrame:
    """Read CSV preferring the PyArrow engine when available, with safe fallback.

    If sep is provided, it is passed to pandas to control the delimiter.
    """
    try:
        return pd.read_csv(path, nrows=nrows, engine="pyarrow", sep=sep)
    except Exception:
        return pd.read_csv(path, nrows=nrows, sep=sep)

def _detect_sep_quick(path: Union[str, Path], candidates: list[str]) -> str:
    """Lightweight delimiter detection from a handful of lines.

    Counts candidate occurrences per non-empty line and picks the delimiter
    with the most consistent non-zero count. Falls back to ','.
    """
    try:
        counts: dict[str, list[int]] = {c: [] for c in candidates}
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines_checked = 0
            for line in f:
                if not line.strip():
                    continue
                for c in candidates:
                    counts[c].append(line.count(c))
                lines_checked += 1
                if lines_checked >= 16:
                    break
        best_sep = ","
        best_score = -1
        for c, arr in counts.items():
            nz = [x for x in arr if x > 0]
            if not nz:
                continue
            mode = max(nz, key=nz.count)
            score = nz.count(mode)
            if score > best_score:
                best_score = score
                best_sep = c
        return best_sep
    except Exception:
        return ","

def _choose_sep_for_path(p: Path, cfg: dict) -> Union[str, None]:
    """Resolve CSV separator via config and auto-detection.

    Supports root-level config keys:
      - csv_sep: 'auto' or explicit sep (',', '|', '\t', ';')
      - csv_sep_candidates: list of candidates for auto
      - csv_separators: list of {pattern, sep} overrides
    """
    # Pattern overrides take precedence
    overrides = cfg.get("csv_separators", []) or []
    path_str = str(p)
    for rule in overrides:
        try:
            pat = rule.get("pattern")
            sep = rule.get("sep")
            if pat and sep and fnmatch.fnmatch(path_str, pat):
                return sep
        except Exception:
            continue

    sep_cfg = str(cfg.get("csv_sep", "auto")).lower()
    if sep_cfg != "auto" and sep_cfg:
        return cfg.get("csv_sep")

    candidates = cfg.get("csv_sep_candidates") or [",", "|", "\t", ";"]
    return _detect_sep_quick(p, candidates)
def load_tables(cfg: dict, logger: logging.Logger) -> dict:
    folder = Path(cfg["input_csv_dir"])
    csv_glob = cfg.get("csv_glob", "*.csv")
    sample_rows = cfg["heuristics"].get("SAMPLE_ROWS", None)
    if not folder.exists():
        logger.error(f"Input folder not found: {folder}")
        raise SystemExit(1)
    out = {}
    for p in folder.glob(csv_glob):
        try:
            tname = norm_name(p.stem)
            sep = _choose_sep_for_path(p, cfg)
            df = read_csv_fast(p, nrows=sample_rows, sep=sep)
            df = df.rename(columns={c: norm_name(c) for c in df.columns})
            df.attrs["_tname"] = tname
            out[tname] = df
            logger.info(f"Loaded {p.name} -> table '{tname}' (rows={len(df)}, cols={len(df.columns)})")
        except Exception as e:
            logger.warning(f"Failed to read {p}: {e}")
    if not out:
        logger.error(f"No CSV files matched in {folder} (glob='{csv_glob}').")
        raise SystemExit(1)
    logger.info(f"Loaded {len(out)} tables: {', '.join(sorted(out.keys()))}")
    return out
def infer_sql_type(s: pd.Series) -> str:
    thr = INFERENCE_THRESHOLD if 'INFERENCE_THRESHOLD' in globals() else 0.98
    sample_nonnull = _sample_nonnull(s, COLUMN_SAMPLE_SIZE) if 'COLUMN_SAMPLE_SIZE' in globals() else s.dropna().head(5000)
    sample_str = sample_nonnull.astype(str) if len(sample_nonnull) else sample_nonnull
    # booleans
    if len(sample_nonnull) and sample_nonnull.isin([0, 1, True, False]).mean() >= thr:
        return "boolean"
    # ints (native or numeric-looking strings)
    if pd.api.types.is_integer_dtype(s) or (len(sample_nonnull) and pd.api.types.is_integer_dtype(sample_nonnull)):
        vmax = sample_nonnull.astype("int64", errors="ignore").abs().max() if len(sample_nonnull) else 0
        return "integer" if (pd.isna(vmax) or vmax <= 2_147_483_647) else "bigint"
    if pd.api.types.is_object_dtype(s):
        nonnull = sample_str
        if len(nonnull) and (nonnull.str.fullmatch(r"[+-]?\d+").mean() >= thr):
            vmax = pd.to_numeric(nonnull, errors="coerce").abs().max()
            return "integer" if (pd.isna(vmax) or vmax <= 2_147_483_647) else "bigint"
    # floats (native or numeric-looking strings with decimals)
    if pd.api.types.is_float_dtype(s) or (
        pd.api.types.is_object_dtype(s)
        and len(sample_nonnull)
        and sample_str.str.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?").mean() >= thr
    ):
        return "double"
    # datetimes / dates (fast path): attempt only if likely date-like
    if not is_likely_datetime(s):
        parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    else:
        parsed = to_datetime_fast(s)
    frac = parsed.notna().mean()
    if frac > 0.98:
        try:
            no_tz = parsed.dt.tz_convert(None)
        except Exception:
            no_tz = parsed.dt.tz_localize(None)
        is_date_only = (no_tz.dt.normalize() == no_tz).mean() > 0.95
        return "date" if is_date_only else "timestamp"
    # strings -> varchar/text
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        lens = (sample_str.str.len() if len(sample_nonnull) else pd.Series([], dtype=int))
        m = int(lens.max()) if len(lens) else 0
        return f"varchar({m})" if m <= 255 else "text"
    # fallback
    return str(s.dtype)
def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        null_rate = s.isna().mean()
        nunique   = s.nunique(dropna=True)
        uniq_ratio= nunique / max(1, n)
        ex_val    = next((str(v) for v in s.dropna().head(1).tolist()), "")
        sqlt      = infer_sql_type(s)
        maxlen    = int(s.dropna().astype(str).str.len().max()) if s.dtype==object or pd.api.types.is_string_dtype(s) else None
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
def likely_pk_names(tname: str, cols: list[str], id_aliases: set[str]) -> list[str]:
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
def pick_pk(df: pd.DataFrame, tname: str, profiles: dict | None, cfg: dict, logger: logging.Logger) -> Tuple[List[str], List[str]]:
    H = cfg["heuristics"]
    id_aliases = set([x.lower() for x in cfg.get("id_aliases", [])])
    logs: List[str] = []
    cols = list(df.columns)
    if profiles is not None and tname in profiles:
        base_prof = profiles[tname].set_index("column")
        prof_t = base_prof.assign(
            null_rate    = pd.to_numeric(base_prof.get("null_rate"), errors="coerce"),
            unique_ratio = pd.to_numeric(base_prof.get("unique_ratio"), errors="coerce"),
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
    # (a) hinted singles by name
    hinted = list(dict.fromkeys([c for c in likely_pk_names(tname, cols, id_aliases) if c in prof_t.index]))
    for c in hinted:
        nr, ur = metric(c, "null_rate"), metric(c, "unique_ratio")
        logs.append(f"[{tname}] try hinted '{c}': null={nr:.3f}, uniq={ur:.3f}")
        if (nr <= H["PK_NULL_RATE_MAX"]) and (ur >= H["PK_UNIQUENESS_MIN"]):
            logs.append(f"[{tname}] PK=['{c}'] (hinted single)")
            return [c], logs
    # (b) any fully unique single (preserve order)
    for c in cols:
        if c not in prof_t.index:
            continue
        nr, ur = metric(c, "null_rate"), metric(c, "unique_ratio")
        if (nr <= H["PK_NULL_RATE_MAX"]) and (ur >= H["PK_UNIQUENESS_MIN"]):
            logs.append(f"[{tname}] PK=['{c}'] (unique single)")
            return [c], logs
        else:
            logs.append(f"[{tname}] reject '{c}' as PK: null={nr:.3f}, uniq={ur:.3f}")
    # (c) composite PK
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
    # (d) fallback
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
def infer_foreign_keys(tables: dict, inferred_pk: dict, cfg: dict, logger: logging.Logger) -> dict:
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
def build_profiles(tables: dict) -> dict:
    return {t: profile_table(df) for t, df in tables.items()}
def build_schema_table(tables: dict, profiles: dict, inferred_pk: dict, inferred_fk: dict, cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    rows = []
    for t, df in tables.items():
        pkset = set(inferred_pk[t])
        fks = inferred_fk.get(t, {})
        fk_cols = {v[0] for v in fks.values()}
        prof_t = profiles[t].set_index("column")
        for c in df.columns:
            sql_type     = prof_t["sql_type"].get(c, "integer")
            null_rate    = to_float_scalar(prof_t["null_rate"].get(c, 0.0))
            unique_ratio = to_float_scalar(prof_t["unique_ratio"].get(c, 1.0))
            example      = prof_t["example"].get(c, "")
            maxlen       = prof_t["maxlen"].get(c, None)
            role = []
            if c in pkset:   role.append("PK")
            if c in fk_cols: role.append("FK")
            nullable = "N" if (not math.isnan(null_rate) and np.isclose(null_rate, 0.0, atol=1e-12)) else "Y"
            rows.append({
                "table": t,
                "column": c,
                "role": ",".join(role) if role else "",
                "sql_type": sql_type,
                "nullable": nullable,
                "unique_ratio": unique_ratio,
                "example": example,
                "maxlen": maxlen,
            })
    schema_df = pd.DataFrame(rows).sort_values(["table", "role", "column"])
    out_path = Path(cfg["output_dir"]) / cfg["schema_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema_df.to_csv(out_path, index=False)
    logger.info(f"Schema saved -> {out_path}")
    return schema_df
def node_html(table: str, df: pd.DataFrame, inferred_pk: dict, inferred_fk: dict, profiles: dict) -> str:
    pkset = set(inferred_pk[table])
    fkcols = {v[0] for v in inferred_fk.get(table, {}).values()}
    prof = profiles[table].set_index("column", drop=False)
    safe_table = _html.escape(str(table))
    rows = [f'''<tr><td colspan="4" bgcolor="#eeeeee"><b>{safe_table}</b></td></tr>''']
    rows.append('<tr><td align="left"><i>col</i></td><td><i>type</i></td><td><i>PK</i>/<i>FK</i></td><td><i>ex</i></td></tr>')
    for c in df.columns:
        sqlt = prof.loc[c].sql_type if c in prof.index else "integer"
        ex   = prof.loc[c].example if c in prof.index else ""
        safe_ex = _html.escape(str(ex)[:18])
        pk   = "●" if c in pkset else ""
        fk   = "◦" if c in fkcols else ""
        rows.append(
            f'<tr>'
            f'<td align="left">{c}</td>'
            f'<td align="left">{sqlt}</td>'
            f'<td align="center">{pk}{"/"+fk if fk and pk else fk}</td>'
            f'<td align="left">{safe_ex}</td>'
            f'</tr>'
        )
    html = ( '<'
             'table BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">'
             + "".join(rows) + '</table>' )
    return html
def render_erd(tables: dict, profiles: dict, inferred_pk: dict, inferred_fk: dict, cfg: dict, logger: logging.Logger) -> Path:
    return render_erd_v2(tables, profiles, inferred_pk, inferred_fk, cfg, logger)
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
def _confidence(coverage: float, source: str) -> float:
    # Deprecated: confidence is no longer computed or displayed
    return coverage
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
def generate_edge_list(tables: dict, inferred_fk: dict, inferred_pk: dict, cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    edges = []
    for child_t, fks in inferred_fk.items():
        for _, (child_cols, parent_t, parent_pk_cols, cov) in fks.items():
            child_cols_list = list(child_cols) if isinstance(child_cols, (list, tuple)) else [child_cols]
            parent_pk_list  = list(parent_pk_cols) if isinstance(parent_pk_cols, (list, tuple)) else [parent_pk_cols]
            child_df = tables[child_t]
            parent_df = tables[parent_t]
            child_series = _as_tuple_str(child_df, child_cols_list)
            parent_series = _as_tuple_str(parent_df, parent_pk_list)
            child_set = set(child_series)
            parent_set = set(parent_series)
            child_unique = len(child_set)
            matches_unique = len(child_set & parent_set)
            coverage = float(matches_unique / child_unique) if child_unique else float("nan")
            # Confidence removed; keep cardinality only
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
                "composite": (len(child_cols_list) > 1) or (len(parent_pk_list) > 1)
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
    edge_df = edge_df.sort_values(sort_cols).drop(columns=[c for c in ["child_columns_key","parent_pk_key"] if c in edge_df.columns])
    out_path = Path(cfg["output_dir"]) / cfg["edges_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(out_path, index=False)
    logger.info(f"Edges saved -> {out_path} ({len(edge_df)} links)")
    return edge_df
# -------------------------------------------------------------
# Edges diagram renderer (from edges DataFrame)
# -------------------------------------------------------------
def render_edges_diagram(edge_df: pd.DataFrame, tables: dict | None, cfg: dict, logger: logging.Logger) -> Path:
    """Render a table-level edges diagram from edge_df.
    Nodes are tables; edges are child_table -> parent_table with labels
    including column mappings and coverage only.
    """
    spline_type = cfg.get("spline_type", "spline")
    font = cfg.get("font", "Helvetica")
    dot = Digraph("EDGES", graph_attr={
        "rankdir": cfg.get("rankdir", "LR"),
        "splines": spline_type,
        "fontname": font,
        "fontsize": "10",
        "bgcolor": "white",
    })
    dot.attr("node", shape="box", fontname=font, fontsize="10")
    dot.attr("edge", arrowsize="0.7", color="black")
    tables_seen = set(tables.keys()) if isinstance(tables, dict) else set()
    if "child_table" in edge_df.columns:
        tables_seen.update(edge_df["child_table"].dropna().astype(str).unique().tolist())
    if "parent_table" in edge_df.columns:
        tables_seen.update(edge_df["parent_table"].dropna().astype(str).unique().tolist())
    for t in sorted(tables_seen):
        dot.node(t, label=t)
    for _, row in edge_df.iterrows():
        child = str(row.get("child_table", ""))
        parent = str(row.get("parent_table", ""))
        if not child or not parent:
            continue
        child_cols_val = row.get("child_columns", [])
        parent_cols_val = row.get("parent_pk", [])
        def _fmt_side(table: str, cols) -> str:
            if isinstance(cols, (list, tuple)):
                if len(cols) == 0:
                    return table
                if len(cols) == 1:
                    return f"{table}.{cols[0]}"
                return f"{table}.({','.join(map(str, cols))})"
            return f"{table}.{cols}" if cols else table
        base = f"{_fmt_side(child, child_cols_val)} -> {_fmt_side(parent, parent_cols_val)}"
        cov = row.get("coverage", None)
        label = base
        if cov is not None and not (isinstance(cov, float) and np.isnan(cov)):
            label = f"{base} ({float(cov):.2f})"
        dot.edge(child, parent, label=label)
    out_name = cfg.get("edges_diagram_output", "EDGES.svg")
    out_stem = Path(cfg["output_dir"]) / Path(out_name).with_suffix("").name
    out_stem = Path(str(out_stem))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    dot.format = "svg"
    svg_path = Path(dot.render(str(out_stem), cleanup=True))
    logger.info(f"Edges diagram rendered -> {svg_path}")
    return svg_path
# =============================================================
# 5) Orchestration
# =============================================================
def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger = setup_logging(cfg)
    logger.info("Starting ERD pipeline (exact notebook logic)")
    t0 = time.perf_counter()
    set_heuristics_from_config(cfg)
    # Load tables
    tables = load_tables(cfg, logger)
    # Build profiles only if ERD/schema rendering is enabled
    if cfg.get("render_erd", True):
        profiles = build_profiles(tables)
    else:
        profiles = None
    # PK inference
    inferred_pk, pk_logs = {}, {}
    for t, df in tables.items():
        pk_cols, logs = pick_pk(df, t, profiles, cfg, logger)
        synth_pk = cfg.get("synthetic_pk_name", "rownum_pk")
        inferred_pk[t] = pk_cols if pk_cols else [synth_pk]
        if pk_cols == [] and synth_pk not in df.columns:
            df[synth_pk] = np.arange(len(df))
        pk_logs[t] = logs
    logger.info("Primary keys inferred for all tables.")
    # FK inference
    inferred_fk = infer_foreign_keys(tables, inferred_pk, cfg, logger)
    # Schema table and ERD (optional)
    if cfg.get("render_erd", True):
        assert profiles is not None
        _ = build_schema_table(tables, profiles, inferred_pk, inferred_fk, cfg, logger)
        _ = render_erd_v2(tables, profiles, inferred_pk, inferred_fk, cfg, logger)
    else:
        logger.info("Skipping schema table and ERD (render_erd=false)")
    # Edges
    edge_df = generate_edge_list(tables, inferred_fk, inferred_pk, cfg, logger)
    if cfg.get("render_edges_diagram", True):
        try:
            _ = render_edges_diagram(edge_df, tables, cfg, logger)
        except Exception as e:
            logger.warning(f"Failed to render edges diagram: {e}")

    logger.info("Pipeline complete.")
    # Log total duration
    elapsed = time.perf_counter() - t0
    logger.info(f"Pipeline complete in {elapsed:,.2f} seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
