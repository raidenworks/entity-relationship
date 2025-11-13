from __future__ import annotations

from typing import Union, Optional, List
import logging
import warnings

import numpy as np
import pandas as pd


# Heuristic parameters (overridden by set_heuristic_params from main)
DATE_LIKE_THRESHOLD: float = 0.5
COLUMN_SAMPLE_SIZE: int = 5000
INFERENCE_THRESHOLD: float = 0.98
SAMPLE_RANDOM_SEED: Optional[int] = None
# String length inference controls
STRING_LENGTH_MODE: str = "sample"  # sample | full | hybrid
STRING_LENGTH_NEAR_CAP: int = 230
STRING_LENGTH_CAP: int = 255


def set_heuristic_params(
    *,
    date_like_threshold: float,
    column_sample_size: int,
    inference_threshold: float,
    sample_random_seed: Optional[int],
    string_length_mode: Optional[str] = None,
    string_length_near_cap: Optional[int] = None,
    string_length_cap: Optional[int] = None,
) -> None:
    global DATE_LIKE_THRESHOLD, COLUMN_SAMPLE_SIZE, INFERENCE_THRESHOLD, SAMPLE_RANDOM_SEED
    global STRING_LENGTH_MODE, STRING_LENGTH_NEAR_CAP, STRING_LENGTH_CAP
    DATE_LIKE_THRESHOLD = float(date_like_threshold)
    COLUMN_SAMPLE_SIZE = int(column_sample_size)
    INFERENCE_THRESHOLD = float(inference_threshold)
    SAMPLE_RANDOM_SEED = sample_random_seed
    if string_length_mode:
        STRING_LENGTH_MODE = str(string_length_mode).lower()
    if string_length_near_cap is not None:
        STRING_LENGTH_NEAR_CAP = int(string_length_near_cap)
    if string_length_cap is not None:
        STRING_LENGTH_CAP = int(string_length_cap)


def _sample_nonnull(s: pd.Series, max_n: int) -> pd.Series:
    nonnull = s.dropna()
    if len(nonnull) <= max_n:
        return nonnull
    return nonnull.sample(n=max_n, random_state=SAMPLE_RANDOM_SEED)


def is_likely_datetime(s: pd.Series, threshold: float | None = None) -> bool:
    thr = DATE_LIKE_THRESHOLD if threshold is None else threshold
    nonnull = s.dropna()
    if len(nonnull) == 0:
        return False
    sample = _sample_nonnull(nonnull, COLUMN_SAMPLE_SIZE).astype(str)
    has_date_sep = sample.str.contains(r"[-/]", na=False).mean()
    has_time_sep = sample.str.contains(r":", na=False).mean()
    has_iso_t = sample.str.contains(r"[Tt]", na=False).mean()
    has_month = sample.str.contains(r"(?i)jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", na=False).mean()
    is_digits = sample.str.fullmatch(r"\d+", na=False).mean()
    len8 = sample.str.len().eq(8).mean()
    len10 = sample.str.len().eq(10).mean()
    len13 = sample.str.len().eq(13).mean()
    len14 = sample.str.len().eq(14).mean()
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
    sample = _sample_nonnull(nonnull, COLUMN_SAMPLE_SIZE)
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
    # Fallback with chunked progress for slow path
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
    # Harden numeric args
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
            parts.append(parsed_chunk.to_numpy(dtype="datetime64[ns]", copy=False))
        except TypeError as te:
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
            try:
                parsed_chunk = pd.to_datetime(chunk.astype(str), errors="coerce")
                parts.append(parsed_chunk.to_numpy(dtype="datetime64[ns]", copy=False))
            except Exception as te2:
                log.error("Fallback string parse failed: %s", te2)
                parts.append(np.full(int(jj - ii), np.datetime64("NaT"), dtype="datetime64[ns]"))
        except Exception as ex:
            log.error("Datetime chunk parse error: %s | i=%s j=%s fmt=%s unit=%s", ex, ii, jj, fmt, unit)
            parsed_chunk = pd.to_datetime(chunk, errors="coerce")
            parts.append(parsed_chunk.to_numpy(dtype="datetime64[ns]", copy=False))

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


def infer_sql_type(s: pd.Series) -> str:
    thr = INFERENCE_THRESHOLD
    sample_nonnull = _sample_nonnull(s, COLUMN_SAMPLE_SIZE)
    sample_str = sample_nonnull.astype(str) if len(sample_nonnull) else sample_nonnull
    # booleans
    if len(sample_nonnull) and sample_nonnull.isin([0, 1, True, False]).mean() >= thr:
        return "boolean"
    # ints
    if pd.api.types.is_integer_dtype(s) or (len(sample_nonnull) and pd.api.types.is_integer_dtype(sample_nonnull)):
        vmax = sample_nonnull.astype("int64", errors="ignore").abs().max() if len(sample_nonnull) else 0
        return "integer" if (pd.isna(vmax) or vmax <= 2_147_483_647) else "bigint"
    if pd.api.types.is_object_dtype(s):
        if len(sample_nonnull) and (sample_str.str.fullmatch(r"[+-]?\d+").mean() >= thr):
            vmax = pd.to_numeric(sample_str, errors="coerce").abs().max()
            return "integer" if (pd.isna(vmax) or vmax <= 2_147_483_647) else "bigint"
    # floats
    if pd.api.types.is_float_dtype(s) or (
        pd.api.types.is_object_dtype(s)
        and len(sample_nonnull)
        and sample_str.str.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?").mean() >= thr
    ):
        return "double"
    # datetimes
    # 1) If already a pandas datetime dtype, classify as date/timestamp regardless of null rate
    if pd.api.types.is_datetime64_any_dtype(s):
        parsed = s
        try:
            no_tz = parsed.dt.tz_convert(None)
        except Exception:
            # either already naive or tz-localize-able
            try:
                no_tz = parsed.dt.tz_localize(None)
            except Exception:
                no_tz = parsed
        nn = no_tz.dropna()
        if len(nn) == 0:
            return "timestamp"
        is_date_only = (nn.dt.normalize() == nn).mean() > 0.95
        return "date" if is_date_only else "timestamp"

    # 2) For non-datetime object columns, attempt parsing when likely; compute success over non-null values
    if not is_likely_datetime(s):
        parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    else:
        parsed = to_datetime_fast(s)
    denom = max(1, int(s.notna().sum()))
    frac_nonnull = int(parsed.notna().sum()) / denom
    if frac_nonnull >= 0.98:
        try:
            no_tz = parsed.dt.tz_convert(None)
        except Exception:
            try:
                no_tz = parsed.dt.tz_localize(None)
            except Exception:
                no_tz = parsed
        nn = no_tz.dropna()
        if len(nn) == 0:
            return "timestamp"
        is_date_only = (nn.dt.normalize() == nn).mean() > 0.95
        return "date" if is_date_only else "timestamp"
    # strings
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        mode = STRING_LENGTH_MODE
        cap = int(STRING_LENGTH_CAP)
        if not len(sample_nonnull):
            return f"varchar(0)"
        # sample length
        lens_sample = sample_str.str.len()
        m_sample = int(lens_sample.max()) if len(lens_sample) else 0
        m_final = m_sample
        if mode == "full" or (mode == "hybrid" and m_sample >= int(STRING_LENGTH_NEAR_CAP)):
            # full scan for exact length when needed
            try:
                m_full = int(s.dropna().astype(str).str.len().max())
                m_final = m_full
            except Exception:
                m_final = m_sample
        if m_final <= cap:
            return f"varchar({m_final})"
        return "text"
    return str(s.dtype)
