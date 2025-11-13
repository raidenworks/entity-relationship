import csv
from pathlib import Path
from typing import Union, Optional

import pandas as pd


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


def choose_sep_for_path(p: Path, cfg: dict) -> Optional[str]:
    """Resolve CSV separator via config and auto-detection.

    Supports root-level config keys:
      - csv_sep: 'auto' or explicit sep (',', '|', '\t', ';')
      - csv_sep_candidates: list of candidates for auto
      - csv_separators: list of {pattern, sep} overrides
    """
    import fnmatch

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


def resolve_quoting(cfg: dict) -> dict:
    """Map config quoting options to pandas read_csv kwargs."""
    kwargs: dict = {}
    quoting = (cfg.get("csv_quoting") or "minimal").lower()
    map_q = {
        "minimal": csv.QUOTE_MINIMAL,
        "none": csv.QUOTE_NONE,
        "all": csv.QUOTE_ALL,
        "nonnumeric": csv.QUOTE_NONNUMERIC,
    }
    kwargs["quoting"] = map_q.get(quoting, csv.QUOTE_MINIMAL)
    if cfg.get("csv_quotechar") is not None:
        kwargs["quotechar"] = cfg.get("csv_quotechar")
    if cfg.get("csv_doublequote") is not None:
        kwargs["doublequote"] = bool(cfg.get("csv_doublequote"))
    if cfg.get("csv_escapechar") is not None:
        kwargs["escapechar"] = cfg.get("csv_escapechar")
    return kwargs


def read_csv_fast(path: Union[str, Path], *, nrows: Optional[int] = None, sep: Optional[str] = None, quoting_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Read CSV preferring PyArrow when available, with safe fallback."""
    quoting_kwargs = quoting_kwargs or {}
    try:
        return pd.read_csv(path, nrows=nrows, engine="pyarrow", sep=sep, **quoting_kwargs)
    except Exception:
        return pd.read_csv(path, nrows=nrows, sep=sep, **quoting_kwargs)


