from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import csv
import pandas as pd

from .io_csv import choose_sep_for_path, resolve_quoting, read_csv_fast


def _prevalidate_csv_structure(path: Path, sep: str, quoting_kwargs: dict, cfg: dict, logger) -> Dict[str, Any]:
    """Lightweight structural validation using Python's csv.reader.

    Stops early if bad line count exceeds max_bad_lines. Returns a dict with:
      { ok: bool, expected_cols: int|None, bad_lines: int }
    """
    result: Dict[str, Any] = {"ok": True, "expected_cols": None, "bad_lines": 0}
    max_bad = int(cfg.get("max_bad_lines", 50) or 50)
    try:
        reader_kwargs = {
            "delimiter": sep,
            "quoting": quoting_kwargs.get("quoting", csv.QUOTE_MINIMAL),
        }
        if "quotechar" in quoting_kwargs:
            reader_kwargs["quotechar"] = quoting_kwargs["quotechar"]
        if "doublequote" in quoting_kwargs:
            reader_kwargs["doublequote"] = bool(quoting_kwargs["doublequote"])
        if "escapechar" in quoting_kwargs and quoting_kwargs["escapechar"] is not None:
            reader_kwargs["escapechar"] = quoting_kwargs["escapechar"]

        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            rdr = csv.reader(f, **reader_kwargs)
            expected: Optional[int] = None
            for row in rdr:
                if not row:
                    continue
                if expected is None:
                    expected = len(row)
                    result["expected_cols"] = expected
                    # skip to next
                    continue
                if len(row) != expected:
                    result["bad_lines"] += 1
                    if result["bad_lines"] > max_bad:
                        result["ok"] = False
                        break
        return result
    except Exception as e:
        logger.debug(f"Prevalidation skipped due to error: {e}")
        return result  # default ok=True, let pandas try


def clean_table(path: Path, cfg: dict, logger) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Attempt to read and validate a CSV into a clean DataFrame.

    Returns (df, info). If df is None, info['reason'] explains why.
    """
    info: Dict[str, Any] = {"file": str(path)}
    try:
        sep = choose_sep_for_path(path, cfg)
        sep_use: str = sep if sep is not None else ","
        quoting_kwargs = resolve_quoting(cfg)
        info["sep"] = sep_use
        info.update({k: quoting_kwargs.get(k) for k in ("quoting", "quotechar", "doublequote", "escapechar") if k in quoting_kwargs})

        # Optional structural validation before full read
        if bool(cfg.get("strict_csv_validation", True)):
            v = _prevalidate_csv_structure(path, sep_use, quoting_kwargs, cfg, logger)
            info.update({"expected_cols": v.get("expected_cols"), "bad_lines": v.get("bad_lines")})
            if not v.get("ok", True):
                info["reason"] = "structural_inconsistent_columns"
                return None, info

        # Read
        df = read_csv_fast(
            path,
            sep=sep_use,
            quoting_kwargs=quoting_kwargs,
            nrows=cfg.get("heuristics", {}).get("SAMPLE_ROWS", None),
        )

        # Basic validation
        if df is None or df.shape[1] <= 1:
            info["reason"] = "single_or_zero_column_after_parse"
            return None, info
        if df.shape[0] == 0:
            info["reason"] = "zero_rows"
            return None, info

        # Normalize headers
        df.columns = [str(c) for c in df.columns]

        # Optionally write a cleaned CSV artifact
        if bool(cfg.get("write_clean_csv", False)):
            try:
                clean_dir = Path(cfg.get("clean_output_dir", Path(cfg.get("output_dir", "./output")) / "clean"))
                clean_dir.mkdir(parents=True, exist_ok=True)
                out_path = clean_dir / f"{path.stem}.csv"
                df.to_csv(
                    out_path,
                    index=False,
                    sep=sep_use,
                    quoting=quoting_kwargs.get("quoting", csv.QUOTE_MINIMAL),
                    quotechar=quoting_kwargs.get("quotechar", '"'),
                    doublequote=quoting_kwargs.get("doublequote", True),
                    escapechar=quoting_kwargs.get("escapechar", None),
                )
                info["clean_path"] = str(out_path)
            except Exception as ce:
                logger.debug(f"Writing cleaned CSV failed for {path.name}: {ce}")

        return df, info
    except Exception as e:
        logger.warning(f"Clean failed for {path}: {e}")
        info["reason"] = f"exception: {e}"
        return None, info



