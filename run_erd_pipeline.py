import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from slugify import slugify
import logging
import time
# -------------------------------------------------------------
from src.types import set_heuristic_params as _set_types_params
# =============================================================
# 0) Config + Logging
# =============================================================
from src.profile import build_profiles as _build_profiles
from src.clean import clean_table
from src.keys import pick_pk as _pick_pk, infer_foreign_keys as _infer_fk
from src.edges import generate_edge_list as _gen_edges
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
    # also propagate to types module for inference
    h = cfg.get("heuristics", {})
    _set_types_params(
        date_like_threshold=DATE_LIKE_THRESHOLD,
        column_sample_size=COLUMN_SAMPLE_SIZE,
        inference_threshold=INFERENCE_THRESHOLD,
        sample_random_seed=SAMPLE_RANDOM_SEED,
        string_length_mode=h.get("STRING_LENGTH_MODE", None),
        string_length_near_cap=h.get("STRING_LENGTH_NEAR_CAP", None),
        string_length_cap=h.get("STRING_LENGTH_CAP", None),
    )


# =============================================================
# 1) Loading & Normalization
# =============================================================
def norm_name(s: str) -> str:
    s = s.strip().replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    return slugify(s, separator="_").lower()


def load_tables(cfg: dict, logger: logging.Logger) -> dict:
    folder = Path(cfg["input_csv_dir"])
    csv_glob = cfg.get("csv_glob", "*.csv")
    if not folder.exists():
        logger.error(f"Input folder not found: {folder}")
        raise SystemExit(1)
    out = {}
    bad_records = []
    for p in folder.glob(csv_glob):
        tname = norm_name(p.stem)
        df, info = clean_table(p, cfg, logger)
        if df is None:
            rec = {"file": str(p), "reason": info.get("reason", "unknown"), "sep": info.get("sep")}
            bad_records.append(rec)
            logger.warning(f"Bad CSV skipped: {p.name} ({rec['reason']})")
            continue
        df = df.rename(columns={c: norm_name(c) for c in df.columns})
        df.attrs["_tname"] = tname
        out[tname] = df
        logger.info(f"Loaded {p.name} -> table '{tname}' (rows={len(df)}, cols={len(df.columns)})")
    if not out:
        logger.error(f"No CSV files matched in {folder} (glob='{csv_glob}').")
        raise SystemExit(1)
    if bad_records:
        import pandas as _pd
        bad_df = _pd.DataFrame(bad_records)
        bad_out = Path(cfg.get("output_dir", "./output")) / "bad_csvs.csv"
        bad_out.parent.mkdir(parents=True, exist_ok=True)
        bad_df.to_csv(bad_out, index=False)
        logger.warning(f"{len(bad_records)} bad CSV(s) logged -> {bad_out}")
        if bool(cfg.get("fail_on_bad_csv", False)):
            logger.error("fail_on_bad_csv=true and bad CSVs detected; aborting.")
            raise SystemExit(1)
    logger.info(f"Loaded {len(out)} tables: {', '.join(sorted(out.keys()))}")
    return out


# =============================================================
# 2) Schema table
# =============================================================
def build_schema_table(tables: dict, profiles: dict, inferred_pk: dict, inferred_fk: dict, cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    rows = []
    for t, df in tables.items():
        pkset = set(inferred_pk[t])
        fks = inferred_fk.get(t, {})
        fk_cols = {v[0] for v in fks.values()}
        prof_t = profiles[t].set_index("column")
        for c in df.columns:
            sql_type = prof_t["sql_type"].get(c, "integer")
            null_rate = float(prof_t["null_rate"].get(c, 0.0)) if "null_rate" in prof_t else 0.0
            unique_ratio = float(prof_t["unique_ratio"].get(c, 1.0)) if "unique_ratio" in prof_t else 1.0
            example = prof_t["example"].get(c, "") if "example" in prof_t else ""
            raw_maxlen = prof_t["maxlen"].get(c, None) if "maxlen" in prof_t else None
            # Guard against NaN maxlen values
            if raw_maxlen is None or (isinstance(raw_maxlen, float) and math.isnan(raw_maxlen)):
                maxlen_out = None
            else:
                try:
                    maxlen_out = int(raw_maxlen)
                except Exception:
                    maxlen_out = None
            # Format NaN ratios as empty strings for CSV output
            unique_ratio_out = ("" if (isinstance(unique_ratio, float) and math.isnan(unique_ratio)) else unique_ratio)
            role = []
            if c in pkset:
                role.append("PK")
            if c in fk_cols:
                role.append("FK")
            nullable = "N" if (not math.isnan(null_rate) and np.isclose(null_rate, 0.0, atol=1e-12)) else "Y"
            rows.append(dict(
                table=t,
                column=c,
                role=",".join(role) if role else "",
                sql_type=str(sql_type),
                nullable=nullable,
                unique_ratio=unique_ratio_out,
                example=example,
                maxlen=maxlen_out,
            ))
    schema_df = pd.DataFrame(rows).sort_values(["table", "role", "column"])
    out_path = Path(cfg["output_dir"]) / cfg["schema_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema_df.to_csv(out_path, index=False)
    logger.info(f"Schema saved -> {out_path}")
    return schema_df


# =============================================================
# 3) Orchestration
# =============================================================
def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger = setup_logging(cfg)
    logger.info("Starting ERD pipeline")
    t0 = time.perf_counter()
    set_heuristics_from_config(cfg)

    # Load & clean tables
    tables = load_tables(cfg, logger)

    # Profiles (if ERD enabled)
    if cfg.get("render_erd", True):
        profiles = _build_profiles(tables)
    else:
        profiles = None

    # PK inference
    inferred_pk, pk_logs = {}, {}
    for t, df in tables.items():
        pk_cols, logs = _pick_pk(df, t, profiles, cfg, logger)
        synth_pk = cfg.get("synthetic_pk_name", "rownum_pk")
        inferred_pk[t] = pk_cols if pk_cols else [synth_pk]
        if pk_cols == [] and synth_pk not in df.columns:
            df[synth_pk] = np.arange(len(df))
        pk_logs[t] = logs
    logger.info("Primary keys inferred for all tables.")

    # FK inference
    inferred_fk = _infer_fk(tables, inferred_pk, cfg, logger)

    # Schema + ERD
    if cfg.get("render_erd", True):
        assert profiles is not None
        _ = build_schema_table(tables, profiles, inferred_pk, inferred_fk, cfg, logger)
        from src.render import render_erd_v2 as _render_erd
        _ = _render_erd(tables, profiles, inferred_pk, inferred_fk, cfg, logger)
    else:
        logger.info("Skipping schema table and ERD (render_erd=false)")

    # Edges
    edge_df = _gen_edges(tables, inferred_fk, inferred_pk, cfg, logger)
    if cfg.get("render_edges_diagram", True):
        try:
            from src.render import render_edges_diagram as _render_edges
            _ = _render_edges(edge_df, tables, cfg, logger)
        except Exception as e:
            logger.warning(f"Failed to render edges diagram: {e}")

    logger.info("Pipeline complete.")
    elapsed = time.perf_counter() - t0
    logger.info(f"Pipeline complete in {elapsed:,.2f} seconds.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)


