from pathlib import Path
import html as _html
import numpy as np
from graphviz import Digraph


def node_html(table: str, df, inferred_pk: dict, inferred_fk: dict, profiles: dict) -> str:
    pkset = set(inferred_pk[table])
    fkcols = {v[0] for v in inferred_fk.get(table, {}).values()}
    prof = profiles[table].set_index("column", drop=False)
    safe_table = _html.escape(str(table))
    rows = [f'''<tr><td colspan="4" bgcolor="#eeeeee"><b>{safe_table}</b></td></tr>''']
    rows.append('<tr><td align="left"><i>col</i></td><td><i>type</i></td><td><i>PK</i>/<i>FK</i></td><td><i>ex</i></td></tr>')
    for c in df.columns:
        sqlt = prof.loc[c].sql_type if c in prof.index else "integer"
        ex = prof.loc[c].example if c in prof.index else ""
        pk = "PK" if c in pkset else ""
        fk = "FK" if c in fkcols else ""
        safe_c = _html.escape(str(c))
        safe_sqlt = _html.escape(str(sqlt))
        safe_ex = _html.escape(str(ex)[:18])
        role_txt = f"{pk}{'/' + fk if fk and pk else fk}"
        rows.append(
            f'<tr>'
            f'<td align="left">{safe_c}</td>'
            f'<td align="left">{safe_sqlt}</td>'
            f'<td align="center">{role_txt}</td>'
            f'<td align="left">{safe_ex}</td>'
            f'</tr>'
        )
    html = (
        '<'
        'table BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">'
        + "".join(rows) + '</table>'
    )
    return html


def render_erd_v2(tables: dict, profiles: dict, inferred_pk: dict, inferred_fk: dict, cfg: dict, logger) -> Path:
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
            parent_pk_list = list(parent_pk) if isinstance(parent_pk, (list, tuple)) else [parent_pk]

            # Compute unique-based coverage
            child_series = (
                child_df[child_cols_list[0]].dropna().astype(str)
                if len(child_cols_list) == 1
                else child_df[child_cols_list].dropna().astype(str).apply(tuple, axis=1).astype(str)
            )
            parent_series = (
                parent_df[parent_pk_list[0]].dropna().astype(str)
                if len(parent_pk_list) == 1
                else parent_df[parent_pk_list].dropna().astype(str).apply(tuple, axis=1).astype(str)
            )
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


def render_edges_diagram(edge_df, tables: dict | None, cfg: dict, logger) -> Path:
    dot = Digraph("EDGES", graph_attr={
        "rankdir": cfg.get("rankdir", "LR"),
        "splines": cfg.get("spline_type", "spline"),
        "fontname": cfg.get("font", "Helvetica"),
        "fontsize": "10",
        "bgcolor": "white",
    })
    dot.attr("node", shape="box", fontname=cfg.get("font", "Helvetica"), fontsize="10")
    dot.attr("edge", arrowsize="0.7", color="black")

    tables_seen = set(tables.keys()) if isinstance(tables, dict) else set()
    if "child_table" in edge_df.columns:
        tables_seen.update(edge_df["child_table"].dropna().astype(str).unique().tolist())
    if "parent_table" in edge_df.columns:
        tables_seen.update(edge_df["parent_table"].dropna().astype(str).unique().tolist())
    for t in sorted(tables_seen):
        dot.node(t, label=t)

    def _fmt_side(table: str, cols) -> str:
        if isinstance(cols, (list, tuple)):
            if len(cols) == 0:
                return table
            if len(cols) == 1:
                return f"{table}.{cols[0]}"
            return f"{table}.({','.join(map(str, cols))})"
        return f"{table}.{cols}" if cols else table

    for _, row in edge_df.iterrows():
        child = str(row.get("child_table", ""))
        parent = str(row.get("parent_table", ""))
        if not child or not parent:
            continue
        child_cols_val = row.get("child_columns", [])
        parent_cols_val = row.get("parent_pk", [])
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

