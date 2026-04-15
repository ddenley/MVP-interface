from __future__ import annotations
import itertools
import pathlib
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

EXCLUDE_TOPIC_IDS = {-1}

__all__ = [
    # ── binning helpers ────────────────────────────────────────────────────
    "add_time_bins",
    # ── 1‑D value distributions ────────────────────────────────────────────
    "build_topic_value_matrix",
    "topic_distribution",
    # ── 2‑D value distributions (e.g. company × year) ──────────────────────
    "build_topic_two_value_matrix",
    # ── normalisation utilities ────────────────────────────────────────────
    "normalize_distribution",
    # ── batch‑precompute to Parquet ────────────────────────────────────────
    "precompute_distribution",
]

# ---------------------------------------------------------------------------
# Time‑bin helpers
# ---------------------------------------------------------------------------

def add_time_bins(
    df: pd.DataFrame,
    *,
    date_col: str = "date_published",
    year_col: str = "year_bin",
    month_col: str = "month_bin",
) -> pd.DataFrame:
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not in DataFrame")

    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df[year_col] = df[date_col].dt.year.astype("Int32").astype("string")
    df[month_col] = df[date_col].dt.to_period("M").astype("string")
    return df

# ---------------------------------------------------------------------------
# Internal helpers for counting
# ---------------------------------------------------------------------------

def _extract_unique_topics(row: pd.Series, topic_cols: Sequence[str]) -> List[int]:
    unique: set[int] = set()
    for col in topic_cols:
        val = row.get(col)
        if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
            for x in val:
                if x == x:  # skip NaN
                    xi = int(x)
                    if xi not in EXCLUDE_TOPIC_IDS:
                        unique.add(xi)
    return list(unique)


def _explode_col_as_list(col: pd.Series) -> pd.Series:
    if col.dtype == "O":
        return col.apply(lambda x: list(x) if isinstance(x, (list, tuple, set)) else ([x] if x == x else []))
    return col.apply(lambda x: [x] if x == x else [])

# ---------------------------------------------------------------------------
# 1‑D value counts  → (topic_id, value)
# ---------------------------------------------------------------------------

def build_topic_value_matrix(
    df: pd.DataFrame,
    *,
    value_col: str,
    topic_cols: Sequence[str] | None = None,
    min_count: int | None = None,
) -> pd.DataFrame:
    if "lens_id" not in df.columns:
        raise KeyError("DataFrame must have a 'lens_id' column")

    if topic_cols is None:
        topic_cols = [
            "abstract_paragraphs_topic_ids",
            "claims_paragraphs_topic_ids",
            "description_paragraphs_topic_ids",
        ]
    missing = [c for c in topic_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Topic columns missing: {missing}")

    if value_col not in df.columns:
        raise KeyError(f"'{value_col}' not found in DataFrame")

    topics_per_row = df.apply(_extract_unique_topics, axis=1, args=(topic_cols,))
    values_per_row = _explode_col_as_list(df[value_col])

    counter: Dict[Tuple[int, str], int] = Counter()
    for topics, vals in zip(topics_per_row, values_per_row):
        if not topics or not vals:
            continue
        for tid, val in itertools.product(topics, set(vals)):
            counter[(tid, str(val))] += 1

    if not counter:
        return pd.DataFrame(columns=["topic_id", "value", "count"])

    t_ids, vals, counts = zip(*((k[0], k[1], v) for k, v in counter.items()))
    out = pd.DataFrame({"topic_id": t_ids, "value": vals, "count": counts})
    if min_count is not None:
        out = out[out["count"] >= min_count]
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def topic_distribution(
    df: pd.DataFrame,
    *,
    value_col: str,
    topic_cols: Sequence[str] | None = None,
    normalize: bool = False,
) -> pd.DataFrame:
    long_df = build_topic_value_matrix(df, value_col=value_col, topic_cols=topic_cols)
    if long_df.empty:
        return pd.DataFrame()

    pivot = long_df.pivot(index="topic_id", columns="value", values="count").fillna(0).astype(int)
    if normalize:
        pivot = pivot.div(pivot.sum(axis=1), axis=0)
    return pivot

# ---------------------------------------------------------------------------
# 2‑D value counts  → (topic_id, value_a, value_b)
# ---------------------------------------------------------------------------

def build_topic_two_value_matrix(
    df: pd.DataFrame,
    *,
    value_a_col: str,
    value_b_col: str,
    topic_cols: Sequence[str] | None = None,
    min_count: int | None = None,
) -> pd.DataFrame:
    for col in (value_a_col, value_b_col):
        if col not in df.columns:
            raise KeyError(f"'{col}' not in DataFrame")

    if topic_cols is None:
        topic_cols = [
            "abstract_paragraphs_topic_ids",
            "claims_paragraphs_topic_ids",
            "description_paragraphs_topic_ids",
        ]

    topics_per = df.apply(_extract_unique_topics, axis=1, args=(topic_cols,))
    vals_a_per = _explode_col_as_list(df[value_a_col])
    vals_b_per = _explode_col_as_list(df[value_b_col])

    counter: Dict[Tuple[int, str, str], int] = Counter()
    for topics, a_vals, b_vals in zip(topics_per, vals_a_per, vals_b_per):
        if not topics or not a_vals or not b_vals:
            continue
        for tid, a, b in itertools.product(topics, set(a_vals), set(b_vals)):
            counter[(tid, str(a), str(b))] += 1

    if not counter:
        return pd.DataFrame(columns=["topic_id", value_a_col, value_b_col, "count"])

    t_ids, a_vals, b_vals, counts = zip(*((k[0], k[1], k[2], v) for k, v in counter.items()))
    out = pd.DataFrame({"topic_id": t_ids, value_a_col: a_vals, value_b_col: b_vals, "count": counts})
    if min_count is not None:
        out = out[out["count"] >= min_count]
    return out.sort_values("count", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_distribution(df: pd.DataFrame, *, axis: str = "topic") -> pd.DataFrame:
    if axis == "topic":
        return df.div(df.sum(axis=1), axis=0)
    if axis == "value":
        return df.T.div(df.T.sum(axis=1), axis=0).T
    raise ValueError("axis must be 'topic' or 'value'")

# ---------------------------------------------------------------------------
# Batch pre‑compute to Parquet
# ---------------------------------------------------------------------------

def precompute_distribution(
    df: pd.DataFrame,
    *,
    value_col: str | None = None,
    second_value_col: str | None = None,
    date_col: str | None = None,
    out_path: str | pathlib.Path,
    topic_cols: Sequence[str] | None = None,
    min_count: int | None = None,
) -> pathlib.Path:
    out_path = pathlib.Path(out_path).expanduser()

    if date_col and ("year_bin" not in df.columns or "month_bin" not in df.columns):
        df = add_time_bins(df, date_col=date_col)

    if second_value_col:
        long_df = build_topic_two_value_matrix(
            df,
            value_a_col=value_col,
            value_b_col=second_value_col,
            topic_cols=topic_cols,
            min_count=min_count,
        )
    else:
        long_df = build_topic_value_matrix(
            df,
            value_col=value_col,
            topic_cols=topic_cols,
            min_count=min_count,
        )

    long_df.to_parquet(out_path, compression="snappy")
    return out_path
