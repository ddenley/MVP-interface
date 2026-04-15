import json
import pathlib
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence
import pandas as pd

__all__ = [
    "add_applicants_column"
]


def _extract_applicant_names(entry: dict) -> List[str]:
    try:
        applicants = (
            entry.get("biblio", {})
            .get("parties", {})
            .get("applicants", [])
        )
        names: Iterable[str] = (
            a.get("extracted_name", {}).get("value", "").strip()
            for a in applicants
        )
        # Keep first occurrence order & drop empties
        uniq = [n for n in OrderedDict.fromkeys(names) if n]
        return uniq or ["UNKNOWN"]
    except Exception:
        return ["UNKNOWN"]


def _build_lensid_to_applicants(jsonl_path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}

    with pathlib.Path(jsonl_path).expanduser().open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                # skip bad line but keep going – data’s messy
                continue

            lid = entry.get("lens_id")
            if not lid:
                continue

            names = _extract_applicant_names(entry)
            if lid in mapping:
                # merge while preserving order
                merged = list(OrderedDict.fromkeys(mapping[lid] + names))
                mapping[lid] = merged
            else:
                mapping[lid] = names

    return mapping


# ---- Public API ----
def add_applicants_column(df: pd.DataFrame, jsonl_path, *, col_name: str = "applicants", inplace: bool = False)\
        -> pd.DataFrame:
    if "lens_id" not in df.columns:
        raise KeyError("DataFrame is missing required 'lens_id' column")

    mapping = _build_lensid_to_applicants(jsonl_path)

    # Perform the mapping – pandas happily broadcasts lists.
    series = df["lens_id"].map(mapping).apply(
        lambda x: x if isinstance(x, Sequence) else ["UNKNOWN"]
    )

    if inplace:
        df[col_name] = series
        return df
    else:
        out = df.copy()
        out[col_name] = series
        return out