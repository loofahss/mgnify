#!/usr/bin/env python3
"""Compare entries between two sequence/prediction files.

Supported formats:
- FASTA: compares by record ID and sequence (for FASTA vs FASTA)
- TSV/CSV: compares by key column and row content (for TSV/CSV vs TSV/CSV)
- Mixed FASTA vs TSV/CSV: compares only record IDs

Examples:
  python compare_entries.py --left pys/predictions/pred_1652.tsv --right pys/predictions/output.fa
  python compare_entries.py --left a.tsv --right b.tsv --key-column protein_id
  python compare_entries.py --left a.fa --right b.fa --ignore-order
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


FASTA_EXTS = {".fa", ".fasta", ".faa", ".fna"}
TABLE_EXTS = {".tsv", ".csv"}


@dataclass
class CompareResult:
    left_total: int
    right_total: int
    left_unique: int
    right_unique: int
    common: int
    same_content_common: int
    different_content_common: int
    only_left_ids: List[str]
    only_right_ids: List[str]
    different_ids: List[str]
    left_duplicates: Dict[str, int]
    right_duplicates: Dict[str, int]


def detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in FASTA_EXTS:
        return "fasta"
    if ext in TABLE_EXTS:
        return "table"

    # Fallback by peeking first non-empty line.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                return "fasta"
            if "\t" in s or "," in s:
                return "table"
            break
    raise ValueError(f"Cannot infer file format: {path}")


def parse_fasta(path: Path) -> Tuple[Dict[str, str], Dict[str, int], int]:
    records: Dict[str, str] = {}
    counts: Counter[str] = Counter()

    current_id = None
    current_seq: List[str] = []
    total = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seq = "".join(current_seq)
                    # Keep first occurrence for content compare; track duplicates separately.
                    if current_id not in records:
                        records[current_id] = seq
                header = line[1:].strip()
                record_id = header.split()[0] if header else ""
                if not record_id:
                    continue
                current_id = record_id
                current_seq = []
                counts[current_id] += 1
                total += 1
            else:
                if current_id is not None:
                    current_seq.append(line)

    if current_id is not None and current_id not in records:
        records[current_id] = "".join(current_seq)

    duplicates = {k: v for k, v in counts.items() if v > 1}
    return records, duplicates, total


def parse_table(path: Path, key_column: str) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, int], int]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    records: Dict[str, Tuple[str, ...]] = {}
    counts: Counter[str] = Counter()
    total = 0

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No header found in table file: {path}")
        if key_column not in reader.fieldnames:
            raise ValueError(
                f"Key column '{key_column}' not found in {path}. Available columns: {reader.fieldnames}"
            )

        value_columns = [c for c in reader.fieldnames if c != key_column]

        for row in reader:
            record_id = (row.get(key_column) or "").strip()
            if not record_id:
                continue
            content = tuple((row.get(col) or "") for col in value_columns)
            if record_id not in records:
                records[record_id] = content
            counts[record_id] += 1
            total += 1

    duplicates = {k: v for k, v in counts.items() if v > 1}
    return records, duplicates, total


def compare_records(
    left_records: Dict[str, object],
    right_records: Dict[str, object],
    left_dups: Dict[str, int],
    right_dups: Dict[str, int],
    left_total: int,
    right_total: int,
) -> CompareResult:
    left_ids = set(left_records.keys())
    right_ids = set(right_records.keys())

    only_left = sorted(left_ids - right_ids)
    only_right = sorted(right_ids - left_ids)
    common = sorted(left_ids & right_ids)

    same_common = 0
    diff_ids: List[str] = []
    for rid in common:
        if left_records[rid] == right_records[rid]:
            same_common += 1
        else:
            diff_ids.append(rid)

    return CompareResult(
        left_total=left_total,
        right_total=right_total,
        left_unique=len(left_ids),
        right_unique=len(right_ids),
        common=len(common),
        same_content_common=same_common,
        different_content_common=len(diff_ids),
        only_left_ids=only_left,
        only_right_ids=only_right,
        different_ids=diff_ids,
        left_duplicates=left_dups,
        right_duplicates=right_dups,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare entries in two files (FASTA/TSV/CSV).")
    parser.add_argument("--left", required=True, help="Path to left file")
    parser.add_argument("--right", required=True, help="Path to right file")
    parser.add_argument(
        "--key-column",
        default="protein_id",
        help="Key column used for TSV/CSV files (default: protein_id)",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="How many example IDs to print for differences (default: 20)",
    )
    args = parser.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)

    if not left_path.exists() or not right_path.exists():
        print("Error: One or both files do not exist.")
        return 2

    left_fmt = detect_format(left_path)
    right_fmt = detect_format(right_path)

    if left_fmt == "fasta":
        left_records, left_dups, left_total = parse_fasta(left_path)
    else:
        left_records, left_dups, left_total = parse_table(left_path, args.key_column)

    if right_fmt == "fasta":
        right_records, right_dups, right_total = parse_fasta(right_path)
    else:
        right_records, right_dups, right_total = parse_table(right_path, args.key_column)

    # Mixed formats: compare IDs only.
    if left_fmt != right_fmt:
        left_id_only = {k: True for k in left_records.keys()}
        right_id_only = {k: True for k in right_records.keys()}
        result = compare_records(
            left_id_only, right_id_only, left_dups, right_dups, left_total, right_total
        )
        compare_mode = "ID-only (mixed formats)"
    else:
        result = compare_records(
            left_records, right_records, left_dups, right_dups, left_total, right_total
        )
        compare_mode = "Full-content"

    print(f"Left file : {left_path} [{left_fmt}]")
    print(f"Right file: {right_path} [{right_fmt}]")
    print(f"Mode      : {compare_mode}")
    print("-" * 60)
    print(f"Left total entries (including duplicates) : {result.left_total}")
    print(f"Right total entries (including duplicates): {result.right_total}")
    print(f"Left unique IDs : {result.left_unique}")
    print(f"Right unique IDs: {result.right_unique}")
    print(f"Common IDs      : {result.common}")
    print(f"Common IDs with same content: {result.same_content_common}")
    print(f"Common IDs with different content: {result.different_content_common}")
    print(f"Only in left : {len(result.only_left_ids)}")
    print(f"Only in right: {len(result.only_right_ids)}")

    if result.left_duplicates:
        print(f"Left duplicate IDs: {len(result.left_duplicates)}")
    if result.right_duplicates:
        print(f"Right duplicate IDs: {len(result.right_duplicates)}")

    limit = max(0, args.show)

    if limit and result.only_left_ids:
        print("\nExamples only in left:")
        for rid in result.only_left_ids[:limit]:
            print(rid)

    if limit and result.only_right_ids:
        print("\nExamples only in right:")
        for rid in result.only_right_ids[:limit]:
            print(rid)

    if limit and result.different_ids:
        print("\nExamples with different content:")
        for rid in result.different_ids[:limit]:
            print(rid)

    fully_same = (
        result.different_content_common == 0
        and len(result.only_left_ids) == 0
        and len(result.only_right_ids) == 0
    )
    print("\nResult:", "ALL ENTRIES MATCH" if fully_same else "ENTRIES DIFFER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
