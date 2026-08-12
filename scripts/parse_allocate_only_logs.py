#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
"""
Filter ALLOCATE_ONLY scheduling logs into a table.

Parses lines like:
  ALLOCATE_ONLY req_id=... role=prefill ins=1 ep=10
  active_requests=3 active_tokens=128.45
  prefill_endpoints=1:10=req:3,tokens:128.45;1:11=req:1,tokens:40.00
  decode_endpoints=2:20=req:5,tokens:200.00 score=... fast_path=...

Usage:
  python3 scripts/parse_allocate_only_logs.py coordinator.log
  python3 scripts/parse_allocate_only_logs.py coordinator.log --format csv -o out.csv
  python3 scripts/parse_allocate_only_logs.py coordinator.log --per-endpoint
  cat coordinator.log | python3 scripts/parse_allocate_only_logs.py -
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


_ALLOCATE_RE = re.compile(
    r"ALLOCATE_ONLY\s+"
    r"req_id=(?P<req_id>\S+)\s+"
    r"role=(?P<role>\S+)\s+"
    r"ins=(?P<ins>\S+)\s+"
    r"ep=(?P<ep>\S+)\s+"
    r"active_requests=(?P<active_requests>\S+)\s+"
    r"active_tokens=(?P<active_tokens>\S+)\s+"
    r"prefill_endpoints=(?P<prefill_endpoints>\S+)\s+"
    r"decode_endpoints=(?P<decode_endpoints>\S+)\s+"
    r"score=(?P<score>\S+)\s+"
    r"fast_path=(?P<fast_path>\S+)"
)

_ENDPOINT_STAT_RE = re.compile(
    r"(?P<ins_ep>[^=;]+)=req:(?P<req>-?\d+),tokens:(?P<tokens>-?\d+(?:\.\d+)?)"
)

_SUMMARY_COLUMNS = [
    "req_id",
    "role",
    "ins",
    "ep",
    "active_requests",
    "active_tokens",
    "prefill_endpoints",
    "decode_endpoints",
    "score",
    "fast_path",
]

_PER_ENDPOINT_COLUMNS = [
    "req_id",
    "role",
    "selected_ins",
    "selected_ep",
    "pool",
    "ins",
    "ep",
    "active_requests",
    "active_tokens",
    "selected_active_requests",
    "selected_active_tokens",
    "score",
    "fast_path",
]


@dataclass(frozen=True)
class EndpointStat:
    ins: str
    ep: str
    active_requests: int
    active_tokens: float


def parse_endpoint_stats(blob: str) -> list[EndpointStat]:
    """Parse '1:10=req:3,tokens:128.45;1:11=req:1,tokens:40.00' or 'none'."""
    if not blob or blob == "none":
        return []
    stats: list[EndpointStat] = []
    for match in _ENDPOINT_STAT_RE.finditer(blob):
        ins_ep = match.group("ins_ep")
        if ":" not in ins_ep:
            continue
        ins, ep = ins_ep.split(":", 1)
        stats.append(
            EndpointStat(
                ins=ins,
                ep=ep,
                active_requests=int(match.group("req")),
                active_tokens=float(match.group("tokens")),
            )
        )
    return stats


def parse_allocate_line(line: str) -> dict[str, str] | None:
    match = _ALLOCATE_RE.search(line)
    if not match:
        return None
    return match.groupdict()


def iter_log_lines(paths: list[str]) -> Iterable[str]:
    if not paths or paths == ["-"]:
        yield from sys.stdin
        return
    for path in paths:
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            yield from fh


def build_summary_rows(records: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{col: rec.get(col, "") for col in _SUMMARY_COLUMNS} for rec in records]


def build_per_endpoint_rows(records: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rec in records:
        for pool, blob in (
            ("prefill", rec.get("prefill_endpoints", "")),
            ("decode", rec.get("decode_endpoints", "")),
        ):
            stats = parse_endpoint_stats(blob)
            if not stats:
                rows.append(
                    {
                        "req_id": rec.get("req_id", ""),
                        "role": rec.get("role", ""),
                        "selected_ins": rec.get("ins", ""),
                        "selected_ep": rec.get("ep", ""),
                        "pool": pool,
                        "ins": "",
                        "ep": "",
                        "active_requests": "",
                        "active_tokens": "",
                        "selected_active_requests": rec.get("active_requests", ""),
                        "selected_active_tokens": rec.get("active_tokens", ""),
                        "score": rec.get("score", ""),
                        "fast_path": rec.get("fast_path", ""),
                    }
                )
                continue
            for stat in stats:
                rows.append(
                    {
                        "req_id": rec.get("req_id", ""),
                        "role": rec.get("role", ""),
                        "selected_ins": rec.get("ins", ""),
                        "selected_ep": rec.get("ep", ""),
                        "pool": pool,
                        "ins": stat.ins,
                        "ep": stat.ep,
                        "active_requests": str(stat.active_requests),
                        "active_tokens": f"{stat.active_tokens:.2f}",
                        "selected_active_requests": rec.get("active_requests", ""),
                        "selected_active_tokens": rec.get("active_tokens", ""),
                        "score": rec.get("score", ""),
                        "fast_path": rec.get("fast_path", ""),
                    }
                )
    return rows


def render_markdown(columns: list[str], rows: list[dict[str, str]], out: TextIO) -> None:
    if not rows:
        out.write("No ALLOCATE_ONLY records found.\n")
        return
    out.write("| " + " | ".join(columns) + " |\n")
    out.write("| " + " | ".join("---" for _ in columns) + " |\n")
    for row in rows:
        out.write("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |\n")


def render_csv(columns: list[str], rows: list[dict[str, str]], out: TextIO) -> None:
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def render_table(columns: list[str], rows: list[dict[str, str]], out: TextIO) -> None:
    if not rows:
        out.write("No ALLOCATE_ONLY records found.\n")
        return
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    out.write(header + "\n")
    out.write(sep + "\n")
    for row in rows:
        out.write(
            "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
            + "\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter ALLOCATE_ONLY logs into a table of endpoint workload data."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        default=["-"],
        help="Log file path(s). Use - or omit for stdin.",
    )
    parser.add_argument(
        "--per-endpoint",
        action="store_true",
        help="Expand prefill/decode endpoint snapshots into one row per endpoint.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "csv"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write to file instead of stdout.",
    )
    args = parser.parse_args(argv)

    records: list[dict[str, str]] = []
    for line in iter_log_lines(args.logs):
        parsed = parse_allocate_line(line)
        if parsed:
            records.append(parsed)

    if args.per_endpoint:
        columns = _PER_ENDPOINT_COLUMNS
        rows = build_per_endpoint_rows(records)
    else:
        columns = _SUMMARY_COLUMNS
        rows = build_summary_rows(records)

    out_fh: TextIO
    close_out = False
    if args.output:
        out_fh = Path(args.output).open("w", encoding="utf-8", newline="")
        close_out = True
    else:
        out_fh = sys.stdout

    try:
        if args.format == "csv":
            render_csv(columns, rows, out_fh)
        elif args.format == "markdown":
            render_markdown(columns, rows, out_fh)
        else:
            render_table(columns, rows, out_fh)
    finally:
        if close_out:
            out_fh.close()

    if not records:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
