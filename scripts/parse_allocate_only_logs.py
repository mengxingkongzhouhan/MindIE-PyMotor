#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
"""
Filter ALLOCATE_ONLY scheduling logs into a table.

Parses lines like:
  ALLOCATE_ONLY req_id=... role=prefill ins=1 ep=10
  active_requests=3 active_tokens=128.45 active_kv_cache=128.45
  prefill_endpoints=1:10=req:3,tokens:128.45,kv:128.45;1:11=req:1,tokens:40.00,kv:200.00
  decode_endpoints=2:20=req:5,tokens:200.00,kv:0.00 score=... fast_path=...

Rows are sorted by processing order (log appearance order across input files).

Usage:
  python3 scripts/parse_allocate_only_logs.py coordinator.log --per-endpoint
  python3 scripts/parse_allocate_only_logs.py 'log/vllm-0-coordinator-*' --per-endpoint
  python3 scripts/parse_allocate_only_logs.py coordinator.log --per-endpoint --pool all
  python3 scripts/parse_allocate_only_logs.py coordinator.log --format csv -o out.csv
  cat coordinator.log | python3 scripts/parse_allocate_only_logs.py -
"""

from __future__ import annotations

import argparse
import csv
import glob
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
    r"(?:active_kv_cache=(?P<active_kv_cache>\S+)\s+)?"
    r"prefill_endpoints=(?P<prefill_endpoints>\S+)\s+"
    r"decode_endpoints=(?P<decode_endpoints>\S+)\s+"
    r"score=(?P<score>\S+)\s+"
    r"fast_path=(?P<fast_path>\S+)"
)

# New: req:N,tokens:T,kv:K  |  Old (compat): req:N,tokens:T
_ENDPOINT_STAT_RE = re.compile(
    r"(?P<ins_ep>[^=;]+)=req:(?P<req>-?\d+),tokens:(?P<tokens>-?\d+(?:\.\d+)?)"
    r"(?:,kv:(?P<kv>-?\d+(?:\.\d+)?))?"
)

_SUMMARY_COLUMNS = [
    "seq",
    "req_id",
    "role",
    "ins",
    "ep",
    "active_requests",
    "active_tokens",
    "active_kv_cache",
    "prefill_endpoints",
    "decode_endpoints",
    "score",
    "fast_path",
]

_PER_ENDPOINT_COLUMNS = [
    "seq",
    "req_id",
    "role",
    "selected_ins",
    "selected_ep",
    "pool",
    "ins",
    "ep",
    "active_requests",
    "active_tokens",
    "active_kv_cache",
    "lb_score",
    "selected_active_requests",
    "selected_active_tokens",
    "selected_active_kv_cache",
    "score",
    "fast_path",
]

_POOL_ORDER = {"prefill": 0, "decode": 1}


@dataclass(frozen=True)
class EndpointStat:
    ins: str
    ep: str
    active_requests: int
    active_tokens: float
    active_kv_cache: float


def _prefill_lb_score(tokens: float, kv: float) -> float:
    """Same formula as Workload.calculate_workload_score for prefill."""
    return tokens + 0.3 * kv


def parse_endpoint_stats(blob: str) -> list[EndpointStat]:
    """Parse endpoint snapshot blob; kv is optional for older logs."""
    if not blob or blob == "none":
        return []
    stats: list[EndpointStat] = []
    for match in _ENDPOINT_STAT_RE.finditer(blob):
        ins_ep = match.group("ins_ep")
        if ":" not in ins_ep:
            continue
        ins, ep = ins_ep.split(":", 1)
        kv_raw = match.group("kv")
        stats.append(
            EndpointStat(
                ins=ins,
                ep=ep,
                active_requests=int(match.group("req")),
                active_tokens=float(match.group("tokens")),
                active_kv_cache=float(kv_raw) if kv_raw is not None else 0.0,
            )
        )
    return stats


def parse_allocate_line(line: str) -> dict[str, str] | None:
    match = _ALLOCATE_RE.search(line)
    if not match:
        return None
    data = match.groupdict()
    if data.get("active_kv_cache") is None:
        data["active_kv_cache"] = ""
    return data


def expand_log_paths(paths: list[str]) -> list[str]:
    """Expand shell-style globs; keep stdin marker '-' as-is."""
    if not paths or paths == ["-"]:
        return ["-"]
    expanded: list[str] = []
    for path in paths:
        if path == "-":
            expanded.append(path)
            continue
        matches = sorted(glob.glob(path))
        if matches:
            expanded.extend(matches)
        else:
            # Allow nonexistent path to surface later as open error / empty.
            expanded.append(path)
    return expanded


def iter_log_lines(paths: list[str]) -> Iterable[tuple[str, int, str]]:
    """Yield (source, line_no, line) in file order."""
    resolved = expand_log_paths(paths)
    if resolved == ["-"]:
        for line_no, line in enumerate(sys.stdin, start=1):
            yield ("-", line_no, line)
        return
    for path in resolved:
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                yield (path, line_no, line)


def build_summary_rows(records: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for rec in records:
        row = {col: rec.get(col, "") for col in _SUMMARY_COLUMNS}
        rows.append(row)
    return rows


def build_per_endpoint_rows(
    records: list[dict[str, str]],
    pools: tuple[str, ...] = ("prefill", "decode"),
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pool_blobs = {
        "prefill": "prefill_endpoints",
        "decode": "decode_endpoints",
    }
    for rec in records:
        for pool in pools:
            blob_key = pool_blobs.get(pool)
            if not blob_key:
                continue
            blob = rec.get(blob_key, "")
            stats = parse_endpoint_stats(blob)
            if not stats:
                rows.append(
                    {
                        "seq": rec.get("seq", ""),
                        "req_id": rec.get("req_id", ""),
                        "role": rec.get("role", ""),
                        "selected_ins": "",
                        "selected_ep": rec.get("ep", ""),
                        "pool": pool,
                        "ins": "",
                        "ep": "",
                        "active_requests": "",
                        "active_tokens": "",
                        "active_kv_cache": "",
                        "lb_score": "",
                        "selected_active_requests": rec.get("active_requests", ""),
                        "selected_active_tokens": rec.get("active_tokens", ""),
                        "selected_active_kv_cache": rec.get("active_kv_cache", ""),
                        "score": rec.get("score", ""),
                        "fast_path": rec.get("fast_path", ""),
                    }
                )
                continue
            # Stable order within one allocate snapshot: ins/ep numeric when possible.
            def _sort_key(stat: EndpointStat) -> tuple:
                try:
                    return (int(stat.ins), int(stat.ep))
                except ValueError:
                    return (stat.ins, stat.ep)

            selected_ins = rec.get("ins", "")
            for stat in sorted(stats, key=_sort_key):
                rows.append(
                    {
                        "seq": rec.get("seq", ""),
                        "req_id": rec.get("req_id", ""),
                        "role": rec.get("role", ""),
                        # Only fill selected_ins on rows belonging to the chosen instance.
                        "selected_ins": (
                            selected_ins if str(stat.ins) == str(selected_ins) else ""
                        ),
                        "selected_ep": rec.get("ep", ""),
                        "pool": pool,
                        "ins": stat.ins,
                        "ep": stat.ep,
                        "active_requests": str(stat.active_requests),
                        "active_tokens": f"{stat.active_tokens:.2f}",
                        "active_kv_cache": f"{stat.active_kv_cache:.2f}",
                        "lb_score": f"{_prefill_lb_score(stat.active_tokens, stat.active_kv_cache):.2f}",
                        "selected_active_requests": rec.get("active_requests", ""),
                        "selected_active_tokens": rec.get("active_tokens", ""),
                        "selected_active_kv_cache": rec.get("active_kv_cache", ""),
                        "score": rec.get("score", ""),
                        "fast_path": rec.get("fast_path", ""),
                    }
                )
    # Keep allocate processing order (seq), then pool, then endpoint.
    rows.sort(
        key=lambda r: (
            int(r.get("seq") or 0),
            _POOL_ORDER.get(r.get("pool", ""), 99),
            _safe_int(r.get("ins", "")),
            _safe_int(r.get("ep", "")),
        )
    )
    return rows


def _safe_int(value: str) -> tuple[int, str]:
    try:
        return (0, str(int(value)))
    except (TypeError, ValueError):
        return (1, value or "")


def _request_key(row: dict[str, str]) -> tuple[str, str]:
    """Group key used to insert blank lines between different requests."""
    return (row.get("seq", ""), row.get("req_id", ""))


def render_markdown(
    columns: list[str],
    rows: list[dict[str, str]],
    out: TextIO,
    separate_requests: bool = False,
) -> None:
    if not rows:
        out.write("No ALLOCATE_ONLY records found.\n")
        return
    out.write("| " + " | ".join(columns) + " |\n")
    out.write("| " + " | ".join("---" for _ in columns) + " |\n")
    prev_key: tuple[str, str] | None = None
    for row in rows:
        key = _request_key(row)
        if separate_requests and prev_key is not None and key != prev_key:
            out.write("\n")
        out.write("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |\n")
        prev_key = key


def render_csv(columns: list[str], rows: list[dict[str, str]], out: TextIO) -> None:
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def render_table(
    columns: list[str],
    rows: list[dict[str, str]],
    out: TextIO,
    separate_requests: bool = False,
) -> None:
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
    prev_key: tuple[str, str] | None = None
    for row in rows:
        key = _request_key(row)
        if separate_requests and prev_key is not None and key != prev_key:
            out.write("\n")
        out.write(
            "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
            + "\n"
        )
        prev_key = key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Filter ALLOCATE_ONLY logs into a table. "
            "Every matched allocate is emitted in processing order."
        )
    )
    parser.add_argument(
        "logs",
        nargs="*",
        default=["-"],
        help="Log file path(s) or globs. Use - or omit for stdin.",
    )
    parser.add_argument(
        "--per-endpoint",
        action="store_true",
        help="Expand endpoint snapshots into one row per endpoint.",
    )
    parser.add_argument(
        "--pool",
        choices=("prefill", "decode", "all"),
        default="prefill",
        help=(
            "Which endpoint pool to expand with --per-endpoint "
            "(default: prefill). Use 'all' to include decode."
        ),
    )
    parser.add_argument(
        "--role",
        choices=("prefill", "decode", "all"),
        default="prefill",
        help="Only keep ALLOCATE_ONLY lines for this role (default: prefill).",
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
    seq = 0
    for _source, _line_no, line in iter_log_lines(args.logs):
        parsed = parse_allocate_line(line)
        if not parsed:
            continue
        if args.role != "all" and parsed.get("role") != args.role:
            continue
        seq += 1
        parsed["seq"] = str(seq)
        records.append(parsed)

    # Processing order = log appearance order (seq already assigned).
    records.sort(key=lambda r: (int(r["seq"]), r.get("req_id", ""), r.get("role", "")))

    if args.per_endpoint:
        columns = _PER_ENDPOINT_COLUMNS
        if args.pool == "all":
            pools: tuple[str, ...] = ("prefill", "decode")
        else:
            pools = (args.pool,)
        rows = build_per_endpoint_rows(records, pools=pools)
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
            # CSV keeps contiguous rows; blank separators are for human table/markdown.
            render_csv(columns, rows, out_fh)
        elif args.format == "markdown":
            render_markdown(columns, rows, out_fh, separate_requests=True)
        else:
            render_table(columns, rows, out_fh, separate_requests=True)
    finally:
        if close_out:
            out_fh.close()

    if not records:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
