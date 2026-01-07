from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_ROOTS = ["Instances_IPTV", "Instances_PW", "Instances_Youtube"]


@dataclass
class InstanceStats:
    path: Path
    channels: int
    programs: int


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def is_instance_json(obj: Dict[str, Any]) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("channels"), list)


def extract_stats(json_obj: Dict[str, Any], path: Path) -> Optional[InstanceStats]:
    if not is_instance_json(json_obj):
        return None

    channels_list = json_obj.get("channels") or []
    if not isinstance(channels_list, list):
        return None

    channels_count = safe_int(json_obj.get("channels_count"), default=len(channels_list))
    if channels_count <= 0:
        channels_count = len(channels_list)

    total_programs = 0
    for ch in channels_list:
        if not isinstance(ch, dict):
            continue
        progs = ch.get("programs")
        if isinstance(progs, list):
            total_programs += len(progs)

    return InstanceStats(path=path, channels=channels_count, programs=total_programs)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return json.loads(text)
    except Exception:
        return None


def collect_instances(roots: List[str]) -> Tuple[List[InstanceStats], Dict[str, int]]:
    all_stats: List[InstanceStats] = []
    counts_by_root: Dict[str, int] = {}

    for root in roots:
        r = Path(root)
        if not r.exists():
            counts_by_root[root] = 0
            continue

        json_files = list(r.rglob("*.json"))
        counts_by_root[root] = len(json_files)

        for fp in json_files:
            obj = read_json(fp)
            if obj is None:
                continue
            st = extract_stats(obj, fp)
            if st is not None:
                all_stats.append(st)

    return all_stats, counts_by_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="How many top instances to print (default 10)")
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS, help="Root folders to scan")
    args = ap.parse_args()

    stats, counts_by_root = collect_instances(args.roots)

    total_files = sum(counts_by_root.values())
    total_instances_parsed = len(stats)

    print("=== Instance counts (JSON files) ===")
    for root in args.roots:
        print(f"{root}: {counts_by_root.get(root, 0)}")
    print(f"TOTAL JSON files: {total_files}")
    print(f"TOTAL parsed as instances: {total_instances_parsed}")
    print()

    stats_sorted = sorted(
        stats,
        key=lambda s: (-s.channels, -s.programs, str(s.path).lower()),
    )

    top_n = stats_sorted[: max(0, args.top)]

    print(f"=== TOP {len(top_n)} instances by channels ===")
    if not top_n:
        print("(none found)")
        return

    for i, st in enumerate(top_n, start=1):
        print(f"{i:02d}. channels={st.channels:<5} programs={st.programs:<6} file={st.path}")

if __name__ == "__main__":
    main()
