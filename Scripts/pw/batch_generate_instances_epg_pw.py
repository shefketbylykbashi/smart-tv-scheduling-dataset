from __future__ import annotations

import argparse
import random
import re
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import urllib.error

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_GENERATOR_SCRIPT = "build_instance_from_epg_pw.py"
DEFAULT_OUTPUT_ROOT = Path("../../Instances/Instances_PW")
DEFAULT_CACHE_DIR = Path("./_epg_cache_pw")

DEFAULT_OPENING = 0
DEFAULT_CLOSING = 1440

DEFAULT_SLEEP_BETWEEN_DOWNLOADS_SEC = 2.0

DEFAULT_MAX_RETRIES = 6
DEFAULT_BASE_BACKOFF_SEC = 2.0


# -----------------------------
# Sources
# -----------------------------
IPTV_EPG_ORG_SOURCES: List[Tuple[str, str]] = [

]

EPG_PW_SOURCES: List[Tuple[str, str]] = [
    ("Australia_AU", "https://epg.pw/xmltv/epg_AU.xml.gz"),
    ("Brazil_BR", "https://epg.pw/xmltv/epg_BR.xml.gz"),
    ("Canada_CA", "https://epg.pw/xmltv/epg_CA.xml.gz"),
    ("China_CN", "https://epg.pw/xmltv/epg_CN.xml.gz"),
    ("Germany_DE", "https://epg.pw/xmltv/epg_DE.xml.gz"),
    ("France_FR", "https://epg.pw/xmltv/epg_FR.xml.gz"),
    ("United_Kingdom_GB", "https://epg.pw/xmltv/epg_GB.xml.gz"),
    ("Hong_Kong_HK", "https://epg.pw/xmltv/epg_HK.xml.gz"),
    ("Indonesia_ID", "https://epg.pw/xmltv/epg_ID.xml.gz"),
    ("India_IN", "https://epg.pw/xmltv/epg_IN.xml.gz"),
    ("Japan_JP", "https://epg.pw/xmltv/epg_JP.xml.gz"),
    ("Malaysia_MY", "https://epg.pw/xmltv/epg_MY.xml.gz"),
    ("New_Zealand_NZ", "https://epg.pw/xmltv/epg_NZ.xml.gz"),
    ("Philippines_PH", "https://epg.pw/xmltv/epg_PH.xml.gz"),
    ("Russian_Federation_RU", "https://epg.pw/xmltv/epg_RU.xml.gz"),
    ("Singapore_SG", "https://epg.pw/xmltv/epg_SG.xml.gz"),
    ("Taiwan_TW", "https://epg.pw/xmltv/epg_TW.xml.gz"),
    ("US_US", "https://epg.pw/xmltv/epg_US.xml.gz"),
    ("Viet_Nam_VN", "https://epg.pw/xmltv/epg_VN.xml.gz"),
    ("South_Africa_ZA", "https://epg.pw/xmltv/epg_ZA.xml.gz"),
]


# -----------------------------
# Helpers
# -----------------------------
SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]+")

def safe_folder_name(name: str) -> str:
    s = name.strip().replace(" ", "_")
    s = re.sub(r"\s+", "_", s)
    s = SAFE_RE.sub("", s)
    return s or "Unknown"

def extract_code_any(url: str) -> str:
    u = url.strip()
    m1 = re.search(r"epg-([a-z]{2})\.xml$", u.lower())
    if m1:
        return m1.group(1)

    m2 = re.search(r"epg_([A-Za-z]{2})\.xml(?:\.gz)?$", u)
    if m2:
        return m2.group(1)

    return "xx"

def download_with_retry(url: str, out_path: Path, max_retries: int, base_backoff_sec: float) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (InstanceGenerator/1.0; batch-epg)",
        "Accept": "*/*",
    }

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            return

        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                wait = base_backoff_sec * (2 ** (attempt - 1))
                wait = min(wait, 90.0)
                wait += random.uniform(0.0, 1.5)
                print(f"[DOWNLOAD] {url} -> HTTP {e.code}. Retry {attempt}/{max_retries} in {wait:.1f}s")
                time.sleep(wait)
                continue
            raise

        except Exception as e:
            wait = base_backoff_sec * (2 ** (attempt - 1))
            wait = min(wait, 90.0)
            wait += random.uniform(0.0, 1.5)
            print(f"[DOWNLOAD] {url} -> {type(e).__name__}: {e}. Retry {attempt}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download after {max_retries} retries: {url}")

def ensure_cached(country: str, url: str, cache_dir: Path,
                  sleep_between_downloads: float,
                  max_retries: int,
                  base_backoff_sec: float,
                  force_refresh: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    code = extract_code_any(url)

    ext = ".gz" if url.lower().endswith(".gz") else ".xml"
    cache_path = cache_dir / f"epg_{code}{ext}"

    if not force_refresh and cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    print(f"[CACHE] Downloading {country} ({code}) -> {cache_path.name}")
    download_with_retry(url, cache_path, max_retries=max_retries, base_backoff_sec=base_backoff_sec)
    time.sleep(max(0.0, sleep_between_downloads))
    return cache_path

def output_path_for(output_root: Path, country: str, code: str, day: str) -> Path:
    country_dir = output_root / safe_folder_name(country)
    country_dir.mkdir(parents=True, exist_ok=True)
    return country_dir / f"epg_{code}_{day}.json"

def run_generator_once(generator_script: str,
                       epg_path: Path,
                       day: str,
                       opening: int,
                       closing: int,
                       out_path: Path) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        generator_script,
        "--epg-url", str(epg_path),
        "--date", day,
        "--opening", str(opening),
        "--closing", str(closing),
        "--out", str(out_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = (p.returncode == 0)
    msg = (p.stdout + "\n" + p.stderr).strip()
    if ok:
        return True, (p.stdout.strip() or f"Wrote {out_path}")
    return False, (msg or "Unknown error")


def parse_dates_arg(dates_arg: str) -> List[str]:
    dates = [d.strip() for d in dates_arg.split(",") if d.strip()]
    if not dates:
        raise ValueError("No dates provided.")
    for d in dates:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            raise ValueError(f"Bad date format: {d} (expected YYYY-MM-DD)")
    return dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["iptv", "epgpw"], default="epgpw", help="Which provider list to use (default epgpw)")
    ap.add_argument("--generator", default=DEFAULT_GENERATOR_SCRIPT, help="Your build_instance_from_epg*.py script")
    ap.add_argument("--out-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--dates", default="2026-01-04", help="Comma-separated dates, e.g. 2026-01-04,2026-01-05")
    ap.add_argument("--opening", type=int, default=DEFAULT_OPENING)
    ap.add_argument("--closing", type=int, default=DEFAULT_CLOSING)
    ap.add_argument("--sleep-between-downloads", type=float, default=DEFAULT_SLEEP_BETWEEN_DOWNLOADS_SEC)
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument("--base-backoff", type=float, default=DEFAULT_BASE_BACKOFF_SEC)
    ap.add_argument("--resume", action="store_true", help="Skip if output JSON already exists")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output even if it exists")
    ap.add_argument("--refresh-cache", action="store_true", help="Force re-download EPG files")
    ap.add_argument("--limit-countries", type=int, default=0, help="For testing: only process first N countries (0=all)")
    args = ap.parse_args()

    generator_script = args.generator
    if not Path(generator_script).exists():
        print(f"[ERROR] Cannot find generator script: {generator_script}")
        sys.exit(1)

    output_root = Path(args.out_root)
    cache_dir = Path(args.cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = parse_dates_arg(args.dates)

    sources = EPG_PW_SOURCES if args.source == "epgpw" else IPTV_EPG_ORG_SOURCES
    if args.limit_countries and args.limit_countries > 0:
        sources = sources[: args.limit_countries]

    total_jobs = len(sources) * len(dates)
    job_idx = 0
    failures: List[Tuple[str, str, str, str]] = []

    for country, url in sources:
        code = extract_code_any(url)

        try:
            local_epg = ensure_cached(
                country=country,
                url=url,
                cache_dir=cache_dir,
                sleep_between_downloads=args.sleep_between_downloads,
                max_retries=args.max_retries,
                base_backoff_sec=args.base_backoff,
                force_refresh=args.refresh_cache,
            )
        except Exception as e:
            failures.append((country, "ALL_DATES", url, f"Cache download failed: {e}"))
            print(f"[FAIL] {country} cache download failed: {e}")
            continue

        for day in dates:
            job_idx += 1
            out_path = output_path_for(output_root, country, code, day)

            if out_path.exists() and not args.overwrite:
                if args.resume:
                    print(f"[{job_idx}/{total_jobs}] [SKIP] {country} | {day} -> {out_path}")
                    continue
                else:
                    print(f"[{job_idx}/{total_jobs}] [WARN] Output exists (use --resume to skip or --overwrite): {out_path}")

            print(f"\n[{job_idx}/{total_jobs}] {country} | {day}")
            ok, info = run_generator_once(
                generator_script=generator_script,
                epg_path=local_epg,
                day=day,
                opening=args.opening,
                closing=args.closing,
                out_path=out_path,
            )

            if ok:
                print(f"[OK] {info}")
            else:
                print(f"[FAIL] {country} | {day}")
                print(info)
                failures.append((country, day, url, info))

    print("\n====================")
    print("Batch finished.")
    print(f"Total planned: {total_jobs} | Failed: {len(failures)} | Success: {total_jobs - len(failures)}")

    if failures:
        fail_log = output_root / "failures.log"
        with open(fail_log, "w", encoding="utf-8") as f:
            for country, day, url, info in failures:
                f.write(f"{country}\t{day}\t{url}\n{info}\n{'-'*80}\n")
        print(f"Failures written to: {fail_log}")


if __name__ == "__main__":
    main()
