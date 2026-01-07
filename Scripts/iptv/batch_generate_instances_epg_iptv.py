import os
import re
import sys
import time
import random
import subprocess
from pathlib import Path
from typing import Tuple
import urllib.request
import urllib.error

# -----------------------------
# CONFIG
# -----------------------------
GENERATOR_SCRIPT = "build_instance_from_epg_iptv.py"
OUTPUT_ROOT = Path("../../Instances/Instances_IPTV")
CACHE_DIR = Path("./_epg_cache")

OPENING = 0
CLOSING = 1440

DATES = ["2026-01-07"]

SLEEP_BETWEEN_DOWNLOADS_SEC = 2.0

MAX_RETRIES = 6
BASE_BACKOFF_SEC = 2.0


IPTV_SOURCES = [
    ("Albania", "https://iptv-epg.org/files/epg-al.xml"),
    ("Argentina", "https://iptv-epg.org/files/epg-ar.xml"),
    ("Armenia", "https://iptv-epg.org/files/epg-am.xml"),
    ("Australia", "https://iptv-epg.org/files/epg-au.xml"),
    ("Austria", "https://iptv-epg.org/files/epg-at.xml"),
    ("Belarus", "https://iptv-epg.org/files/epg-by.xml"),
    ("Belgium", "https://iptv-epg.org/files/epg-be.xml"),
    ("Bolivia", "https://iptv-epg.org/files/epg-bo.xml"),
    ("Bosnia and Herzegovina", "https://iptv-epg.org/files/epg-ba.xml"),
    ("Brazil", "https://iptv-epg.org/files/epg-br.xml"),
    ("Bulgaria", "https://iptv-epg.org/files/epg-bg.xml"),
    ("Canada", "https://iptv-epg.org/files/epg-ca.xml"),
    ("Chile", "https://iptv-epg.org/files/epg-cl.xml"),
    ("Colombia", "https://iptv-epg.org/files/epg-co.xml"),
    ("Costa Rica", "https://iptv-epg.org/files/epg-cr.xml"),
    ("Croatia", "https://iptv-epg.org/files/epg-hr.xml"),
    ("Czech Republic", "https://iptv-epg.org/files/epg-cz.xml"),
    ("Denmark", "https://iptv-epg.org/files/epg-dk.xml"),
    ("Dominican Republic", "https://iptv-epg.org/files/epg-do.xml"),
    ("Ecuador", "https://iptv-epg.org/files/epg-ec.xml"),
    ("Egypt", "https://iptv-epg.org/files/epg-eg.xml"),
    ("El Salvador", "https://iptv-epg.org/files/epg-sv.xml"),
    ("Finland", "https://iptv-epg.org/files/epg-fi.xml"),
    ("France", "https://iptv-epg.org/files/epg-fr.xml"),
    ("Georgia", "https://iptv-epg.org/files/epg-ge.xml"),
    ("Germany", "https://iptv-epg.org/files/epg-de.xml"),
    ("Ghana", "https://iptv-epg.org/files/epg-gh.xml"),
    ("Greece", "https://iptv-epg.org/files/epg-gr.xml"),
    ("Guatemala", "https://iptv-epg.org/files/epg-gt.xml"),
    ("Honduras", "https://iptv-epg.org/files/epg-hn.xml"),
    ("Hong Kong", "https://iptv-epg.org/files/epg-hk.xml"),
    ("Hungary", "https://iptv-epg.org/files/epg-hu.xml"),
    ("Iceland", "https://iptv-epg.org/files/epg-is.xml"),
    ("India", "https://iptv-epg.org/files/epg-in.xml"),
    ("Indonesia", "https://iptv-epg.org/files/epg-id.xml"),
    ("Israel", "https://iptv-epg.org/files/epg-il.xml"),
    ("Italy", "https://iptv-epg.org/files/epg-it.xml"),
    ("Japan", "https://iptv-epg.org/files/epg-jp.xml"),
    ("Lebanon", "https://iptv-epg.org/files/epg-lb.xml"),
    ("Lithuania", "https://iptv-epg.org/files/epg-lt.xml"),
    ("Luxembourg", "https://iptv-epg.org/files/epg-lu.xml"),
    ("Macedonia", "https://iptv-epg.org/files/epg-mk.xml"),
    ("Malaysia", "https://iptv-epg.org/files/epg-my.xml"),
    ("Malta", "https://iptv-epg.org/files/epg-mt.xml"),
    ("Mexico", "https://iptv-epg.org/files/epg-mx.xml"),
    ("Montenegro", "https://iptv-epg.org/files/epg-me.xml"),
    ("Netherlands", "https://iptv-epg.org/files/epg-nl.xml"),
    ("New Zealand", "https://iptv-epg.org/files/epg-nz.xml"),
    ("Nicaragua", "https://iptv-epg.org/files/epg-ni.xml"),
    ("Nigeria", "https://iptv-epg.org/files/epg-ng.xml"),
    ("Norway", "https://iptv-epg.org/files/epg-no.xml"),
    ("Panama", "https://iptv-epg.org/files/epg-pa.xml"),
    ("Paraguay", "https://iptv-epg.org/files/epg-py.xml"),
    ("Peru", "https://iptv-epg.org/files/epg-pe.xml"),
    ("Poland", "https://iptv-epg.org/files/epg-pl.xml"),
    ("Portugal", "https://iptv-epg.org/files/epg-pt.xml"),
    ("Romania", "https://iptv-epg.org/files/epg-ro.xml"),
    ("Russia", "https://iptv-epg.org/files/epg-ru.xml"),
    ("Saudi Arabia", "https://iptv-epg.org/files/epg-sa.xml"),
    ("Serbia", "https://iptv-epg.org/files/epg-rs.xml"),
    ("Singapore", "https://iptv-epg.org/files/epg-sg.xml"),
    ("Slovenia", "https://iptv-epg.org/files/epg-si.xml"),
    ("South Africa", "https://iptv-epg.org/files/epg-za.xml"),
    ("South Korea", "https://iptv-epg.org/files/epg-kr.xml"),
    ("Spain", "https://iptv-epg.org/files/epg-es.xml"),
    ("Sweden", "https://iptv-epg.org/files/epg-se.xml"),
    ("Switzerland", "https://iptv-epg.org/files/epg-ch.xml"),
    ("Taiwan", "https://iptv-epg.org/files/epg-tw.xml"),
    ("Thailand", "https://iptv-epg.org/files/epg-th.xml"),
    ("Turkey", "https://iptv-epg.org/files/epg-tr.xml"),
    ("Uganda", "https://iptv-epg.org/files/epg-ug.xml"),
    ("Ukraine", "https://iptv-epg.org/files/epg-ua.xml"),
    ("United Arab Emirates", "https://iptv-epg.org/files/epg-ae.xml"),
    ("United Kingdom", "https://iptv-epg.org/files/epg-gb.xml"),
    ("United States", "https://iptv-epg.org/files/epg-us.xml"),
    ("Uruguay", "https://iptv-epg.org/files/epg-uy.xml"),
    ("Venezuela", "https://iptv-epg.org/files/epg-ve.xml"),
    ("Vietnam", "https://iptv-epg.org/files/epg-vn.xml"),
    ("Zimbabwe", "https://iptv-epg.org/files/epg-zw.xml"),
]

# -----------------------------
# Helpers
# -----------------------------
def safe_folder_name(name: str) -> str:
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s

def code_from_url(url: str) -> str:
    m = re.search(r"epg-([a-z]{2})\.xml$", url.strip().lower())
    return m.group(1) if m else "xx"

def download_with_retry(url: str, out_path: Path) -> None:
    """
    Download to out_path with retry/backoff on 429/5xx.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (InstanceGenerator/1.0; +https://example.local)",
        "Accept": "*/*",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            return

        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                wait = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
                wait = min(wait, 60.0)  # cap
                wait += random.uniform(0.0, 1.5)  # jitter
                print(f"[DOWNLOAD] {url} -> HTTP {e.code}. Retry {attempt}/{MAX_RETRIES} in {wait:.1f}s")
                time.sleep(wait)
                continue
            raise

        except Exception as e:
            wait = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
            wait = min(wait, 60.0)
            wait += random.uniform(0.0, 1.5)
            print(f"[DOWNLOAD] {url} -> {type(e).__name__}. Retry {attempt}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download after {MAX_RETRIES} retries: {url}")

def ensure_cached(country: str, url: str) -> Path:

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    code = code_from_url(url)
    cache_path = CACHE_DIR / f"epg-{code}.xml"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    print(f"[CACHE] Downloading {country} ({code})...")
    download_with_retry(url, cache_path)
    time.sleep(SLEEP_BETWEEN_DOWNLOADS_SEC)
    return cache_path

def run_one(country: str, url_or_path: str, day: str) -> Tuple[bool, str]:
    country_dir = OUTPUT_ROOT / safe_folder_name(country)
    country_dir.mkdir(parents=True, exist_ok=True)

    code = code_from_url(url_or_path)
    if code == "xx":
        m = re.search(r"epg-([a-z]{2})\.xml$", str(url_or_path).lower())
        code = m.group(1) if m else "xx"

    out_path = country_dir / f"iptv-epg_{code}_{day}.json"

    cmd = [
        sys.executable,
        GENERATOR_SCRIPT,
        "--epg-url", str(url_or_path),
        "--date", day,
        "--opening", str(OPENING),
        "--closing", str(CLOSING),
        "--out", str(out_path),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = (p.returncode == 0)
    if ok:
        return True, p.stdout.strip() or f"Wrote {out_path}"
    else:
        msg = (p.stdout + "\n" + p.stderr).strip()
        return False, msg or "Unknown error"

def main():
    if not Path(GENERATOR_SCRIPT).exists():
        print(f"[ERROR] Cannot find generator script: {GENERATOR_SCRIPT}")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total_jobs = len(IPTV_SOURCES) * len(DATES)
    job_idx = 0
    failures = []

    for country, url in IPTV_SOURCES:
        try:
            local_xml = ensure_cached(country, url)
        except Exception as e:
            failures.append((country, "ALL_DATES", url, f"Cache download failed: {e}"))
            print(f"[FAIL] {country} cache download failed: {e}")
            continue

        for day in DATES:
            job_idx += 1
            print(f"\n[{job_idx}/{total_jobs}] {country} | {day}")
            ok, info = run_one(country, str(local_xml), day)

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
        fail_log = OUTPUT_ROOT / "failures.log"
        with open(fail_log, "w", encoding="utf-8") as f:
            for country, day, url, info in failures:
                f.write(f"{country}\t{day}\t{url}\n{info}\n{'-'*80}\n")
        print(f"Failures written to: {fail_log}")

if __name__ == "__main__":
    main()
