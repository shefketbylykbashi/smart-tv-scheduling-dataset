from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple


YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

REGION_LANG_POOL: List[Tuple[str, str]] = [
    ("US", "en"), ("GB", "en"), ("CA", "en"), ("AU", "en"),
    ("DE", "de"), ("FR", "fr"), ("ES", "es"), ("IT", "it"),
    ("PT", "pt"), ("BR", "pt"),
    ("TR", "tr"), ("PL", "pl"), ("NL", "nl"),
    ("SE", "sv"), ("NO", "no"), ("FI", "fi"), ("DK", "da"),
    ("GR", "el"), ("RO", "ro"), ("BG", "bg"),
    ("RS", "sr"), ("UA", "uk"), ("RU", "ru"),
    ("IL", "he"),
    ("AE", "ar"), ("SA", "ar"), ("EG", "ar"),
    ("IN", "hi"),
    ("JP", "ja"), ("KR", "ko"),
    ("ID", "id"), ("TH", "th"), ("VN", "vi"),
    ("HK", "zh"), ("TW", "zh"),
]

QUERY_SETS: List[List[str]] = [
    ["live", "livestream"],
    ["news live", "breaking news live"],
    ["sports live", "football live"],
    ["music live", "dj set live"],
    ["gaming live", "esports live"],
    ["tech live", "ai live"],
    ["crypto live", "trading live"],
    ["education live", "lecture live"],
    ["travel live", "city walk live"],
    ["podcast live", "talk show live"],
]

def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_iso_z(s: str) -> datetime:
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def http_get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (BatchSeedsCache/1.0)", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    return json.loads(data)

def yt_search_upcoming(api_key: str, q: str, max_pages: int, page_size: int,
                       region_code: Optional[str], relevance_language: Optional[str]) -> List[str]:
    video_ids: List[str] = []
    page_token = None

    for _ in range(max_pages):
        params = {
            "part": "snippet",
            "type": "video",
            "eventType": "upcoming",
            "q": q,
            "maxResults": str(page_size),
            "key": api_key,
        }
        if region_code:
            params["regionCode"] = region_code
        if relevance_language:
            params["relevanceLanguage"] = relevance_language
        if page_token:
            params["pageToken"] = page_token

        url = f"{YOUTUBE_API}/search?{urllib.parse.urlencode(params)}"
        js = http_get_json(url)
        for it in js.get("items", []):
            vid = (it.get("id") or {}).get("videoId")
            if vid:
                video_ids.append(vid)

        page_token = js.get("nextPageToken")
        if not page_token:
            break

        time.sleep(0.15)

    seen = set()
    out = []
    for v in video_ids:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def yt_videos_details(api_key: str, video_ids: List[str]) -> List[dict]:
    out: List[dict] = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        params = {
            "part": "snippet,liveStreamingDetails,topicDetails,contentDetails",
            "id": ",".join(chunk),
            "key": api_key,
        }
        url = f"{YOUTUBE_API}/videos?{urllib.parse.urlencode(params)}"
        js = http_get_json(url)
        out.extend(js.get("items", []))
        time.sleep(0.15)
    return out

def cache_filename(region: str, lang: str, queries: List[str]) -> str:
    key = f"{region}|{lang}|{'|'.join(queries)}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"seeds_{region}_{lang}_{h}.json"

def get_or_build_seeds(
    api_key: str,
    cache_dir: Path,
    region: str,
    lang: str,
    queries: List[str],
    seed_limit: int,
    max_pages: int,
    page_size: int,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_filename(region, lang, queries)
    path = cache_dir / fname

    if path.exists() and path.stat().st_size > 50_000:
        return path

    ids: List[str] = []
    for q in queries:
        ids.extend(yt_search_upcoming(api_key, q, max_pages, page_size, region, lang))

    seen = set()
    uniq = []
    for v in ids:
        if v not in seen:
            seen.add(v)
            uniq.append(v)

    uniq = uniq[:seed_limit]
    seeds = yt_videos_details(api_key, uniq)
    if not seeds:
        raise RuntimeError(f"No seeds fetched for {region}/{lang} queries={queries}")

    path.write_text(json.dumps(seeds, ensure_ascii=False), encoding="utf-8")
    return path


@dataclass
class Spec:
    tier: str
    idx: int
    start_utc: datetime
    days: int
    channels: int
    region: str
    lang: str
    switch_penalty: int
    termination_penalty: int
    max_consecutive_genre: int
    min_duration: int
    seed_limit: int
    rng_seed: int
    queries: List[str]


def randint_weighted(rng: random.Random, lo: int, hi: int, bias: str = "mid") -> int:
    if lo >= hi:
        return lo
    if bias == "low":
        u = rng.random() ** 2
    elif bias == "high":
        u = 1.0 - (rng.random() ** 2)
    else:
        u = (rng.random() + rng.random()) / 2.0
    return lo + int(u * (hi - lo))


def build_spec(rng: random.Random, tier: str, idx: int, start_range: Tuple[datetime, datetime]) -> Spec:
    lo, hi = start_range
    minutes = int((hi - lo).total_seconds() // 60)
    start_dt = (lo + timedelta(minutes=rng.randint(0, max(0, minutes)))).replace(minute=0, second=0, microsecond=0)

    days = randint_weighted(rng, 1, 20, bias="mid")
    if tier == "Premium":
        channels = randint_weighted(rng, 1500, 2000, bias="high")
        max_consec = rng.choice([1, 2, 2, 3])
        switch_pen = randint_weighted(rng, 12, 45, bias="high")
        term_pen = randint_weighted(rng, 18, 90, bias="high")
        min_dur = rng.choice([8, 10, 12, 15])
        seed_limit = rng.choice([1200, 1500, 1800])
    else:
        channels = randint_weighted(rng, 300, 2000, bias="mid")
        max_consec = rng.choice([1, 2, 2, 3, 4])
        switch_pen = randint_weighted(rng, 5, 40, bias="mid")
        term_pen = randint_weighted(rng, 10, 80, bias="mid")
        min_dur = rng.choice([5, 8, 10, 12])
        seed_limit = rng.choice([500, 600, 800, 1000, 1200])

    region, lang = rng.choice(REGION_LANG_POOL)
    queries = rng.choice(QUERY_SETS)
    rng_seed = rng.randint(1, 2_000_000_000)

    return Spec(
        tier=tier,
        idx=idx,
        start_utc=start_dt,
        days=days,
        channels=channels,
        region=region,
        lang=lang,
        switch_penalty=switch_pen,
        termination_penalty=term_pen,
        max_consecutive_genre=max_consec,
        min_duration=min_dur,
        seed_limit=seed_limit,
        rng_seed=rng_seed,
        queries=queries,
    )


def run_generator(
    generator: Path,
    api_key: str,
    seeds_file: Path,
    out_path: Path,
    spec: Spec,
    python_exe: str,
) -> Tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    opening = 0
    closing = spec.days * 1440

    cmd = [
        python_exe,
        str(generator),
        "--api-key", api_key,
        "--seeds-json", str(seeds_file),
        "--start-utc", iso_z(spec.start_utc),
        "--opening", str(opening),
        "--closing", str(closing),
        "--channels", str(spec.channels),
        "--min-duration", str(spec.min_duration),
        "--max-consecutive-genre", str(spec.max_consecutive_genre),
        "--switch-penalty", str(spec.switch_penalty),
        "--termination-penalty", str(spec.termination_penalty),
        "--region", spec.region,
        "--lang", spec.lang,
        "--seed-limit", str(spec.seed_limit),
        "--rng-seed", str(spec.rng_seed),
        "--out", str(out_path),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode == 0:
        return True, (p.stdout.strip() or f"Wrote {out_path}")
    return False, (p.stdout + "\n" + p.stderr).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("YOUTUBE_API_KEY", ""), required=False)
    ap.add_argument("--generator", default="youtube_instance_generator2.py")
    ap.add_argument("--out-root", default="Instances_Youtube")
    ap.add_argument("--cache-dir", default="SeedsCache")
    ap.add_argument("--premium-count", type=int, default=10)
    ap.add_argument("--gold-count", type=int, default=90)
    ap.add_argument("--start-range-from", default="2026-01-01T00:00:00Z")
    ap.add_argument("--start-range-to", default="2026-12-31T00:00:00Z")
    ap.add_argument("--max-pages", type=int, default=2)
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--sleep-between", type=float, default=0.25)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument("--resume", action="store_true", help="Skip instances that already exist on disk")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite even if output exists (overrides --resume)")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Use --api-key or set env var YOUTUBE_API_KEY")

    generator = Path(args.generator)
    if not generator.exists():
        raise SystemExit(f"Generator not found: {generator}")

    out_root = Path(args.out_root)
    premium_dir = out_root / "Premium"
    gold_dir = out_root / "Gold"
    cache_dir = Path(args.cache_dir)

    start_from = parse_iso_z(args.start_range_from)
    start_to = parse_iso_z(args.start_range_to)

    rng = random.Random(args.rng_seed)

    specs: List[Spec] = []
    for i in range(1, args.premium_count + 1):
        specs.append(build_spec(rng, "Premium", i, (start_from, start_to)))
    for i in range(1, args.gold_count + 1):
        specs.append(build_spec(rng, "Gold", i, (start_from, start_to)))

    rng.shuffle(specs)
    total = len(specs)

    failures = []
    python_exe = sys.executable

    for k, spec in enumerate(specs, start=1):
        tier_dir = premium_dir if spec.tier == "Premium" else gold_dir
        start_tag = spec.start_utc.strftime("%Y%m%d")
        out_path = tier_dir / f"YT_{spec.tier}_{spec.idx:03d}_ch{spec.channels}_d{spec.days}_{spec.region}_{spec.lang}_{start_tag}.json"
        print(out_path)
        print(args.resume)
        if out_path.exists() and not args.overwrite:
            if args.resume:
                print(f"[SKIP] Already exists: {out_path}")
                continue
            else:
                print(f"[WARN] Output already exists (use --resume to skip, or --overwrite to replace): {out_path}")
                
        print(f"\n[{k}/{total}] {spec.tier} #{spec.idx} -> {out_path}")
        print(
            f"  start={iso_z(spec.start_utc)} days={spec.days} closing={spec.days*1440} "
            f"channels={spec.channels} region={spec.region} lang={spec.lang} "
            f"switch={spec.switch_penalty} term={spec.termination_penalty} maxConsec={spec.max_consecutive_genre} seedLimit={spec.seed_limit}"
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                seeds_file = get_or_build_seeds(
                    api_key=args.api_key,
                    cache_dir=cache_dir,
                    region=spec.region,
                    lang=spec.lang,
                    queries=spec.queries,
                    seed_limit=spec.seed_limit,
                    max_pages=args.max_pages,
                    page_size=args.page_size,
                )

                ok, info = run_generator(generator, args.api_key, seeds_file, out_path, spec, python_exe)
                if ok:
                    print("[OK]", info)
                    break

                if attempt <= args.max_retries:
                    wait = min(60.0, (2 ** (attempt - 1)) + rng.uniform(0.0, 1.0))
                    print(f"[RETRY] generator failed (attempt {attempt}/{args.max_retries}) sleeping {wait:.1f}s")
                    time.sleep(wait)
                    continue

                print("[FAIL]", info)
                failures.append((spec, info))
                break

            except Exception as e:
                if attempt <= args.max_retries:
                    wait = min(60.0, (2 ** (attempt - 1)) + rng.uniform(0.0, 1.0))
                    print(f"[RETRY] seed fetch/cache error: {e} (attempt {attempt}/{args.max_retries}) sleeping {wait:.1f}s")
                    time.sleep(wait)
                    continue
                print("[FAIL]", str(e))
                failures.append((spec, str(e)))
                break

        time.sleep(max(0.0, args.sleep_between))

    print("\n====================")
    print(f"Batch finished. Total={total} Failed={len(failures)} Success={total - len(failures)}")


if __name__ == "__main__":
    main()
