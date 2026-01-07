from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from bisect import bisect_left

# -----------------------------
# Models
# -----------------------------
@dataclass
class ProgramItem:
    program_id: str
    start: int
    end: int
    genre: str
    score: int


@dataclass
class Channel:
    channel_id: int
    channel_name: str
    programs: List[ProgramItem]


@dataclass
class PriorityBlock:
    start: int
    end: int
    allowed_channels: List[int]


@dataclass
class TimePreference:
    start: int
    end: int
    preferred_genre: str
    bonus: int


@dataclass
class Instance:
    opening_time: int
    closing_time: int
    min_duration: int
    max_consecutive_genre: int
    channels_count: int
    switch_penalty: int
    termination_penalty: int
    priority_blocks: List[PriorityBlock]
    time_preferences: List[TimePreference]
    channels: List[Channel]


def to_json(obj) -> str:
    def default(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")
    return json.dumps(obj, default=default, ensure_ascii=False, indent=2)


# -----------------------------
# Utilities
# -----------------------------
SLUG_RE = re.compile(r"[^a-z0-9]+", re.UNICODE)

def slugify(s: str, max_len: int = 48) -> str:
    s = (s or "").lower().strip()
    s = s.replace("&", " and ")
    s = SLUG_RE.sub("_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s

def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x

def stable_int(s: str, mod: int = 10_000_000) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def overlaps(a1: int, a2: int, b1: int, b2: int) -> bool:
    return not (a2 <= b1 or b2 <= a1)

def iso_to_dt(s: str) -> datetime:
    if s.endswith("Z"):
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    return datetime.fromisoformat(s)

def dt_to_minutes_from_window(start_dt_utc: datetime, window_start_utc: datetime) -> int:
    delta = start_dt_utc - window_start_utc
    return int(delta.total_seconds() // 60)


# -----------------------------
# YouTube API
# -----------------------------
YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

def http_get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (YouTubeInstanceGenerator/1.0)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    return json.loads(data)

def yt_search_upcoming(api_key: str, q: str, max_pages: int, page_size: int,
                       region_code: Optional[str] = None, relevance_language: Optional[str] = None) -> List[str]:
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

        items = js.get("items", [])
        for it in items:
            vid = (it.get("id") or {}).get("videoId")
            if vid:
                video_ids.append(vid)

        page_token = js.get("nextPageToken")
        if not page_token:
            break

        time.sleep(0.1)

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
        time.sleep(0.1)
    return out


# -----------------------------
# Genre mapping
# -----------------------------
TARGET_GENRES = [
    "news", "sports", "movie", "series", "documentary", "talk",
    "music", "kids", "gaming", "education", "religion", "tech",
    "finance", "lifestyle", "entertainment", "unknown"
]

LEXICON: Dict[str, List[str]] = {
    "news": [
        "news", "breaking", "headline", "journal", "bulletin",
        "noticias", "actualites", "nachrichten", "telegiornale", "jornal",
        "lajme", "ditari", "edicion", "informativ", "kronike", "vesti", "vijesti",
        "новости", "вести", "дневник", "новини",
        "haber", "اخبار", "نشرة", "新闻", "新聞", "ニュース"
    ],
    "sports": [
        "sport", "sports", "match", "game", "live", "highlights",
        "futbol", "fútbol", "football", "soccer", "basket", "nba", "ufc", "boxing",
        "ndeshje", "liga", "kampionat",
        "спорт", "матч", "футбол",
        "spor", "mac", "رياضة", "مباراة", "กีฬา", "경기"
    ],
    "gaming": [
        "gaming", "gameplay", "esports", "e-sports", "twitch", "streamer",
        "minecraft", "fortnite", "valorant", "cs2", "csgo", "dota", "league of legends", "lol",
        "gaming live", "ゲーム", "игра", "juego", "jogos"
    ],
    "music": [
        "music", "song", "concert", "festival", "live music", "dj", "mix", "playlist",
        "muzik", "koncert", "musique", "musica", "musik",
        "музыка", "концерт",
        "موسيقى", "حفلة", "เพลง", "음악", "音楽"
    ],
    "talk": [
        "talk", "show", "debate", "interview", "discussion", "panel", "podcast", "studio",
        "debat", "intervist", "analize",
        "интервью", "дебат",
        "حوار", "نقاش", "토크", "討論"
    ],
    "documentary": [
        "documentary", "doc", "history", "nature", "science", "investigation", "true story",
        "dokumentar", "documentaire", "documental", "doku",
        "документ", "наука", "природа",
        "وثائقي", "สารคดี", "纪录片"
    ],
    "education": [
        "education", "learning", "lesson", "course", "lecture", "class", "tutorial", "how to",
        "university", "seminar", "webinar",
        "обучение", "урок",
        "تعليم", "درس", "课程", "강의", "講義"
    ],
    "tech": [
        "tech", "technology", "developer", "programming", "coding", "ai", "ml", "data", "cloud",
        "python", "c#", "dotnet", ".net", "javascript", "react", "kubernetes",
        "teknoloji", "технолог", "tecnologia", "technologie"
    ],
    "finance": [
        "finance", "markets", "stock", "stocks", "crypto", "bitcoin", "forex",
        "economy", "economics", "trading", "investing",
        "finanza", "finanzas", "markt", "börse",
        "эконом", "акции",
        "مال", "اقتصاد", "استثمار"
    ],
    "religion": [
        "religion", "church", "mass", "prayer", "sermon",
        "islam", "quran", "ramadan",
        "kisha", "mesh",
        "церковь", "молитва",
        "دين", "صلاة", "قرآن"
    ],
    "kids": [
        "kids", "children", "cartoon", "animation", "anime kids",
        "femije", "vizatimor",
        "дети", "мульт",
        "اطفال", "كرتون",
        "어린이", "子供", "儿童"
    ],
    "movie": [
        "movie", "film", "cinema", "feature film", "premiere",
        "pelicula", "filme", "kino", "spielfilm",
        "фильм", "кино",
        "فيلم"
    ],
    "series": [
        "series", "episode", "season", "sitcom", "soap", "novela",
        "serial", "telenovela",
        "сериал", "серия",
        "مسلسل"
    ],
    "lifestyle": [
        "lifestyle", "vlog", "travel", "cooking", "recipe", "fitness", "workout", "yoga",
        "beauty", "makeup", "fashion", "food", "kitchen",
        "viaje", "cocina", "receta", "moda",
        "красота", "путешеств"
    ],
}

CATEGORY_HINTS = {
    "1": "film",         
    "2": "autos",        
    "10": "music",       
    "15": "pets",        
    "17": "sports",      
    "19": "travel",      
    "20": "gaming",      
    "22": "people",      
    "23": "comedy",      
    "24": "entertainment",
    "25": "news",        
    "26": "lifestyle",   
    "27": "education",
    "28": "science",     
    "29": "nonprofit",   
}

def infer_genre_from_seed(seed: dict) -> Tuple[str, float]:
    snippet = seed.get("snippet") or {}
    title = snippet.get("title") or ""
    desc = snippet.get("description") or ""
    chan = snippet.get("channelTitle") or ""
    cat_id = snippet.get("categoryId") or ""
    topics = ((seed.get("topicDetails") or {}).get("topicCategories")) or []

    text = normalize_text(f"{title} {desc} {chan} " + " ".join(topics))
    if not text:
        return "unknown", 0.1

    scores: Dict[str, float] = {g: 0.0 for g in TARGET_GENRES}

    for genre, kws in LEXICON.items():
        for kw in kws:
            kwn = normalize_text(kw)
            if kwn and kwn in text:
                scores[genre] += min(6.0, 2.0 + len(kwn) / 6.0)

    if cat_id and cat_id in CATEGORY_HINTS:
        hint = CATEGORY_HINTS[cat_id]
        if hint == "music":
            scores["music"] += 2.5
        elif hint == "sports":
            scores["sports"] += 2.5
        elif hint == "gaming":
            scores["gaming"] += 2.5
        elif hint == "news":
            scores["news"] += 2.5
        elif hint == "education":
            scores["education"] += 2.0
        elif hint == "science":
            scores["tech"] += 1.8
        elif hint == "film":
            scores["movie"] += 1.6
        elif hint == "lifestyle":
            scores["lifestyle"] += 1.6
        else:
            scores["entertainment"] += 1.2

    topic_text = normalize_text(" ".join(topics))
    if "sport" in topic_text:
        scores["sports"] += 1.2
    if "music" in topic_text:
        scores["music"] += 1.2
    if "gaming" in topic_text:
        scores["gaming"] += 1.2
    if "politic" in topic_text or "news" in topic_text:
        scores["news"] += 1.0
    if "education" in topic_text or "science" in topic_text or "technology" in topic_text:
        scores["education"] += 0.8
        scores["tech"] += 0.8

    best_genre = max(scores.items(), key=lambda x: x[1])[0]
    best_score = scores[best_genre]
    total = sum(scores.values()) + 1e-9
    conf = float(min(0.95, max(0.2, best_score / total)))

    if best_score < 1.5:
        return "unknown", 0.2
    return best_genre, conf


# -----------------------------
# Scoring
# -----------------------------
def time_weight_2days(minute: int) -> float:
    minute_in_day = minute % 1440
    if 1080 <= minute_in_day < 1410:
        return 1.22
    if 360 <= minute_in_day < 600:
        return 1.07
    if 0 <= minute_in_day < 300:
        return 0.92
    return 1.0

GENRE_BASE = {
    "news": 72,
    "sports": 80,
    "movie": 78,
    "series": 74,
    "documentary": 68,
    "talk": 62,
    "music": 60,
    "kids": 56,
    "gaming": 66,
    "education": 64,
    "religion": 50,
    "tech": 66,
    "finance": 65,
    "lifestyle": 63,
    "entertainment": 62,
    "unknown": 55,
}

def compute_score(seed_key: str, genre: str, title: str, start_min: int, duration: int, conf: float) -> int:
    base = GENRE_BASE.get(genre, 60)

    if duration < 20:
        dur = -6
    elif duration < 45:
        dur = +2
    elif duration <= 120:
        dur = +6
    elif duration <= 240:
        dur = +4
    else:
        dur = +1

    conf_bonus = int(10 * (conf - 0.2))

    noise = (stable_int(f"{seed_key}|{title}|{start_min}") % 13) - 6

    w = time_weight_2days(start_min)

    score = int((base + dur + conf_bonus + noise) * w)
    return clamp(score, 5, 100)


# -----------------------------
# Program schedule synthesis
# -----------------------------
def pick_program_duration(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.25:
        return rng.choice([20, 25, 30, 35, 40])
    if r < 0.60:
        return rng.choice([45, 60, 75, 90])
    if r < 0.85:
        return rng.choice([100, 120, 150, 180])
    return rng.choice([210, 240, 300])

def jitter_minutes(rng: random.Random, max_jitter: int) -> int:
    return rng.randint(-max_jitter, max_jitter)

def synthesize_channel_schedule(
    channel_id: int,
    channel_name: str,
    seeds: List[dict],
    window_start_utc: datetime,
    opening_time: int,
    closing_time: int,
    min_programs: int,
    max_programs: int,
    rng: random.Random,
) -> Channel:
    closing_time = max(closing_time, opening_time + 1)
    target_programs = rng.randint(min_programs, max_programs)
    pool_size = rng.randint(2, 6)
    pool = rng.sample(seeds, k=min(pool_size, len(seeds))) if seeds else []

    anchors: List[Tuple[int, dict]] = []
    for s in pool:
        lsd = s.get("liveStreamingDetails") or {}
        ss = lsd.get("scheduledStartTime")
        if not ss:
            continue
        dt = iso_to_dt(ss).astimezone(timezone.utc)
        m = dt_to_minutes_from_window(dt, window_start_utc)
        if opening_time <= m < closing_time:
            anchors.append((m, s))

    anchors.sort(key=lambda x: x[0])

    programs: List[ProgramItem] = []
    attempts = 0
    max_attempts = target_programs * 40

    def add_program(start_m: int, dur: int, seed: dict, kind_tag: str) -> None:
        nonlocal attempts
        attempts += 1

        start_m = clamp(start_m, opening_time, closing_time - 1)
        end_m = clamp(start_m + dur, opening_time + 1, closing_time)
        if end_m <= start_m:
            return

        snippet = seed.get("snippet") or {}
        title = snippet.get("title") or f"{kind_tag} Live"
        desc = snippet.get("description") or ""
        video_id = seed.get("id") or seed.get("id", "")
        seed_key = str(video_id) if isinstance(video_id, str) else json.dumps(video_id)

        genre, conf = infer_genre_from_seed(seed)
        score = compute_score(seed_key, genre, title, start_m, end_m - start_m, conf)

        pid = f"ch{channel_id}_{slugify(title, 30)}_{start_m:04d}"
        item = ProgramItem(program_id=pid, start=start_m, end=end_m, genre=genre, score=score)
        insert_non_overlapping(programs, item)

    for (m, seed) in anchors[:max(2, target_programs // 3)]:
        dur = pick_program_duration(rng)
        add_program(m, dur, seed, "Upcoming")

    while len(programs) < target_programs and attempts < max_attempts:
        seed = rng.choice(pool) if pool else rng.choice(seeds)
        if rng.random() < 0.45:
            day = rng.choice([0, 1])
            base = day * 1440 + rng.randint(1080, 1410)
            start_m = base + jitter_minutes(rng, 20)
        else:
            start_m = rng.randint(opening_time, closing_time - 1)

        dur = pick_program_duration(rng)
        add_program(start_m, dur, seed, "Synthetic")

        if len(programs) > max_programs:
            break

    programs.sort(key=lambda p: (p.start, p.end))
    cleaned: List[ProgramItem] = []
    for p in programs:
        if not cleaned:
            cleaned.append(p)
            continue
        last = cleaned[-1]
        if p.start < last.end:
            if p.score > last.score:
                last.end = p.start
                if last.end - last.start >= 5:
                    cleaned[-1] = last
                else:
                    cleaned.pop()
                cleaned.append(p)
            else:
                p.start = last.end
                if p.end - p.start >= 5:
                    cleaned.append(p)
        else:
            cleaned.append(p)

    cleaned = [p for p in cleaned if (p.end - p.start) >= 5]
    programs = strict_non_overlap_sweep(programs)
    return Channel(channel_id=channel_id, channel_name=channel_name, programs=cleaned)

def strict_non_overlap_sweep(items: List[ProgramItem]) -> List[ProgramItem]:
    items.sort(key=lambda p: (p.start, p.end))
    out: List[ProgramItem] = []
    last_end = -1
    for p in items:
        if p.start >= last_end:
            out.append(p)
            last_end = p.end
    return out

# -----------------------------
# Time preferences
# -----------------------------
def generate_time_preferences_2days(channels: List[Channel], opening: int, closing: int, blocks_per_day: int = 6) -> List[TimePreference]:
    total_minutes = closing - opening
    blocks = blocks_per_day * 2
    step = max(30, total_minutes // blocks)

    def priors(minute: int) -> List[str]:
        m = minute % 1440
        if 300 <= m < 600:
            return ["news", "talk", "education"]
        if 600 <= m < 1020:
            return ["talk", "documentary", "gaming", "lifestyle"]
        if 1020 <= m < 1410:
            return ["sports", "movie", "music", "series"]
        return ["music", "movie", "gaming"]

    prefs: List[TimePreference] = []
    cur = opening
    while cur < closing:
        nxt = min(closing, cur + step)

        weights: Dict[str, float] = {}
        for ch in channels:
            for p in ch.programs:
                if not overlaps(p.start, p.end, cur, nxt):
                    continue
                ov = min(p.end, nxt) - max(p.start, cur)
                mass = ov * (0.35 + p.score / 100.0)
                weights[p.genre] = weights.get(p.genre, 0.0) + mass

        if not weights:
            pref = priors(cur)[0]
            prefs.append(TimePreference(start=cur, end=nxt, preferred_genre=pref, bonus=18))
            cur = nxt
            continue

        non_unknown = {g: w for g, w in weights.items() if g != "unknown"}
        source = non_unknown if non_unknown else weights

        total = sum(source.values()) + 1e-9
        best_g, best_w = max(source.items(), key=lambda x: x[1])
        dominance = best_w / total

        weak = dominance < 0.33
        if weak:
            boosted = dict(source)
            for g in priors(cur):
                boosted[g] = boosted.get(g, 0.0) + 0.12 * total
            best_g, best_w = max(boosted.items(), key=lambda x: x[1])
            dominance = best_w / (sum(boosted.values()) + 1e-9)

        prime_boost = 1.0 if (cur % 1440) < 1020 else 1.15
        bonus = int(clamp(int(12 + 60 * dominance * prime_boost), 10, 70))

        prefs.append(TimePreference(start=cur, end=nxt, preferred_genre=best_g, bonus=bonus))
        cur = nxt

    return prefs


# -----------------------------
# Priority blocks
# -----------------------------
def generate_priority_blocks_2days(
    channels: List[Channel],
    opening: int,
    closing: int,
    bucket: int = 15,
    max_blocks: int = 6,
) -> List[PriorityBlock]:
    buckets = list(range(opening, closing, bucket))
    bucket_mass = {b: 0.0 for b in buckets}
    bucket_ch_mass: Dict[int, Dict[int, float]] = {b: {} for b in buckets}

    for ch in channels:
        for p in ch.programs:
            for b in buckets:
                b_end = min(closing, b + bucket)
                if not overlaps(p.start, p.end, b, b_end):
                    continue
                ov = min(p.end, b_end) - max(p.start, b)
                mass = ov * (p.score / 100.0)
                bucket_mass[b] += mass
                bucket_ch_mass[b][ch.channel_id] = bucket_ch_mass[b].get(ch.channel_id, 0.0) + mass

    masses = sorted(bucket_mass.values())
    if not masses:
        return []

    idx = int(0.93 * (len(masses) - 1))
    thresh = masses[idx] if masses else 0.0

    peak = [b for b, m in bucket_mass.items() if m >= thresh and m > 0.0]
    peak.sort()
    if not peak:
        return []

    raw: List[Tuple[int, int]] = []
    s = peak[0]
    prev = peak[0]
    for b in peak[1:]:
        if b == prev + bucket:
            prev = b
        else:
            raw.append((s, min(closing, prev + bucket)))
            s = b
            prev = b
    raw.append((s, min(closing, prev + bucket)))

    merged = [(a, b) for a, b in raw if (b - a) >= 30]
    merged = merged[:max_blocks]

    blocks: List[PriorityBlock] = []
    for a, b in merged:
        ch_mass: Dict[int, float] = {}
        for t in buckets:
            if t < a or t >= b:
                continue
            for cid, m in bucket_ch_mass[t].items():
                ch_mass[cid] = ch_mass.get(cid, 0.0) + m

        ordered = sorted(ch_mass.items(), key=lambda x: x[1], reverse=True)
        total_m = sum(ch_mass.values()) or 1.0

        allowed: List[int] = []
        acc = 0.0
        for cid, m in ordered:
            allowed.append(cid)
            acc += m
            if acc / total_m >= 0.65 and len(allowed) >= 3:
                break

        blocks.append(PriorityBlock(start=a, end=b, allowed_channels=allowed))

    return blocks


# -----------------------------
# Main build
# -----------------------------
def build_instance_from_youtube(
    api_key: str,
    start_date_utc: datetime,
    opening_time: int,
    closing_time: int,
    channels_count: int,
    min_programs: int,
    max_programs: int,
    min_duration: int,
    max_consecutive_genre: int,
    switch_penalty: int,
    termination_penalty: int,
    search_queries: List[str],
    region_code: Optional[str],
    relevance_language: Optional[str],
    max_search_pages: int,
    search_page_size: int,
    seed_limit: int,
    rng_seed: int,
    seeds_override: Optional[List[dict]] = None,
) -> Instance:
    rng = random.Random(rng_seed)

    if seeds_override is not None and len(seeds_override) > 0:
        seeds = seeds_override
    else:
        all_video_ids: List[str] = []
        for q in search_queries:
            ids = yt_search_upcoming(
                api_key=api_key,
                q=q,
                max_pages=max_search_pages,
                page_size=search_page_size,
                region_code=region_code,
                relevance_language=relevance_language,
            )
            all_video_ids.extend(ids)

        seen = set()
        uniq_ids = []
        for v in all_video_ids:
            if v not in seen:
                seen.add(v)
                uniq_ids.append(v)

        if not uniq_ids:
            raise RuntimeError("No upcoming livestreams found. Try different queries or remove region/language filters.")

        uniq_ids = uniq_ids[:seed_limit]
        seeds = yt_videos_details(api_key, uniq_ids)

        if not seeds:
            raise RuntimeError("Failed to fetch video details for seeds.")

    channels: List[Channel] = []
    for cid in range(channels_count):
        base_seed = seeds[cid % len(seeds)]
        sn = base_seed.get("snippet") or {}
        base_title = sn.get("channelTitle") or "Live Channel"
        g, conf = infer_genre_from_seed(base_seed)

        label = g.upper() if g != "unknown" else "LIVE"
        channel_name = f"YT - {label} - {base_title} #{cid:04d}"

        ch = synthesize_channel_schedule(
            channel_id=cid,
            channel_name=channel_name,
            seeds=seeds,
            window_start_utc=start_date_utc,
            opening_time=opening_time,
            closing_time=closing_time,
            min_programs=min_programs,
            max_programs=max_programs,
            rng=rng,
        )
        channels.append(ch)

    time_prefs = generate_time_preferences_2days(channels, opening_time, closing_time, blocks_per_day=6)
    prio_blocks = generate_priority_blocks_2days(channels, opening_time, closing_time, bucket=15, max_blocks=6)

    return Instance(
        opening_time=opening_time,
        closing_time=closing_time,
        min_duration=min_duration,
        max_consecutive_genre=max_consecutive_genre,
        channels_count=len(channels),
        switch_penalty=switch_penalty,
        termination_penalty=termination_penalty,
        priority_blocks=prio_blocks,
        time_preferences=time_prefs,
        channels=channels,
    )

def insert_non_overlapping(programs: List[ProgramItem], item: ProgramItem) -> bool:

    starts = [p.start for p in programs]
    i = bisect_left(starts, item.start)

    if i > 0 and programs[i - 1].end > item.start:
        return False

    if i < len(programs) and item.end > programs[i].start:
        return False

    programs.insert(i, item)
    return True

def load_seeds_from_file(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("--seeds-json must be a JSON array (list) of seed video items")
    return data
# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("YOUTUBE_API_KEY", ""), help="YouTube API key (or set env YOUTUBE_API_KEY)")
    ap.add_argument("--start-utc", default="2026-01-04T00:00:00Z", help="Window start UTC (ISO8601), e.g. 2026-01-04T00:00:00Z")
    ap.add_argument("--opening", type=int, default=0)
    ap.add_argument("--closing", type=int, default=2880)
    ap.add_argument("--channels", type=int, default=2000)
    ap.add_argument("--min-programs", type=int, default=5)
    ap.add_argument("--max-programs", type=int, default=30)
    ap.add_argument("--min-duration", type=int, default=10)
    ap.add_argument("--max-consecutive-genre", type=int, default=2)
    ap.add_argument("--switch-penalty", type=int, default=10)
    ap.add_argument("--termination-penalty", type=int, default=20)
    ap.add_argument("--region", default="", help="Optional regionCode, e.g. US, GB, DE")
    ap.add_argument("--lang", default="", help="Optional relevanceLanguage, e.g. en, de, es")
    ap.add_argument("--query", action="append", default=[], help='Search query, repeatable. Default uses ["live", "livestream", "esports live", "news live", "music live"]')
    ap.add_argument("--max-pages", type=int, default=10, help="Search pages per query (each page up to 50 results)")
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--seed-limit", type=int, default=800, help="Max unique seed videos to fetch details for (quota-friendly)")
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument("--out", default="instance_youtube_2days.json")
    ap.add_argument("--seeds-json", default="", help="Path to a cached seeds JSON file (list of video items). If provided, generator will NOT call YouTube search.")
    ap.add_argument("--save-seeds-json", default="", help="Optional path to save fetched seeds JSON for later reuse.")

    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api-key or set env var YOUTUBE_API_KEY")

    start_dt = iso_to_dt(args.start_utc).astimezone(timezone.utc)

    queries = args.query or ["live", "livestream", "esports live", "news live", "music live", "sports live", "tech live", "crypto live"]
    region = args.region.strip().upper() or None
    lang = args.lang.strip().lower() or None

    seeds_override = None
    if args.seeds_json:
        seeds_override = load_seeds_from_file(args.seeds_json)

    inst = build_instance_from_youtube(
        api_key=args.api_key,
        start_date_utc=start_dt,
        opening_time=args.opening,
        closing_time=args.closing,
        channels_count=args.channels,
        min_programs=args.min_programs,
        max_programs=args.max_programs,
        min_duration=args.min_duration,
        max_consecutive_genre=args.max_consecutive_genre,
        switch_penalty=args.switch_penalty,
        termination_penalty=args.termination_penalty,
        search_queries=queries,
        region_code=region,
        relevance_language=lang,
        max_search_pages=args.max_pages,
        search_page_size=args.page_size,
        seed_limit=args.seed_limit,
        rng_seed=args.rng_seed,
        seeds_override=seeds_override,
    )
    if args.save_seeds_json and seeds_override is None:
        pass
    out = to_json(inst)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
