from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Iterable
import unicodedata
from collections import defaultdict
import gzip
import io

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None 

import urllib.request
import xml.etree.ElementTree as ET


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


# -----------------------------
# Utilities
# -----------------------------

SLUG_RE = re.compile(r"[^a-z0-9]+")

def slugify(s: str, max_len: int = 40) -> str:
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = SLUG_RE.sub("_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def stable_int(s: str, mod: int = 10_000_000) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def parse_xmltv_dt(s: str) -> datetime:
    """
    XMLTV time format often: "YYYYMMDDhhmmss +0000"
    We'll parse timezone offset and return aware datetime.
    """
    s = s.strip()
    m = re.match(r"^(\d{14})\s*([+-]\d{4})$", s)
    if not m:
        raise ValueError(f"Unrecognized XMLTV datetime: {s!r}")
    dt_part, off_part = m.group(1), m.group(2)

    dt_naive = datetime.strptime(dt_part, "%Y%m%d%H%M%S")
    sign = 1 if off_part[0] == "+" else -1
    hours = int(off_part[1:3])
    mins = int(off_part[3:5])
    offset = timezone(sign * timedelta(hours=hours, minutes=mins))
    return dt_naive.replace(tzinfo=offset)


def minutes_since_midnight(local_dt: datetime) -> int:
    return local_dt.hour * 60 + local_dt.minute


def overlaps(a1: int, a2: int, b1: int, b2: int) -> bool:
    return not (a2 <= b1 or b2 <= a1)


# -----------------------------
# Genre classification
# -----------------------------

GENRE_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("news", ["news", "lajme", "edicion", "journal", "breaking", "raport", "informim"]),
    ("sports", ["sport", "match", "ndeshje", "liga", "uefa", "fifa", "champions", "basket", "tenis"]),
    ("movie", ["film", "movie", "cinema", "thriller", "komedi", "drame", "aksion"]),
    ("series", ["serial", "telenovel", "episode", "sezon", "episod", "sitcom"]),
    ("talk", ["talk", "show", "debat", "intervist", "studio", "panel", "discussion"]),
    ("documentary", ["doc", "documentary", "histori", "history", "science", "natyre", "nature"]),
    ("kids", ["kids", "karton", "cartoon", "anim", "femije", "masha", "peppa"]),
    ("music", ["music", "koncert", "clip", "festival", "top", "hit", "muzik"]),
    ("religion", ["relig", "ramazan", "kur'an", "kisha", "mass", "church", "islam"]),
    ("weather", ["weather", "moti", "forecast", "parashikim"]),
    ("education", ["edu", "mesim", "lecture", "kurs", "learning", "shkoll"]),
    ("promo", ["promo", "reklam", "advert", "trailer", "ipko promo"]),
]

def infer_genre(title: str, desc: str) -> str:
    t = f"{title} {desc}".lower()
    for genre, kws in GENRE_KEYWORDS:
        for kw in kws:
            if kw in t:
                return genre
    return "entertainment"


# -----------------------------
# Scoring
# -----------------------------

def time_weight(minute: int) -> float:
    if 360 <= minute < 600:
        return 1.08
    if 600 <= minute < 1020:
        return 1.00
    if 1020 <= minute < 1380:
        return 1.18
    return 0.95


GENRE_BASE = {
    "news": 72,
    "sports": 78,
    "movie": 85,
    "series": 76,
    "documentary": 70,
    "talk": 63,
    "music": 58,
    "kids": 55,
    "weather": 40,
    "education": 60,
    "religion": 52,
    "promo": 25,
    "entertainment": 65,
}

BOOST_KEYWORDS = {
    "live": 8, "direkt": 8, "final": 10, "premiere": 7, "ekskluzive": 6,
    "exclusive": 6, "special": 6, "derbi": 8
}

def compute_score(channel_key: str, title: str, desc: str, genre: str, start_min: int, duration: int) -> int:
    base = GENRE_BASE.get(genre, 60)

    if duration < 15:
        dur_bonus = -8
    elif duration < 30:
        dur_bonus = -2
    elif duration <= 120:
        dur_bonus = +6
    else:
        dur_bonus = +2

    text = f"{title} {desc}".lower()
    kw_bonus = 0
    for kw, pts in BOOST_KEYWORDS.items():
        if kw in text:
            kw_bonus += pts

    noise = (stable_int(f"{channel_key}|{title}|{start_min}") % 13) - 6

    w = time_weight(start_min)
    score = int((base + dur_bonus + kw_bonus + noise) * w)
    return clamp(score, 5, 100)


# -----------------------------
# Streaming XML parser
# -----------------------------

def stream_epg(url_or_path: str) -> Iterable[Tuple[str, dict]]:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        resp = urllib.request.urlopen(url_or_path)
        raw = resp.read()
        if url_or_path.endswith(".gz"):
            fh = io.BytesIO(gzip.decompress(raw))
        else:
            fh = io.BytesIO(raw)
    else:
        if url_or_path.endswith(".gz"):
            fh = gzip.open(url_or_path, "rb")
        else:
            fh = open(url_or_path, "rb")

    ctx = ET.iterparse(fh, events=("end",))
    for event, elem in ctx:
        tag = elem.tag

        if tag == "channel":
            ch_id = elem.attrib.get("id", "").strip()
            display = elem.findtext("display-name") or ch_id
            yield ("channel", {"id": ch_id, "name": display.strip()})
            elem.clear()

        elif tag == "programme":
            ch = elem.attrib.get("channel", "").strip()
            start_s = elem.attrib.get("start", "")
            stop_s = elem.attrib.get("stop", "")
            title_el = elem.find("title")
            desc_el = elem.find("desc")
            title = (title_el.text or "").strip() if title_el is not None else ""
            desc = (desc_el.text or "").strip() if desc_el is not None else ""
            try:
                start_dt = parse_xmltv_dt(start_s)
                stop_dt = parse_xmltv_dt(stop_s)
            except Exception:
                elem.clear()
                continue

            categories = [ (c.text or "").strip() for c in elem.findall("category") if c is not None ]
            yield ("programme", {
                "channel": ch,
                "start": start_dt,
                "stop": stop_dt,
                "title": title,
                "desc": desc,
                "categories": categories,
            })
            elem.clear()

    try:
        fh.close()
    except Exception:
        pass


# -----------------------------
# Build Instance
# -----------------------------

def build_instance(
    epg_source: str,
    target_date: date,
    tz_name: str,
    opening_time: int,
    closing_time: int,
    min_duration: int,
    max_consecutive_genre: int,
    switch_penalty: int,
    termination_penalty: int,
) -> Instance:
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo not available. Use Python 3.9+.")
    tz = ZoneInfo(tz_name)

    local_start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=tz)
    local_end = local_start + timedelta(days=1)

    channel_names: Dict[str, str] = {}
    programmes_by_channel: Dict[str, List[ProgramItem]] = {}

    for kind, payload in stream_epg(epg_source):
        if kind == "channel":
            channel_names[payload["id"]] = payload["name"]
            continue

        if kind == "programme":
            ch = payload["channel"]
            start_utc = payload["start"].astimezone(timezone.utc)
            stop_utc = payload["stop"].astimezone(timezone.utc)

            start_local = start_utc.astimezone(tz)
            stop_local = stop_utc.astimezone(tz)

            if stop_local <= local_start or start_local >= local_end:
                continue

            prog_start_min = minutes_since_midnight(max(start_local, local_start))
            prog_stop_min = minutes_since_midnight(min(stop_local, local_end))

            if not overlaps(prog_start_min, prog_stop_min, opening_time, closing_time):
                continue

            s = clamp(prog_start_min, opening_time, closing_time)
            e = clamp(prog_stop_min, opening_time, closing_time)
            if e <= s:
                continue

            title = payload["title"] or "Untitled"
            desc = payload["desc"] or ""
            genre, gconf = infer_genre_advanced(title, desc, payload.get("categories", []))
            duration = e - s
            if duration < min_duration:
                if genre not in ("news", "weather", "sports"):
                    continue

            prog_id = f"{slugify(ch, 18)}_{slugify(title, 24)}_{target_date.isoformat()}_{s:04d}"
            score = compute_score(ch, title, desc, genre, s, duration)

            programmes_by_channel.setdefault(ch, []).append(
                ProgramItem(program_id=prog_id, start=s, end=e, genre=genre, score=score)
            )

    for ch, items in programmes_by_channel.items():
        items.sort(key=lambda p: (p.start, p.end))
        merged: List[ProgramItem] = []
        for p in items:
            if not merged:
                merged.append(p)
                continue
            last = merged[-1]
            if p.start < last.end:
                if p.score > last.score:
                    last.end = p.start
                    if last.end - last.start < min_duration and last.genre not in ("news", "weather", "sports"):
                        merged.pop()
                    merged.append(p)
                else:
                    p.start = last.end
                    if p.end - p.start >= min_duration or p.genre in ("news", "weather", "sports"):
                        merged.append(p)
            else:
                merged.append(p)
        programmes_by_channel[ch] = [p for p in merged if p.end > p.start]

    def channel_rank(ch: str) -> float:
        items = programmes_by_channel.get(ch, [])
        if not items:
            return -1e9
        coverage = sum((p.end - p.start) for p in items)
        avg_score = sum(p.score * (p.end - p.start) for p in items) / max(1, coverage)
        return 0.55 * coverage + 3.2 * avg_score

    selected_keys = sorted(programmes_by_channel.keys())
    channels_count = len(selected_keys)

    key_to_id = {k: i for i, k in enumerate(selected_keys)}

    channels: List[Channel] = []
    for k in selected_keys:
        channels.append(
            Channel(
                channel_id=key_to_id[k],
                channel_name=channel_names.get(k, k),
                programs=programmes_by_channel.get(k, []),
            )
        )

    # -----------------------------
    # Advanced Time Preferences
    # -----------------------------
    blocks = 5
    total = closing_time - opening_time
    step = max(30, total // blocks)
    time_prefs = generate_time_preferences_advanced(
        selected_keys=selected_keys,
        programmes_by_channel=programmes_by_channel,
        opening_time=opening_time,
        closing_time=closing_time,
        blocks=25,
    )

    # -----------------------------
    # Advanced Priority Blocks
    # -----------------------------
    bucket = 15
    buckets = list(range(opening_time, closing_time, bucket))

    bucket_mass: Dict[int, float] = {b: 0.0 for b in buckets}
    bucket_channel_mass: Dict[int, Dict[int, float]] = {b: {} for b in buckets}

    for ch_key in selected_keys:
        ch_id = key_to_id[ch_key]
        for p in programmes_by_channel.get(ch_key, []):
            for b in buckets:
                b_end = min(closing_time, b + bucket)
                if not overlaps(p.start, p.end, b, b_end):
                    continue
                ov = min(p.end, b_end) - max(p.start, b)
                mass = ov * (p.score / 100.0)
                bucket_mass[b] += mass
                bucket_channel_mass[b][ch_id] = bucket_channel_mass[b].get(ch_id, 0.0) + mass

    if bucket_mass:
        masses = sorted(bucket_mass.values())
        thresh = masses[int(0.92 * (len(masses) - 1))] if len(masses) > 5 else (masses[-1] if masses else 0.0)

        peak_buckets = [b for b, m in bucket_mass.items() if m >= thresh and m > 0.0]
        peak_buckets.sort()

        raw_blocks: List[Tuple[int, int]] = []
        if peak_buckets:
            s = peak_buckets[0]
            prev = peak_buckets[0]
            for b in peak_buckets[1:]:
                if b == prev + bucket:
                    prev = b
                else:
                    raw_blocks.append((s, min(closing_time, prev + bucket)))
                    s = b
                    prev = b
            raw_blocks.append((s, min(closing_time, prev + bucket)))

        merged_blocks = [(s, e) for (s, e) in raw_blocks if (e - s) >= 30]
        merged_blocks = merged_blocks[:4]

        priority_blocks: List[PriorityBlock] = []
        for s, e in merged_blocks:
            ch_mass: Dict[int, float] = {}
            for b in buckets:
                if b < s or b >= e:
                    continue
                for ch_id, m in bucket_channel_mass[b].items():
                    ch_mass[ch_id] = ch_mass.get(ch_id, 0.0) + m

            ordered = sorted(ch_mass.items(), key=lambda x: x[1], reverse=True)
            allowed: List[int] = []
            total_m = sum(ch_mass.values()) or 1.0
            acc = 0.0
            for ch_id, m in ordered:
                allowed.append(ch_id)
                acc += m
                if acc / total_m >= 0.60 and len(allowed) >= 2:
                    break

            priority_blocks.append(PriorityBlock(start=s, end=e, allowed_channels=allowed))
    else:
        priority_blocks = []

    if len(priority_blocks) < 2:
        fallback = [
            PriorityBlock(start=720, end=780, allowed_channels=list(range(min(3, len(selected_keys))))),
            PriorityBlock(start=1215, end=closing_time, allowed_channels=list(range(min(3, len(selected_keys))))),
        ]
        seen = {(b.start, b.end) for b in priority_blocks}
        for b in fallback:
            if (b.start, b.end) not in seen and b.start >= opening_time and b.end <= closing_time:
                priority_blocks.append(b)

    instance = Instance(
        opening_time=opening_time,
        closing_time=closing_time,
        min_duration=min_duration,
        max_consecutive_genre=max_consecutive_genre,
        channels_count=len(selected_keys),
        switch_penalty=switch_penalty,
        termination_penalty=termination_penalty,
        priority_blocks=priority_blocks,
        time_preferences=time_prefs,
        channels=channels,
    )
    return instance


def to_json(obj) -> str:
    def default(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")
    return json.dumps(obj, default=default, ensure_ascii=False, indent=2)

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

GENRE_LEXICON = {
    "news": [
        "news", "breaking", "headline", "journal", "bulletin",
        "lajme", "ditari", "edicion", "informativ", "kronike", "vesti", "vijesti",
        "nachrichten", "journal", "journale", "actualites", "noticias", "tg", "telegiornale",
        "jornal", "notiziario",
        "новости", "вести", "дневник", "новини",
        "haber",
        "اخبار", "نشرة",
    ],
    "sports": [
        "sport", "sports", "match", "game", "live",
        "ndeshje", "futboll", "basket", "tenis", "liga", "kampionat",
        "fussball", "bundesliga", "ligue", "liga", "calcio",
        "deporte", "partido",
        "спорт", "матч", "футбол", "баскетбол", "теннис",
        "spor", "mac",
        "رياضة", "مباراة",
    ],
    "movie": [
        "movie", "film", "cinema", "feature", "premiere",
        "pelicula", "filme", "cine", "cinema",
        "kino", "spielfilm",
        "фильм", "кино",
        "film artistik",
        "فيلم",
    ],
    "series": [
        "series", "episode", "season", "sitcom", "drama series",
        "serial", "telenovel", "telenovela", "novela", "soap",
        "serie", "episodio", "saison",
        "сериал", "серия", "сезон",
        "dizi",
        "مسلسل",
    ],
    "documentary": [
        "documentary", "doc", "history", "nature", "science",
        "dokumentar", "dokumentare", "histori", "shkence", "natyre",
        "doku", "dokumentation",
        "documentaire",
        "documental",
        "документальный", "документал", "наука", "природа",
        "belgesel",
        "وثائقي",
    ],
    "kids": [
        "kids", "children", "cartoon", "animation", "anime",
        "femije", "vizatimor", "animuar",
        "kinder", "zeichentrick",
        "enfants", "dessin anime",
        "niños", "infantil", "dibujos",
        "дети", "мульт", "мультик", "анимация",
        "çocuk",
        "اطفال", "كرتون",
    ],
    "music": [
        "music", "concert", "festival", "hit", "top",
        "muzik", "koncert", "festival",
        "musik", "konzert",
        "musique", "concert",
        "musica", "concierto",
        "музыка", "концерт",
        "muzik",
        "موسيقى", "حفلة",
    ],
    "talk": [
        "talk", "show", "debate", "interview", "discussion", "panel", "studio",
        "debat", "intervist", "analize", "opinion",
        "talkshow",
        "ток шоу", "дебат", "интервью", "студия",
        "sohbet", "tartisma",
        "حوار", "نقاش",
    ],
    "weather": [
        "weather", "forecast",
        "moti", "parashikim",
        "wetter",
        "meteo",
        "tiempo", "pronostico",
        "погода",
        "hava durumu",
        "طقس",
    ],
    "religion": [
        "religion", "church", "mass", "islam", "quran",
        "kisha", "mesh", "ramazan",
        "kirche", "messe",
        "église", "messe",
        "церковь", "молитва",
        "cami", "kur'an",
        "دين", "صلاة", "قرآن",
    ],
    "education": [
        "education", "learning", "lesson", "course", "lecture",
        "mesim", "kurs", "shkoll",
        "bildung", "unterricht",
        "éducation", "cours",
        "educacion", "clase",
        "обучение", "урок",
        "egitim", "ders",
        "تعليم", "درس",
    ],
    "promo": [
        "promo", "trailer", "advert", "advertisement",
        "reklam", "reklama",
        "werbung",
        "publicite",
        "реклама", "трейлер",
        "fragman",
        "اعلان",
    ],
}

CATEGORY_MAP = {
    "news": "news",
    "sports": "sports",
    "movie": "movie",
    "film": "movie",
    "series": "series",
    "tv series": "series",
    "documentary": "documentary",
    "kids": "kids",
    "children": "kids",
    "music": "music",
    "talk show": "talk",
    "weather": "weather",
    "religion": "religion",
    "education": "education",
    "promo": "promo",
    "advertisement": "promo",
    "shopping": "promo",
}

def infer_genre_advanced(title: str, desc: str, categories: List[str] | None = None) -> tuple[str, float]:
    """
    Returns (genre, confidence 0..1).
    Uses:
      1) <category> if present
      2) multilingual keyword match over normalized title/desc
    """
    categories = categories or []

    for c in categories:
        c_norm = normalize_text(c)
        if c_norm in CATEGORY_MAP:
            return CATEGORY_MAP[c_norm], 0.95
        for k, v in CATEGORY_MAP.items():
            if k in c_norm:
                return v, 0.90

    text = normalize_text(f"{title} {desc}")
    if not text:
        return "unknown", 0.0

    scores = defaultdict(int)
    for genre, keys in GENRE_LEXICON.items():
        for kw in keys:
            kw_norm = normalize_text(kw)
            if not kw_norm:
                continue
            if kw_norm in text:
                scores[genre] += max(2, min(8, len(kw_norm) // 3 + 2))

    if not scores:
        return "unknown", 0.15

    best_genre, best_score = max(scores.items(), key=lambda x: x[1])
    total = sum(scores.values())
    confidence = min(0.92, max(0.25, best_score / total))
    return best_genre, float(confidence)


def generate_time_preferences_advanced(
    selected_keys: List[str],
    programmes_by_channel: Dict[str, List[ProgramItem]],
    opening_time: int,
    closing_time: int,
    blocks: int = 5,
) -> List[TimePreference]:
    total = closing_time - opening_time
    step = max(30, total // blocks)

    def prior_genres_for_minute(m: int) -> List[str]:
        if 300 <= m < 600:
            return ["news", "talk"]
        if 600 <= m < 1020:
            return ["talk", "documentary", "kids"]
        if 1020 <= m < 1320:
            return ["movie", "series", "sports", "news"]
        return ["movie", "music", "series"]

    time_prefs: List[TimePreference] = []
    cur = opening_time

    while cur < closing_time:
        nxt = min(closing_time, cur + step)

        genre_weight: Dict[str, float] = {}
        total_mass = 0.0

        for ch_key in selected_keys:
            for p in programmes_by_channel.get(ch_key, []):
                if not overlaps(p.start, p.end, cur, nxt):
                    continue
                ov = min(p.end, nxt) - max(p.start, cur)

                mass = ov * (0.4 + (p.score / 100.0))
                total_mass += mass
                genre_weight[p.genre] = genre_weight.get(p.genre, 0.0) + mass

        if not genre_weight:
            pref = prior_genres_for_minute(cur)[0]
            time_prefs.append(TimePreference(start=cur, end=nxt, preferred_genre=pref, bonus=20))
            cur = nxt
            continue

        non_unknown = {g: w for g, w in genre_weight.items() if g not in ("unknown",)}
        source = non_unknown if non_unknown else genre_weight

        preferred, best_w = max(source.items(), key=lambda x: x[1])
        dominance = best_w / max(1e-9, sum(source.values()))

        unknown_share = genre_weight.get("unknown", 0.0) / max(1e-9, sum(genre_weight.values()))
        weak_signal = (dominance < 0.34) or (unknown_share > 0.55)

        if weak_signal:
            boosted = dict(source)
            for g in prior_genres_for_minute(cur):
                boosted[g] = boosted.get(g, 0.0) + 0.12 * sum(source.values())
            preferred, best_w = max(boosted.items(), key=lambda x: x[1])
            dominance = best_w / max(1e-9, sum(boosted.values()))

        prime = 1.0 if cur < 1020 else 1.15
        bonus = int(clamp(int(12 + 58 * dominance * prime), 10, 65))

        time_prefs.append(TimePreference(start=cur, end=nxt, preferred_genre=preferred, bonus=bonus))
        cur = nxt

    return time_prefs

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epg-url", required=True, help="URL or local path to XMLTV EPG (e.g. https://iptv-epg.org/files/epg-al.xml)")
    ap.add_argument("--date", required=True, help="Target date in YYYY-MM-DD (e.g. 2026-01-04)")
    ap.add_argument("--tz", default="Europe/Belgrade", help="Timezone for conversion (default: Europe/Belgrade)")
    ap.add_argument("--opening", type=int, default=480, help="Opening time in minutes from midnight (default 480=08:00)")
    ap.add_argument("--closing", type=int, default=1380, help="Closing time in minutes from midnight (default 1380=23:00)")
    ap.add_argument("--min-duration", type=int, default=30)
    ap.add_argument("--max-consecutive-genre", type=int, default=2)
    ap.add_argument("--switch-penalty", type=int, default=10)
    ap.add_argument("--termination-penalty", type=int, default=20)
    ap.add_argument("--out", default="", help="Output JSON path (optional). If omitted, prints to stdout.")
    args = ap.parse_args()

    y, m, d = map(int, args.date.split("-"))
    inst = build_instance(
        epg_source=args.epg_url,
        target_date=date(y, m, d),
        tz_name=args.tz,
        opening_time=args.opening,
        closing_time=args.closing,
        min_duration=args.min_duration,
        max_consecutive_genre=args.max_consecutive_genre,
        switch_penalty=args.switch_penalty,
        termination_penalty=args.termination_penalty,
    )

    out = to_json(inst)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
