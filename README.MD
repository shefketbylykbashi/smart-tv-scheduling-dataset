# Smart TV Scheduling Dataset

## Overview

This repository provides a **large-scale, multi-source dataset** for **Smart TV scheduling and optimization**.  
Instances are constructed from **real Electronic Program Guide (EPG) broadcasts** and **future-oriented livestream metadata**, then transformed into a **unified JSON instance format** suitable for benchmarking scheduling, optimization, and decision-support algorithms.

The dataset comprises **515 fully validated instances**, summarized below.

| Data source | Number of instances | Temporal coverage | 
|------------|--------------------|-------------------|
| IPTV (EPG) | 316 | 04–07 January 2026 |
| Provider-based EPG (PW) | 100 | 04–08 January 2026 |
| YouTube livestreams | 99 | Upcoming events |

---

## Data Sources

Instances are generated from **three complementary sources**, each contributing distinct structural and temporal characteristics.

### 1) IPTV EPG (iptv-epg.org)

- Public EPG feeds in **XMLTV** format
- Coverage: **80+ countries**
- Represents real broadcast schedules used by IPTV applications
- Typically high channel counts with dense and structured daily programming

Source: https://iptv-epg.org/

---

### 2) EPG.PW (epg.pw)

- Aggregated EPG portal providing **XMLTV** feeds
- Includes enriched channel metadata and program descriptions
- Coverage: **20+ countries**
- Mix of currently airing and future announced programs

Source: https://epg.pw/xmltv.html

---

### 3) YouTube Upcoming Livestreams (YouTube Data API v3)

- Uses YouTube Data API v3 to retrieve **upcoming livestreams**
- Livestreams are mapped into synthetic TV-like channels and programs
- Designed to represent **dynamic and future-oriented schedules** with uncertainty-like characteristics
- Two tiers are generated:
  - **Gold**: 300–1500 channels
  - **Premium**: 1000–2000 channels

Source: https://www.youtube.com/

---

## Repository Structure

```text
smart-tv-scheduling-dataset/
│
├── README.md
├── CITATION.cff
├── LICENSE
│
├── Instances/
│   ├── Instances_IPTV/
│   │   ├── Albania/
│   │   ├── Argentina/
│   │   └── ...
│   │
│   ├── Instances_PW/
│   │   ├── Australia_AU/
│   │   ├── Brazil_BR/
│   │   └── ...
│   │
│   │
│   └── Instances_Youtube/
│       ├── Premium/
│       └── Gold/
│
├── Scripts/
│   ├── iptv/
│   │   ├── build_instance_from_epg_iptv.py
│   │   └── batch_generate_instances_epg_iptv.py
│   │
│   ├── pw/
│   │   ├── build_instance_from_epg_pw.py
│   │   └── batch_generate_instances_epg_pw.py
│   │
│   ├── youtube/
│   │   ├── build_instance_from_youtube.py
│   │   └── batch_generate_instances_youtube.py
│   │
│   └── validators/
│       └── validate_and_rank_instances.py
│
└── Metadata/
    ├── instance_schema.json
    ├── parameters_description.md
    └── statistics_summary.md
```

---

## Instance Format

All instances are stored as JSON and follow a common schema across all sources.  

Schema: `Metadata/instance_schema.json`

### Key components

- Global scheduling parameters
- Channels and program lists
- Priority blocks
- Time preferences

Detailed documentation, see: `Metadata/parameters_description.md`.

---

## Instance Generation

The repository includes both single-instance generators and batch pipelines.

---
## Validation and Quality Control

All instances included in this repository undergo **automatic validation and consistency checks**
prior to release. The validation process ensures that:

- Programs within the same channel do not overlap in time
- All time intervals fall within the global scheduling window
- Priority blocks and time-preference intervals are well-formed and non-conflicting
- Genre, channel, and program identifiers are consistent
- All instances conform to the JSON schema defined in `Metadata/instance_schema.json`

Validation scripts are provided in `Scripts/validators/` and can be used to independently
verify instance correctness and compute basic quality statistics.

---

## Script Parameters

### Scripts/iptv/build_instance_from_epg_iptv.py

Builds a single instance from an IPTV XMLTV feed.

| Parameter | Type | Required | Default | Description |
|---------|------|----------|---------|-------------|
| `--epg-url` | string | Yes | – | XMLTV URL or local file path |
| `--date` | string | Yes | – | Target date in YYYY-MM-DD |
| `--tz` | string | No | Europe/Belgrade | Timezone for day boundaries |
| `--opening` | int | No | 480 | Scheduling window start (minutes) |
| `--closing` | int | No | 1380 | Scheduling window end (minutes) |
| `--min-duration` | int | No | 30 | Minimum program duration |
| `--max-consecutive-genre` | int | No | 2 | Max consecutive same-genre programs |
| `--switch-penalty` | int | No | 10 | Channel switching penalty |
| `--termination-penalty` | int | No | 20 | Early termination penalty |
| `--out` | string | No | stdout | Output JSON path |

---

### Scripts/iptv/batch_generate_instances_epg_iptv.py

Batch generation for multiple countries and dates using IPTV feeds.

Configured via constants inside the script.

| Setting | Type | Purpose |
|-------|------|---------|
| `OPENING` | int | Opening minute |
| `CLOSING` | int | Closing minute |
| `DATES` | list[str] | Dates to generate |
| `IPTV_SOURCES` | list[(country, url)] | IPTV XMLTV sources |

---

### Scripts/pw/build_instance_from_epg_pw.py

Builds a single instance from an EPG.PW XMLTV feed.

| Parameter | Type | Required | Default | Description |
|---------|------|----------|---------|-------------|
| `--epg-url` | string | Yes | – | XMLTV URL or local file path |
| `--date` | string | Yes | – | Target date |
| `--tz` | string | No | Europe/Belgrade | Timezone |
| `--opening` | int | No | 480 | Scheduling window start |
| `--closing` | int | No | 1380 | Scheduling window end |
| `--min-duration` | int | No | 30 | Minimum program duration |
| `--max-consecutive-genre` | int | No | 2 | Max consecutive genre |
| `--switch-penalty` | int | No | 10 | Switching penalty |
| `--termination-penalty` | int | No | 20 | Early termination penalty |
| `--out` | string | No | stdout | Output JSON |

---

### Scripts/pw/batch_generate_instances_epg_pw.py

Batch generator with caching, retry logic, and resume/overwrite controls.

| Parameter | Type | Required | Default | Description |
|---------|------|----------|---------|-------------|
| `--source` | string | No | epgpw | Provider selector |
| `--generator` | string | No | internal | Generator script |
| `--out-root` | string | No | internal | Output root |
| `--cache-dir` | string | No | internal | Cache folder |
| `--dates` | string | No | 2026-01-04 | Dates |
| `--opening` | int | No | internal | Opening minute |
| `--closing` | int | No | internal | Closing minute |
| `--sleep-between-downloads` | float | No | internal | Throttle |
| `--max-retries` | int | No | internal | Retry count |
| `--base-backoff` | float | No | internal | Backoff base |
| `--resume` | flag | No | False | Skip existing |
| `--overwrite` | flag | No | False | Overwrite |
| `--refresh-cache` | flag | No | False | Force refresh |
| `--limit-countries` | int | No | 0 | Limit countries |

---

### Scripts/youtube/build_instance_from_youtube.py

Builds a single instance from **upcoming YouTube livestream metadata**.

| Parameter | Type | Required | Default | Description |
|---------|------|----------|---------|-------------|
| `--api-key` | string | Yes | env | YouTube Data API key |
| `--start-utc` | string | No | 2026-01-04T00:00:00Z | Start window |
| `--opening` | int | No | 0 | Scheduling window start |
| `--closing` | int | No | 2880 | Scheduling window end |
| `--channels` | int | No | 2000 | Synthetic channels |
| `--min-programs` | int | No | 5 | Min programs |
| `--max-programs` | int | No | 30 | Max programs |
| `--min-duration` | int | No | 10 | Min duration |
| `--max-consecutive-genre` | int | No | 2 | Max consecutive genre |
| `--switch-penalty` | int | No | 10 | Switching penalty |
| `--termination-penalty` | int | No | 20 | Early termination |
| `--region` | string | No | empty | Region filter |
| `--lang` | string | No | empty | Language filter |
| `--query` | repeatable | No | internal | Search terms |
| `--max-pages` | int | No | 10 | Pages per query |
| `--page-size` | int | No | 50 | Results per page |
| `--seed-limit` | int | No | 800 | Seed limit |
| `--rng-seed` | int | No | 12345 | Random seed |
| `--out` | string | No | instance_youtube_2days.json | Output file |
| `--seeds-json` | string | No | empty | Cached seeds |
| `--save-seeds-json` | string | No | empty | Save seeds |

---

### Scripts/youtube/batch_generate_instances_youtube.py

Batch generation of **Gold and Premium** YouTube instances.

| Parameter | Type | Required | Default | Description |
|---------|------|----------|---------|-------------|
| `--api-key` | string | Yes | env | YouTube API key |
| `--generator` | string | No | internal | Generator |
| `--out-root` | string | No | Instances_Youtube | Output folder |
| `--cache-dir` | string | No | SeedsCache | Seed cache |
| `--premium-count` | int | No | 10 | Premium instances |
| `--gold-count` | int | No | 90 | Gold instances |
| `--start-range-from` | string | No | 2026-01-01T00:00:00Z | Start range |
| `--start-range-to` | string | No | 2026-12-31T00:00:00Z | End range |
| `--max-pages` | int | No | 2 | Pages |
| `--page-size` | int | No | 50 | Page size |
| `--sleep-between` | float | No | 0.25 | Delay |
| `--max-retries` | int | No | 5 | Retries |
| `--rng-seed` | int | No | 12345 | Random seed |
| `--resume` | flag | No | False | Skip |
| `--overwrite` | flag | No | False | Overwrite |

## Execution

> All commands below assume you are running them from the corresponding `Scripts/<source>/` directory, or that your `PYTHONPATH`/paths are set accordingly.

---
### Source: IPTV
**Single instance**
```bash
python build_instance_from_epg_iptv.py \
  --epg-url "https://iptv-epg.org/files/epg-al.xml" \
  --date "2026-01-07" \
  --opening 0 \
  --closing 1440 \
  --out instance_all_channels_2026-01-04.json
```

**Batch generation**
```bash
python batch_generate_instances_epg_iptv.py
```

### Source: PW (EPG.PW)

**Single instance**
```bash
python build_instance_from_epg_pw.py \
  --epg-url "https://epg.pw/xmltv/epg_AU.xml.gz" \
  --date "2026-01-07" \
  --opening 0 \
  --closing 1440 \
  --out epg_AU_2026-01-04.json
```

**Batch generation**
```bash
python batch_generate_instances_epg_pw.py \
  --source epgpw \
  --dates 2026-01-07 \
  --resume
```

### Source: YouTube
**Single instance**
```bash
python build_instance_from_youtube.py \
  --api-key "YOUR_YOUTUBE_API_KEY" \
  --start-utc "2026-01-07T00:00:00Z" \
  --opening 0 \
  --closing 2880 \
  --channels 2000 \
  --region US \
  --lang en \
  --out instance_youtube_2026-01-04_3days.json
```

**Batch generation**
```bash
python batch_generate_instances_youtube.py \
  --api-key "YOUR_YOUTUBE_API_KEY" \
  --out-root "../../Instances/Instances_Youtube" \
  --resume \
  --generator build_instance_from_youtube.py
```
---

## Citation

If you use this dataset in academic or applied research, please cite it as described in the
`CITATION.cff` file included in this repository.

Proper citation helps ensure reproducibility and supports continued dataset maintenance.

---
## License

### Dataset

The dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
You are free to use, share, and adapt the data for research and educational purposes, provided that
appropriate credit is given.

### Code

All scripts and source code in this repository are released under the **MIT License**.

See the `LICENSE` file for full license texts and terms.

---
## Contributing

Contributions are welcome, including new instances, additional data sources,
improvements to generation or validation scripts, and documentation enhancements.

Please ensure that any contributed data conforms to the existing instance schema
and passes the provided validation tools before submitting a pull request.
