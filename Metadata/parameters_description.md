# Parameters Description

This document provides a **formal and detailed explanation of all parameters** used across the instance generators in the *Smart TV Scheduling Dataset*.  
The goal is to ensure **clarity, reproducibility, and correct interpretation** of the generated instances when used in scheduling, optimization, and benchmarking experiments.

All parameters are consistent across data sources (IPTV EPG, EPG.PW, YouTube) unless explicitly stated otherwise.

---

## 1. Core Instance Parameters (Common to All Sources)

These parameters appear **inside every generated instance JSON file** and define the scheduling problem.

### `opening_time`
- **Type:** Integer (minutes)
- **Description:**  
  Start of the scheduling horizon, expressed in minutes from the beginning of the instance timeline.
- **Typical values:**  
  - `0` → start at 00:00  
  - `480` → start at 08:00
- **Usage:**  
  Programs starting before this time are clipped or discarded.

---

### `closing_time`
- **Type:** Integer (minutes)
- **Description:**  
  End of the scheduling horizon (exclusive).
- **Typical values:**  
  - `1440` → one-day instance  
  - `2880` → two-day instance  
  - `days × 1440` → multi-day instance
- **Constraint:**  
  Must be strictly greater than `opening_time`.

---

### `min_duration`
- **Type:** Integer (minutes)
- **Description:**  
  Minimum duration allowed for a program segment.
- **Purpose:**  
  Filters out very short items (e.g., promos or fillers) that are not meaningful for scheduling.
- **Typical values:** `10–30`

---

### `max_consecutive_genre`
- **Type:** Integer
- **Description:**  
  Maximum number of consecutive scheduled programs with the same genre.
- **Interpretation:**  
  - Can be treated as a **hard constraint** (disallow sequences longer than this)
  - Or as a **soft constraint** (penalize excessive repetition)
- **Typical values:** `1–4`

---

### `channels_count`
- **Type:** Integer
- **Description:**  
  Total number of channels included in the instance.
- **Note:**  
  Should always match `channels.length`.

---

### `switch_penalty`
- **Type:** Integer
- **Description:**  
  Penalty applied when switching from one channel to another between consecutive scheduled segments.
- **Purpose:**  
  Models user dissatisfaction or operational costs related to frequent channel switching.
- **Typical values:** `5–50`

---

### `termination_penalty`
- **Type:** Integer
- **Description:**  
  Penalty applied if the schedule terminates before reaching `closing_time`.
- **Purpose:**  
  Encourages full utilization of the scheduling horizon.
- **Typical values:** `10–100`

---

## 2. Structural Constraint Parameters

### `priority_blocks`
- **Type:** Array of objects
- **Description:**  
  Defines **hard time windows** where only a subset of channels may be scheduled.
- **Fields:**
  - `start` – start time (minutes)
  - `end` – end time (minutes)
  - `allowed_channels` – list of allowed `channel_id`s
- **Usage:**  
  Models contractual obligations, premium content windows, or regulatory constraints.

---

### `time_preferences`
- **Type:** Array of objects
- **Description:**  
  Defines **soft preferences** over time intervals.
- **Fields:**
  - `start`, `end` – preference window
  - `preferred_genre` – genre favored in that window
  - `bonus` – additive utility when satisfied
- **Purpose:**  
  Encodes viewer behavior patterns (e.g., news in the morning, movies in prime time).

---

## 3. Channel-Level Parameters

### `channel_id`
- **Type:** Integer
- **Description:**  
  Numeric identifier for a channel (typically `0 … channels_count−1`).

---

### `channel_name`
- **Type:** String
- **Description:**  
  Human-readable channel label.

---

### `programs`
- **Type:** Array
- **Description:**  
  List of all programs available on this channel during the scheduling horizon.

---

## 4. Program-Level Parameters

### `program_id`
- **Type:** String
- **Description:**  
  Unique identifier for a program within a channel.
- **Construction:**  
  Typically derived from channel, title, date, and start time.

---

### `start`
- **Type:** Integer (minutes)
- **Description:**  
  Program start time (inclusive).

---

### `end`
- **Type:** Integer (minutes)
- **Description:**  
  Program end time (exclusive).
- **Constraint:**  
  Must satisfy `end > start`.

---

### `genre`
- **Type:** String
- **Description:**  
  Program genre label.
- **Examples:**  
  `news`, `sports`, `movie`, `series`, `documentary`, `kids`, `music`, `talk`, `entertainment`

---

### `score`
- **Type:** Integer (`0–100`)
- **Description:**  
  Program quality or utility score.
- **Derivation:**  
  Deterministically computed from:
  - Genre base score
  - Program duration
  - Keyword-based relevance
  - Time-of-day weighting
- **Purpose:**  
  Used by solvers as part of the objective function.