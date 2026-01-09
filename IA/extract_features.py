#!/usr/bin/env python3
"""
Feature extraction script for Smart TV Scheduling problem instances.

Extracts 15 features from each instance JSON file and outputs a normalized CSV.

Usage:
    python extract_features.py <instances_folder> <output_csv_path>

Example:
    python extract_features.py ./instances ./features.csv
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_instance(filepath: str) -> dict[str, Any]:
    """Load a JSON instance file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_all_programs(instance: dict) -> list[dict]:
    """Extract all programs from all channels."""
    programs = []
    for channel in instance.get('channels', []):
        programs.extend(channel.get('programs', []))
    return programs


def compute_avg_concurrent_choices(instance: dict, programs: list[dict], sample_points: int = 100) -> float:
    """
    Compute average number of programs available at sampled time points.
    Samples uniformly across the scheduling window.
    """
    opening = instance['opening_time']
    closing = instance['closing_time']
    time_window = closing - opening

    if time_window <= 0 or not programs:
        return 0.0

    # Sample time points
    step = max(1, time_window // sample_points)
    sample_times = range(opening, closing, step)

    concurrent_counts = []
    for t in sample_times:
        count = sum(1 for p in programs if p['start'] <= t < p['end'])
        concurrent_counts.append(count)

    return statistics.mean(concurrent_counts) if concurrent_counts else 0.0


def compute_priority_block_coverage(instance: dict) -> float:
    """
    Compute fraction of time window covered by priority blocks.
    Handles overlapping blocks by merging intervals.
    """
    opening = instance['opening_time']
    closing = instance['closing_time']
    time_window = closing - opening

    if time_window <= 0:
        return 0.0

    priority_blocks = instance.get('priority_blocks', [])
    if not priority_blocks:
        return 0.0

    # Clip blocks to scheduling window and collect intervals
    intervals = []
    for block in priority_blocks:
        start = max(block['start'], opening)
        end = min(block['end'], closing)
        if start < end:
            intervals.append((start, end))

    if not intervals:
        return 0.0

    # Merge overlapping intervals
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Calculate total covered time
    covered = sum(end - start for start, end in merged)
    return covered / time_window


def compute_bonus_density(instance: dict) -> float:
    """Compute total bonus potential per minute of time window."""
    opening = instance['opening_time']
    closing = instance['closing_time']
    time_window = closing - opening

    if time_window <= 0:
        return 0.0

    time_preferences = instance.get('time_preferences', [])
    total_bonus = sum(pref.get('bonus', 0) for pref in time_preferences)

    return total_bonus / time_window


def extract_features(instance: dict) -> dict[str, float]:
    """
    Extract all 15 features from an instance.

    Returns a dictionary with feature names as keys.
    """
    # Get all programs
    programs = get_all_programs(instance)

    # Basic parameters
    opening = instance['opening_time']
    closing = instance['closing_time']
    time_window = closing - opening
    num_channels = instance['channels_count']

    # Program durations
    durations = [p['end'] - p['start'] for p in programs]
    avg_duration = statistics.mean(durations) if durations else 0
    std_duration = statistics.stdev(durations) if len(durations) > 1 else 0

    # Popularity scores
    scores = [p['score'] for p in programs]
    avg_score = statistics.mean(scores) if scores else 0

    # Unique genres
    genres = set(p['genre'] for p in programs)

    # Program density
    total_program_minutes = sum(durations)
    program_density = total_program_minutes / (time_window * num_channels) if (time_window * num_channels) > 0 else 0

    # Min duration ratio
    min_duration_ratio = instance['min_duration'] / avg_duration if avg_duration > 0 else 0

    features = {
        # Scale features (1-3)
        'num_channels': float(num_channels),
        'total_programs': float(len(programs)),
        'time_window_minutes': float(time_window),

        # Constraint parameters (4-7)
        'min_duration_ratio': float(min_duration_ratio),
        'max_consecutive_genre': float(instance['max_consecutive_genre']),
        'switch_penalty': float(instance['switch_penalty']),
        'termination_penalty': float(instance['termination_penalty']),

        # Program characteristics (8-11)
        'avg_program_duration': float(avg_duration),
        'std_program_duration': float(std_duration),
        'avg_popularity_score': float(avg_score),
        'num_unique_genres': float(len(genres)),

        # Density & competition features (12-13)
        'program_density': float(program_density),
        'avg_concurrent_choices': float(compute_avg_concurrent_choices(instance, programs)),

        # Bonus & restriction features (14-15)
        'priority_block_coverage': float(compute_priority_block_coverage(instance)),
        'bonus_density': float(compute_bonus_density(instance)),
    }

    return features


def normalize_features(all_features: list[dict[str, float]]) -> list[dict[str, float]]:
    """
    Normalize features to [0, 1] range by dividing by maximum value.

    Features already in [0, 1] (like program_density, priority_block_coverage)
    are still normalized by their max to maintain consistency.
    """
    if not all_features:
        return all_features

    # Get all feature names
    feature_names = list(all_features[0].keys())

    # Find max for each feature
    max_values = {}
    for name in feature_names:
        values = [f[name] for f in all_features]
        max_val = max(values) if values else 1.0
        max_values[name] = max_val if max_val != 0 else 1.0  # Avoid division by zero

    # Normalize
    normalized = []
    for features in all_features:
        norm_features = {
            name: features[name] / max_values[name]
            for name in feature_names
        }
        normalized.append(norm_features)

    return normalized


def process_instances(folder_path: str) -> tuple[list[str], list[dict[str, float]]]:
    """
    Process all JSON instance files in a folder.

    Returns tuple of (instance_names, features_list).
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    # Find all JSON files
    json_files = sorted(folder.glob('*.json'))

    if not json_files:
        raise ValueError(f"No JSON files found in: {folder_path}")

    instance_names = []
    all_features = []

    for json_file in json_files:
        try:
            instance = load_instance(json_file)
            features = extract_features(instance)
            instance_names.append(json_file.stem)  # filename without extension
            all_features.append(features)
            print(f"Processed: {json_file.name}")
        except Exception as e:
            print(f"Warning: Failed to process {json_file.name}: {e}", file=sys.stderr)

    if not all_features:
        raise ValueError("No instances were successfully processed")

    return instance_names, all_features


def save_to_csv(
        instance_names: list[str],
        features: list[dict[str, float]],
        output_path: str
) -> None:
    """Save features to CSV file."""
    if not features:
        raise ValueError("No features to save")

    # Get feature names in consistent order
    feature_names = list(features[0].keys())

    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['instance_name'] + feature_names)

        # Data rows
        for name, feat in zip(instance_names, features):
            row = [name] + [feat[fn] for fn in feature_names]
            writer.writerow(row)

    print(f"\nSaved {len(features)} instances to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from Smart TV Scheduling problem instances.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features extracted (15 total):
  Scale:
    1. num_channels          - Number of TV channels
    2. total_programs        - Total programs across all channels
    3. time_window_minutes   - Scheduling window duration (E - O)

  Constraints:
    4. min_duration_ratio    - min_duration / avg_program_duration
    5. max_consecutive_genre - Max consecutive same-genre programs
    6. switch_penalty        - Channel switch penalty
    7. termination_penalty   - Early termination penalty

  Program characteristics:
    8. avg_program_duration  - Mean program length in minutes
    9. std_program_duration  - Std dev of program durations
    10. avg_popularity_score - Mean popularity score
    11. num_unique_genres    - Count of distinct genres

  Density & competition:
    12. program_density        - Total program minutes / (window × channels)
    13. avg_concurrent_choices - Avg programs available at any time

  Bonus & restrictions:
    14. priority_block_coverage - % of time window with priority blocks
    15. bonus_density           - Total bonus potential / time_window
        """
    )

    parser.add_argument(
        'instances_folder',
        type=str,
        help='Path to folder containing instance JSON files'
    )

    parser.add_argument(
        'output_csv',
        type=str,
        help='Path for output CSV file'
    )

    args = parser.parse_args()

    try:
        # Process all instances
        print(f"Processing instances from: {args.instances_folder}\n")
        instance_names, raw_features = process_instances(args.instances_folder)

        # Normalize features
        print(f"\nNormalizing {len(raw_features)} instances...")
        normalized_features = normalize_features(raw_features)

        # Save to CSV
        save_to_csv(instance_names, normalized_features, args.output_csv)

        print("\nFeature extraction complete!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
