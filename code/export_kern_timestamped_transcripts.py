"""Export beat-aligned lyric cells from Kern transcripts."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REST_TOKEN = "<SP>"
FORMAT_VERSION = 2


def _quarter_duration(kern_cell: str, path: Path) -> float:
    match = re.match(r"(\d+)(\.*)", kern_cell)
    if match is None:
        raise ValueError(f"{path}: Kern cell has no reciprocal duration: {kern_cell!r}")
    reciprocal = int(match.group(1))
    if reciprocal <= 0:
        raise ValueError(f"{path}: invalid reciprocal duration in {kern_cell!r}")
    duration = 4.0 / reciprocal
    dots = len(match.group(2))
    duration *= sum(0.5**index for index in range(dots + 1))
    return duration


def parse_kern_rows(text: str, path: Path) -> tuple[list[bool], list[float]]:
    """Return rest flags and note starts measured in quarter beats."""
    lines = text.splitlines()
    layout: list[str] | None = None
    kern_index = None
    rows: list[bool] = []
    starts: list[float] = []
    beat_position = 0.0
    for row_number, line in enumerate(lines, start=1):
        if not line:
            continue
        columns = line.split("\t")
        if "**kern" in columns:
            if columns.count("**kern") != 1:
                raise ValueError(f"{path}: expected exactly one **kern spine")
            layout = columns
            kern_index = columns.index("**kern")
            continue
        if layout is None:
            continue
        if line.startswith(("!", "*", "=")):
            continue
        if len(columns) < len(layout):
            raise ValueError(
                f"{path}: row {row_number} has {len(columns)} cells; expected at least {len(layout)}"
            )
        assert kern_index is not None
        kern_cell = columns[kern_index]
        rows.append("r" in kern_cell)
        starts.append(beat_position)
        beat_position += _quarter_duration(kern_cell, path)
    if layout is None or kern_index is None:
        raise ValueError(f"{path}: missing **kern spine declaration")
    return rows, starts


def validate_onsets(value: Any, record_id: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{record_id!r}: beat onsets must be a list of numbers")
    result: list[float] = []
    for onset in value:
        if isinstance(onset, bool) or not isinstance(onset, (int, float)):
            raise ValueError(f"{record_id!r}: beat onsets must contain numbers")
        number = float(onset)
        if not math.isfinite(number) or number < 0:
            raise ValueError(
                f"{record_id!r}: beat onsets must be finite and non-negative"
            )
        result.append(number)
    return result


def interpolate_beat_times(
    beat_times: list[float], beat_positions: list[float], record_id: str
) -> list[float]:
    """Map fractional quarter-beat positions to seconds by piecewise linear interpolation."""
    if not beat_times:
        raise ValueError(f"{record_id!r}: beat timestamp list cannot be empty")
    if any(later <= earlier for earlier, later in zip(beat_times, beat_times[1:])):
        raise ValueError(f"{record_id!r}: beat timestamps must be strictly increasing")
    last_index = len(beat_times) - 1
    if any(position < 0 for position in beat_positions):
        raise ValueError(f"{record_id!r}: Kern note starts cannot be negative")
    # Generated scores may contain trailing instrumental bars not present in
    # the lyric timing export. Continue the established tempo using the median
    # beat interval rather than silently dropping those slots.
    step = beat_times[-1] - beat_times[-2] if len(beat_times) > 1 else 0.5
    result: list[float] = []
    for position in beat_positions:
        lower = min(int(math.floor(position)), last_index)
        if lower >= last_index:
            result.append(beat_times[-1] + (position - last_index) * step)
            continue
        fraction = position - lower
        result.append(
            beat_times[lower] + fraction * (beat_times[lower + 1] - beat_times[lower])
        )
    return result


def build_records(
    items: Iterable[tuple[str, str, Any, list[list[Any]]]],
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, int]]:
    full: dict[str, list[list[float | str]]] = {}
    rests_only: dict[str, list[list[float | str]]] = {}
    beat_onsets: dict[str, list[float]] = {}
    note_count = rest_count = 0
    for record_id, text, onset_values, lyric_rows in items:
        if record_id in full:
            raise ValueError(f"Duplicate record ID: {record_id!r}")
        rest_flags, beat_positions = parse_kern_rows(text, Path(record_id))
        beat_times = validate_onsets(onset_values, record_id)
        if len(beat_times) < 2 or any(
            later <= earlier for earlier, later in zip(beat_times, beat_times[1:])
        ):
            raise ValueError(
                f"{record_id!r}: beat onsets must contain at least two strictly increasing times"
            )
        onsets = interpolate_beat_times(beat_times, beat_positions, record_id)
        if len(lyric_rows) < len(rest_flags):
            lyric_rows = lyric_rows + [[0.0, ""]] * (len(rest_flags) - len(lyric_rows))
        elif len(lyric_rows) > len(rest_flags):
            lyric_rows = lyric_rows[: len(rest_flags)]
        beat_onsets[record_id] = beat_times
        full[record_id] = []
        rests_only[record_id] = []
        for onset, is_rest, lyric_row in zip(onsets, rest_flags, lyric_rows):
            lyric = str(lyric_row[1])
            token = REST_TOKEN if is_rest else lyric
            mask_token = REST_TOKEN if is_rest else ""
            full[record_id].append([onset, token])
            rests_only[record_id].append([onset, mask_token])
            if is_rest:
                rest_count += 1
            else:
                note_count += 1
    return (
        {"format_version": FORMAT_VERSION, "records": full, "beat_onsets": beat_onsets},
        {
            "format_version": FORMAT_VERSION,
            "records": rests_only,
            "beat_onsets": beat_onsets,
        },
        (note_count, rest_count),
    )


def load_annotation_beats(path: Path) -> dict[str, list[float]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read annotation JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Annotation JSON must be an object keyed by record ID")
    result: dict[str, list[float]] = {}
    if raw.get("format_version") == 2 and isinstance(raw.get("beat_onsets"), dict):
        return {
            str(key).removesuffix("_prediction").strip("_"): validate_onsets(
                value, str(key)
            )
            for key, value in raw["beat_onsets"].items()
        }
    for record_id, record in raw.items():
        try:
            times = record["alignment"]["refined"]["times"]
        except (KeyError, TypeError):
            raise ValueError(
                f"{record_id!r}: missing alignment.refined.times"
            ) from None
        values = validate_onsets(times, record_id)
        if values:
            first = values[0]
            values = [value - first for value in values]
        result[record_id.removesuffix("_prediction").strip("_")] = values
    return result


def read_dataset_beats(
    dataset: Any, beat_column: str, id_column: str
) -> dict[str, Any]:
    try:
        from datasets import DatasetDict
    except ImportError as exc:
        raise ValueError("HF dataset support requires the 'datasets' package") from exc
    splits = dataset.values() if isinstance(dataset, DatasetDict) else (dataset,)
    result: dict[str, Any] = {}
    for split in splits:
        for index, record_id in enumerate(split[id_column]):
            if record_id in result:
                raise ValueError(f"Duplicate record ID in dataset: {record_id!r}")
            result[record_id] = split[beat_column][index]
    return result


def write_json_atomic(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(
            f"Output already exists: {path}. Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dataset", type=Path, help="HF Dataset or DatasetDict directory"
    )
    mode.add_argument("--kern-files", nargs="+", help="Kern paths or glob patterns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rest-only-output", type=Path, required=False)
    parser.add_argument(
        "--beat-onsets-json",
        type=Path,
        help="JSON annotation mapping for direct Kern files",
    )
    parser.add_argument(
        "--lyric-reference",
        type=Path,
        required=True,
        help="Existing timestamp JSON supplying lyric tokens",
    )
    parser.add_argument(
        "--beat-dataset",
        type=Path,
        help="HF dataset supplying beats for direct Kern files",
    )
    parser.add_argument("--kern-column", default="transcript")
    parser.add_argument("--beat-column", default="beat_onsets")
    parser.add_argument("--id-column", default="file_name")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_hf_dataset(path: Path) -> Any:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ValueError("HF dataset support requires the 'datasets' package") from exc
    return load_from_disk(str(path.expanduser()))


def main() -> int:
    args = parse_args()
    if args.rest_only_output is None:
        args.rest_only_output = args.output.with_stem(args.output.stem + "-with_rest")
    if (
        args.output.expanduser().resolve()
        == args.rest_only_output.expanduser().resolve()
    ):
        raise SystemExit("--output and --rest-only-output must be different paths")
    if args.dataset and (args.beat_onsets_json or args.beat_dataset):
        raise SystemExit(
            "--beat-onsets-json/--beat-dataset are only valid with --kern-files"
        )
    if args.kern_files and bool(args.beat_onsets_json) == bool(args.beat_dataset):
        raise SystemExit(
            "Direct mode requires exactly one of --beat-onsets-json or --beat-dataset"
        )
    if args.dataset:
        try:
            from datasets import DatasetDict
        except ImportError as exc:
            raise SystemExit(
                "HF dataset support requires the 'datasets' package"
            ) from exc
        dataset = load_hf_dataset(args.dataset)
        splits = dataset.values() if isinstance(dataset, DatasetDict) else (dataset,)
        for split in splits:
            missing = {args.kern_column, args.beat_column, args.id_column} - set(
                split.column_names
            )
            if missing:
                raise SystemExit(
                    f"Dataset is missing columns: {', '.join(sorted(missing))}"
                )
        items = (
            (record_id, split[args.kern_column][index], split[args.beat_column][index])
            for split in splits
            for index, record_id in enumerate(split[args.id_column])
        )
    else:
        paths: list[Path] = []
        for pattern in args.kern_files:
            matches = [Path(item) for item in glob.glob(pattern)]
            paths.extend(matches or [Path(pattern)])
        if not paths or any(not path.is_file() for path in paths):
            raise SystemExit(
                "Every --kern-files input must resolve to an existing file"
            )
        beats = (
            load_annotation_beats(args.beat_onsets_json)
            if args.beat_onsets_json
            else read_dataset_beats(
                load_hf_dataset(args.beat_dataset), args.beat_column, args.id_column
            )
        )
        reference_root = json.loads(args.lyric_reference.read_text(encoding="utf-8"))
        reference = reference_root.get("records", reference_root)
        reference = {
            str(key).removesuffix("_prediction").strip("_"): value
            for key, value in reference.items()
        }
        items = []
        for path in paths:
            record_id = path.stem.removesuffix("_prediction").strip("_")
            if record_id not in beats:
                raise SystemExit(f"No beat onsets found for {record_id!r}")
            if record_id not in reference:
                raise SystemExit(f"No lyric reference found for {record_id!r}")
            items.append(
                (
                    record_id,
                    path.read_text(encoding="utf-8"),
                    beats[record_id],
                    reference[record_id],
                )
            )
    try:
        full, masked, counts = build_records(items)
        write_json_atomic(args.output.expanduser(), full, args.overwrite)
        write_json_atomic(args.rest_only_output.expanduser(), masked, args.overwrite)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Exported {len(full['records'])} records: {counts[0]} notes, {counts[1]} rests"
    )
    print(f"Full output: {args.output}\nRest-only output: {args.rest_only_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
