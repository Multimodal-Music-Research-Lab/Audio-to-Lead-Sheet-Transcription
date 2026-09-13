"""Append modular-aligned lyrics as a ``**text`` spine to two-spine Kern files."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REST_TOKEN = "<SP>"
UNASSIGNED_TOKEN = "<UNASSIGNED>"
PREDICTION_SUFFIX = "_prediction"


def normalise_record_id(value: str) -> str:
    """Match the record-ID convention used by the modular alignment scripts."""
    return value.removesuffix(PREDICTION_SUFFIX).strip("_")


def lyric_cells(record: Any, record_id: str) -> list[str]:
    """Extract rendered alignment cells, replacing non-lyric sentinel values."""
    if not isinstance(record, dict) or not isinstance(record.get("aligned"), list):
        raise ValueError(f"{record_id}: missing list-valued aligned field")
    result = []
    for row_number, row in enumerate(record["aligned"], start=1):
        if not isinstance(row, list) or len(row) < 2 or not isinstance(row[1], str):
            raise ValueError(f"{record_id}: malformed aligned row {row_number}")
        token = row[1]
        # A **text spine needs '.' for rests and notes with no assigned syllable.
        result.append("." if token in {REST_TOKEN, UNASSIGNED_TOKEN, ""} else token)
    return result


def render_with_lyrics(source: str, cells: list[str], path: Path) -> str:
    """Add a third **text spine while preserving the existing Kern rows."""
    output: list[str] = []
    declared = False
    cell_index = 0
    terminated = False

    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line:
            continue
        if line.startswith("!!"):
            output.append(line)
            continue
        columns = line.split("\t")
        if not declared:
            if columns.count("**kern") != 1:
                output.append(line)
                continue
            if len(columns) != 2:
                raise ValueError(
                    f"{path}: declaration line must contain exactly two spines"
                )
            if "**text" in columns:
                raise ValueError(f"{path}: already contains a **text spine")
            output.append("\t".join([*columns, "**text"]))
            declared = True
            continue

        if len(columns) != 2:
            raise ValueError(
                f"{path}: line {line_number} has {len(columns)} cells; expected two"
            )
        first = columns[0]
        if first.startswith("*-"):
            output.append("\t".join([*columns, "*-"]))
            terminated = True
        elif first.startswith("*"):
            output.append("\t".join([*columns, "*"]))
        elif first.startswith("="):
            output.append("\t".join([*columns, "="]))
        elif first.startswith("!"):
            output.append("\t".join([*columns, "!"]))
        else:
            if cell_index == len(cells):
                raise ValueError(
                    f"{path}: more Kern data rows than aligned lyric cells"
                )
            output.append("\t".join([*columns, cells[cell_index]]))
            cell_index += 1

    if not declared:
        raise ValueError(f"{path}: missing **kern declaration")
    if not terminated:
        raise ValueError(f"{path}: missing spine terminator")
    if cell_index != len(cells):
        raise ValueError(
            f"{path}: {len(cells) - cell_index} aligned lyric cells have no Kern data row"
        )
    return "\n".join(output) + "\n"


def write_text_atomic(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(
            f"Output already exists: {path}. Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alignment_json", type=Path)
    parser.add_argument("kern_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pattern", default="*.krn")
    parser.add_argument(
        "--record-id",
        action="append",
        help="render only this alignment record ID; repeatable",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = json.loads(args.alignment_json.read_text(encoding="utf-8"))
    records = root.get("records", root) if isinstance(root, dict) else None
    if not isinstance(records, dict):
        raise SystemExit("alignment JSON must contain an object-valued records field")
    selected = set(args.record_id or records)
    unknown = selected - set(records)
    if unknown:
        raise SystemExit(
            f"record IDs absent from alignment JSON: {', '.join(sorted(unknown))}"
        )

    paths = sorted(args.kern_dir.glob(args.pattern))
    by_exact_id = {path.stem: path for path in paths}
    if len(by_exact_id) != len(paths):
        raise SystemExit("Kern inputs have duplicate exact record IDs")
    by_normalized_id = {}
    for path in paths:
        normalized_id = normalise_record_id(path.stem)
        if normalized_id in by_normalized_id:
            by_normalized_id[normalized_id] = None
        else:
            by_normalized_id[normalized_id] = path
    paths_by_record_id = {}
    for record_id in selected:
        path = by_exact_id.get(
            record_id, by_normalized_id.get(normalise_record_id(record_id))
        )
        if path is not None:
            paths_by_record_id[record_id] = path
    missing = selected - set(paths_by_record_id)
    if missing:
        raise SystemExit(
            f"Kern files absent for aligned IDs: {', '.join(sorted(missing))}"
        )

    for record_id in sorted(selected):
        source_path = paths_by_record_id[record_id]
        output = render_with_lyrics(
            source_path.read_text(encoding="utf-8"),
            lyric_cells(records[record_id], record_id),
            source_path,
        )
        write_text_atomic(args.output_dir / source_path.name, output, args.overwrite)
    print(f"wrote {len(selected)} Kern files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
