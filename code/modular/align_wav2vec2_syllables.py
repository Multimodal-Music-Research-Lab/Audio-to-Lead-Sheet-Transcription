"""Align LyricsMatcher syllables using Wav2Vec2 character timestamps."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from align_soulxsinger_syllables import (
    REST_TOKEN,
    UNASSIGNED_TOKEN,
    _extension_cost,
    _pair_cost,
    _reference_payload,
    _select_shift,
    _shifted_syllables,
    _syllabify_word_details,
    build_intervals,
    normalise_key,
)


def _validate_characters(value: Any, record_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"record {record_id!r} characters must be a non-empty list")
    result = []
    previous_end = -math.inf
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"record {record_id!r} character {index} must be an object"
            )
        char = item.get("char")
        start, end = item.get("start"), item.get("end")
        source_index = item.get("source_index")
        if not isinstance(char, str) or len(char) != 1:
            raise ValueError(f"record {record_id!r} character {index} has invalid char")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise ValueError(
                f"record {record_id!r} character {index} has invalid timestamps"
            )
        start, end = float(start), float(end)
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise ValueError(
                f"record {record_id!r} character {index} timestamps must be finite and increasing"
            )
        if start < previous_end:
            raise ValueError(
                f"record {record_id!r} character timestamps are not ordered"
            )
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index < 0
        ):
            raise ValueError(
                f"record {record_id!r} character {index} has invalid source_index"
            )
        result.append(
            {
                **item,
                "char": char,
                "start": start,
                "end": end,
                "source_index": source_index,
            }
        )
        previous_end = end
    return result


def project_character_syllables(
    record: dict[str, Any], record_id: str = "record"
) -> list[dict[str, Any]]:
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"record {record_id!r} text must be a non-empty string")
    characters = _validate_characters(record.get("characters"), record_id)
    if any(
        later["source_index"] <= earlier["source_index"]
        for earlier, later in zip(characters, characters[1:])
    ):
        raise ValueError(f"record {record_id!r} source_index values are not ordered")
    projected = []
    phrase_index = 0
    phrase_syllable_index = 0
    character_cursor = 0
    for word_match in re.finditer(r"[^\s]+", text):
        raw_word = word_match.group()
        letters = []
        for char in raw_word:
            if not (char.isalpha() or char == "'"):
                continue
            while (
                character_cursor < len(characters)
                and characters[character_cursor]["char"].casefold() != char.casefold()
            ):
                character_cursor += 1
            if character_cursor >= len(characters):
                raise ValueError(
                    f"record {record_id!r} has too few timestamped characters"
                )
            letters.append((character_cursor, char))
            character_cursor += 1
        if not letters:
            continue
        word = "".join(char for _, char in letters)
        spans = []
        for source_index, char in letters:
            span = characters[source_index]
            if span["char"].casefold() != char.casefold():
                raise ValueError(
                    f"record {record_id!r} missing or mismatched timestamp for character position {source_index}"
                )
            spans.append(span)
        details = _syllabify_word_details(word)
        cursor = 0
        for syllable_index, (syllable, phones, stress) in enumerate(details):
            needed = [char for char in syllable if char.isalpha() or char == "'"]
            if not needed:
                raise ValueError(
                    f"record {record_id!r} syllabifier produced empty spelling for {word!r}"
                )
            end_cursor = cursor + len(needed)
            if end_cursor > len(spans):
                raise ValueError(
                    f"record {record_id!r} syllable spelling does not match {word!r}"
                )
            syllable_spans = spans[cursor:end_cursor]
            cursor = end_cursor
            projected.append(
                {
                    "word_index": word_match.start(),
                    "syllable_index": syllable_index,
                    "phrase_index": phrase_index,
                    "phrase_syllable_index": phrase_syllable_index,
                    "word": word,
                    "text": syllable,
                    "phones": phones,
                    "stress": stress,
                    "onset": syllable_spans[0]["start"],
                    "end": syllable_spans[-1]["end"],
                    "character_count": len(syllable_spans),
                    "character_confidence": sum(
                        float(span.get("confidence", 0.0)) for span in syllable_spans
                    )
                    / len(syllable_spans),
                }
            )
            phrase_syllable_index += 1
        if cursor != len(spans):
            raise ValueError(
                f"record {record_id!r} syllable spelling does not consume {word!r}"
            )
    if character_cursor != len(characters):
        raise ValueError(f"record {record_id!r} has extra timestamped characters")
    if not projected:
        raise ValueError(f"record {record_id!r} contains no timestamped syllables")
    return projected


def align_record(
    character_record: dict[str, Any],
    reference_rows: list[list[Any]],
    record_id: str = "record",
    **kwargs: Any,
) -> dict[str, Any]:
    reference_notes = build_intervals(reference_rows, "reference")
    syllables = project_character_syllables(character_record, record_id)
    matchable_notes, phrase_index = [], 0
    for index, note in enumerate(reference_notes):
        if note["token"] == REST_TOKEN:
            phrase_index += 1
        else:
            matchable_notes.append((index, {**note, "phrase_index": phrase_index}))
    notes = [note for _, note in matchable_notes]
    selected_shift, groups, tested_shifts = _select_shift(
        syllables,
        notes,
        kwargs.get("shift_seconds", 0.0),
        kwargs.get("beat_onsets"),
        kwargs.get("auto_shift", False),
        kwargs.get("max_shift_beats", 1.0),
        kwargs.get("shift_candidates"),
        kwargs.get("source_onset_tolerance", 0.10),
    )
    shifted = _shifted_syllables(syllables, selected_shift)
    assignments = []
    for group in groups:
        source_index = group[0][0]
        for group_index, (_, compact_note_index) in enumerate(group):
            note_index, note = matchable_notes[compact_note_index]
            cost, onset_error, overlap = _pair_cost(shifted[source_index], note)
            if len(group) == 1:
                assignment_type, rendered = "one_to_one", shifted[source_index]["text"]
            elif group_index == 0:
                assignment_type, rendered = (
                    "melisma_start",
                    shifted[source_index]["text"] + "_",
                )
            else:
                assignment_type, rendered = "melisma_continuation", "."
            assignments.append(
                {
                    "syllable": shifted[source_index],
                    "note_index": note_index,
                    "type": assignment_type,
                    "assigned_note_count": len(group),
                    "rendered_token": rendered,
                    "overlap": overlap,
                    "onset_error": onset_error,
                    "time_cost": cost
                    if group_index == 0
                    else _extension_cost(shifted[source_index], note),
                }
            )
    by_note = {item["note_index"]: item for item in assignments}
    rendered_reference, rendered_aligned = [], []
    for index, note in enumerate(reference_notes):
        if note["token"] == REST_TOKEN:
            reference_token = aligned_token = REST_TOKEN
        elif index in by_note:
            reference_token, aligned_token = (
                note["token"],
                by_note[index]["rendered_token"],
            )
        else:
            reference_token = aligned_token = UNASSIGNED_TOKEN
        rendered_reference.append([note["onset"], reference_token])
        rendered_aligned.append([note["onset"], aligned_token])
    return {
        "shift_seconds": selected_shift,
        "tested_shifts": tested_shifts,
        "reference": rendered_reference,
        "aligned": rendered_aligned,
        "assignments": assignments,
        "counts": {
            "assigned_notes": len(assignments),
            "assigned_syllables": len(groups),
            "melisma_continuation_notes": sum(
                x["type"] == "melisma_continuation" for x in assignments
            ),
            "rests": sum(x["token"] == REST_TOKEN for x in reference_notes),
            "unassigned_notes": sum(
                x["token"] != REST_TOKEN and i not in by_note
                for i, x in enumerate(reference_notes)
            ),
        },
    }


def align_data(
    timestamp_data: dict[str, Any], reference_data: dict[str, Any], **kwargs: Any
) -> tuple[dict[str, Any], dict[str, str]]:
    records = timestamp_data.get("records", timestamp_data)
    if not isinstance(records, dict):
        raise ValueError("timestamp JSON must contain an object-valued records field")
    reference_records, beat_data = _reference_payload(reference_data)
    normalised_records = {
        normalise_key(str(key)): value for key, value in records.items()
    }
    normalised_reference = {
        normalise_key(str(key)): value for key, value in reference_records.items()
    }
    normalised_beats = {
        normalise_key(str(key)): value for key, value in (beat_data or {}).items()
    }
    results, errors = {}, {}
    for key in sorted(set(normalised_records) & set(normalised_reference)):
        try:
            record_kwargs = dict(kwargs)
            record_kwargs["beat_onsets"] = normalised_beats.get(key)
            results[key] = align_record(
                normalised_records[key],
                normalised_reference[key],
                record_id=key,
                **record_kwargs,
            )
        except Exception as error:
            errors[key] = str(error)
    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timestamps")
    parser.add_argument("reference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shift-seconds", type=float, default=0.0)
    parser.add_argument(
        "--auto-shift", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-shift-beats", type=float, default=1.0)
    parser.add_argument("--shift-candidates", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; use --overwrite: {output}")
    records, errors = align_data(
        json.loads(Path(args.timestamps).read_text()),
        json.loads(Path(args.reference).read_text()),
        shift_seconds=args.shift_seconds,
        auto_shift=args.auto_shift,
        max_shift_beats=args.max_shift_beats,
        shift_candidates=args.shift_candidates,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=".align-")
    os.close(fd)
    Path(temporary).write_text(
        json.dumps(
            {
                "base_shift_seconds": args.shift_seconds,
                "records": records,
                "errors": errors,
            },
            indent=2,
        )
        + "\n"
    )
    Path(temporary).replace(output)
    print(f"aligned {len(records)} records; {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
