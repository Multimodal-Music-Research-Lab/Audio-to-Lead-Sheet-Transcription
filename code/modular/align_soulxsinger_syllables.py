"""Timestamp-only alignment of SoulX-Singer syllables to unshifted GT notes."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

REST_TOKEN = "<SP>"
UNASSIGNED_TOKEN = "<UNASSIGNED>"
PREFIXED_KEY = re.compile(r"^\d+_(?P<video_id>.+)$")
GAP_COST = 0.7
STRESS_RANK = {"primary": 3, "secondary": 2, "unstressed": 1, "unknown": 0}


def _syllable_stress(phones: list[str]) -> str:
    if any(phone.endswith("1") for phone in phones):
        return "primary"
    if any(phone.endswith("2") for phone in phones):
        return "secondary"
    return "unstressed" if phones else "unknown"


def _syllabify_word_details(word: str) -> list[tuple[str, list[str], str]]:
    """Return LyricsMatcher spelling syllables with phones and lexical stress."""
    from lyrics_syllabification import (
        split_projected_syllables,
    )
    from lyrics_syllabification import (
        syllabify_word as matcher_syllabify,
    )

    projected_spelling, _, phone_syllables = matcher_syllabify(word)
    syllables = split_projected_syllables(projected_spelling)
    if (
        not syllables
        or len(syllables) != len(phone_syllables)
        or not all(isinstance(syllable, str) and syllable for syllable in syllables)
    ):
        raise ValueError(f"could not produce aligned syllables for {word!r}")
    return [
        (text, list(phones), _syllable_stress(list(phones)))
        for text, phones in zip(syllables, phone_syllables)
    ]


def syllabify_word(word: str) -> list[str]:
    """Return LyricsMatcher's orthographic syllables, never its phonemes."""
    return [text for text, _, _ in _syllabify_word_details(word)]


def build_intervals(rows: list[list[Any]], label: str) -> list[dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError(f"{label}: fewer than two usable onsets")
    onsets = [float(row[0]) for row in rows if isinstance(row, list) and len(row) >= 2]
    if len(onsets) != len(rows) or not all(math.isfinite(onset) for onset in onsets):
        raise ValueError(f"{label}: malformed or non-finite onset")
    gaps = [later - earlier for earlier, later in zip(onsets, onsets[1:])]
    if any(gap < 0.01 for gap in gaps):
        raise ValueError(f"{label}: onsets must increase by at least 10 ms")
    final_duration = median(gaps)
    return [
        {
            "onset": onset,
            "end": onsets[index + 1]
            if index + 1 < len(onsets)
            else onset + final_duration,
            "token": row[1],
        }
        for index, (onset, row) in enumerate(zip(onsets, rows))
    ]


def project_syllables(rows: list[list[Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    phrase_index = 0
    phrase_syllable_index = 0
    for word_index, row in enumerate(rows):
        onset, word = float(row[0]), str(row[1])
        if word == REST_TOKEN:
            phrase_index += 1
            phrase_syllable_index = 0
            continue
        syllables = _syllabify_word_details(word)
        end = (
            float(rows[word_index + 1][0]) if word_index + 1 < len(rows) else onset + 1
        )
        duration = (end - onset) / len(syllables)
        for syllable_index, (text, phones, stress) in enumerate(syllables):
            syllable_onset = onset + syllable_index * duration
            projected.append(
                {
                    "word_index": word_index,
                    "syllable_index": syllable_index,
                    "phrase_index": phrase_index,
                    "phrase_syllable_index": phrase_syllable_index,
                    "word": word,
                    "text": text,
                    "phones": phones,
                    "stress": stress,
                    "onset": syllable_onset,
                    "end": syllable_onset + duration,
                }
            )
            phrase_syllable_index += 1
    return projected


def overlap_duration(first: dict[str, Any], second: dict[str, Any]) -> float:
    return max(
        0.0, min(first["end"], second["end"]) - max(first["onset"], second["onset"])
    )


def _pair_cost(
    source: dict[str, Any], reference: dict[str, Any]
) -> tuple[float, float, float]:
    """Return a text-independent temporal cost, onset error, and overlap."""
    onset_error = abs(source["onset"] - reference["onset"])
    source_centre = (source["onset"] + source["end"]) / 2
    reference_centre = (reference["onset"] + reference["end"]) / 2
    centre_error = abs(source_centre - reference_centre)
    overlap = overlap_duration(source, reference)
    return onset_error + 0.25 * centre_error - 0.25 * overlap, onset_error, overlap


def _extension_cost(source: dict[str, Any], reference: dict[str, Any]) -> float | None:
    """Score a continuation note that remains inside one syllable interval."""
    overlap = overlap_duration(source, reference)
    if overlap <= 0:
        return None
    duration = max(reference["end"] - reference["onset"], 0.01)
    stress_rank = STRESS_RANK[source.get("stress", "unknown")]
    # A continuation is cheaper than discarding a temporally covered note.
    # Primary/secondary syllables win competing otherwise-tied extensions.
    return 0.30 - 0.06 * stress_rank - 0.20 * min(1.0, overlap / duration)


def _timestamp_alignment(
    syllables: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    source_onset_tolerance: float = 0.0,
) -> tuple[list[list[tuple[int, int]]], float, float]:
    """Globally align intervals, allowing a syllable to continue over notes."""
    count_source, count_notes = len(syllables), len(notes)
    costs = [
        [(math.inf, math.inf, math.inf, "") for _ in range(count_notes + 1)]
        for _ in range(count_source + 1)
    ]
    costs[0][0] = (0.0, 0.0, 0, "")
    for source_index in range(1, count_source + 1):
        previous = costs[source_index - 1][0]
        costs[source_index][0] = (
            previous[0] + GAP_COST,
            previous[1],
            previous[2] + 1,
            "source_gap",
        )
    for note_index in range(1, count_notes + 1):
        previous = costs[0][note_index - 1]
        costs[0][note_index] = (
            previous[0] + GAP_COST,
            previous[1],
            previous[2] + 1,
            "gt_gap",
        )
    for source_index in range(1, count_source + 1):
        for note_index in range(1, count_notes + 1):
            source = syllables[source_index - 1]
            note = notes[note_index - 1]
            pair_cost, onset_error, _ = _pair_cost(source, note)
            diagonal = costs[source_index - 1][note_index - 1]
            up = costs[source_index - 1][note_index]
            left = costs[source_index][note_index - 1]
            candidates = [
                (up[0] + GAP_COST, up[1], up[2] + 1, "source_gap"),
                (left[0] + GAP_COST, left[1], left[2] + 1, "gt_gap"),
            ]
            if overlap_duration(source, note) > 0 and (
                source.get("phrase_syllable_index", 0) == 0
                or note["onset"] >= source["onset"] - source_onset_tolerance
            ):
                candidates.append(
                    (
                        diagonal[0] + pair_cost,
                        diagonal[1] + onset_error,
                        diagonal[2],
                        "pair",
                    )
                )
            same_phrase = note_index > 1 and notes[note_index - 2].get(
                "phrase_index"
            ) == notes[note_index - 1].get("phrase_index")
            next_source = (
                syllables[source_index] if source_index < count_source else None
            )
            before_next_syllable = (
                next_source is None
                or next_source.get("phrase_index") != source.get("phrase_index")
                or note["end"] <= next_source["onset"]
            )
            if left[3] in {"pair", "extend"} and same_phrase and before_next_syllable:
                extension_cost = _extension_cost(
                    syllables[source_index - 1], notes[note_index - 1]
                )
                if extension_cost is not None:
                    candidates.append(
                        (left[0] + extension_cost, left[1], left[2], "extend")
                    )
            costs[source_index][note_index] = min(candidates, key=lambda item: item[:3])
    pairs: list[tuple[int, int]] = []
    source_index, note_index = count_source, count_notes
    while source_index or note_index:
        move = costs[source_index][note_index][3]
        if move == "pair":
            pairs.append((source_index - 1, note_index - 1))
            source_index -= 1
            note_index -= 1
        elif move == "extend":
            pairs.append((source_index - 1, note_index - 1))
            note_index -= 1
        elif move == "source_gap":
            source_index -= 1
        else:
            note_index -= 1
    pairs.reverse()
    groups: list[list[tuple[int, int]]] = []
    for source_index, note_index in pairs:
        if groups and groups[-1][0][0] == source_index:
            groups[-1].append((source_index, note_index))
        else:
            groups.append([(source_index, note_index)])
    result = costs[count_source][count_notes]
    return groups, result[0], result[1]


def _shifted_syllables(
    syllables: list[dict[str, Any]], shift_seconds: float
) -> list[dict[str, Any]]:
    return [
        {
            **syllable,
            "onset": syllable["onset"] + shift_seconds,
            "end": syllable["end"] + shift_seconds,
        }
        for syllable in syllables
    ]


def _select_shift(
    syllables: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    base_shift: float,
    beat_onsets: list[float] | None,
    auto_shift: bool,
    max_shift_beats: float,
    shift_candidates: int | None = None,
    source_onset_tolerance: float = 0.0,
) -> tuple[float, list[list[tuple[int, int]]], int]:
    if not auto_shift:
        shifted = _shifted_syllables(syllables, base_shift)
        return (
            base_shift,
            _timestamp_alignment(shifted, notes, source_onset_tolerance)[0],
            1,
        )
    if beat_onsets is None:
        raise ValueError("automatic shift requires v2 reference beat_onsets")
    beat_gaps = [
        later - earlier for earlier, later in zip(beat_onsets, beat_onsets[1:])
    ]
    if not beat_gaps or any(gap <= 0 for gap in beat_gaps):
        raise ValueError("automatic shift requires strictly increasing beat_onsets")
    radius = max_shift_beats * median(beat_gaps)
    if shift_candidates is not None:
        if shift_candidates < 1:
            raise ValueError("shift_candidates must be at least 1")
        if shift_candidates == 1:
            candidate_shifts = [base_shift]
        else:
            candidate_shifts = [
                base_shift - radius + (2 * radius * index / (shift_candidates - 1))
                for index in range(shift_candidates)
            ]
    else:
        # Preserve the original default: a 10 ms grid across the beat range.
        candidate_shifts = [
            base_shift + step / 100
            for step in range(-round(radius * 100), round(radius * 100) + 1)
        ]
    best: (
        tuple[tuple[float, float, float], float, list[list[tuple[int, int]]]] | None
    ) = None
    for shift in candidate_shifts:
        pairs, cost, onset_error = _timestamp_alignment(
            _shifted_syllables(syllables, shift), notes, source_onset_tolerance
        )
        candidate = ((cost, onset_error, abs(shift - base_shift)), shift, pairs)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], len(candidate_shifts)


def align_record(
    soulxsinger_rows: list[list[Any]],
    reference_rows: list[list[Any]],
    shift_seconds: float = 0.0,
    beat_onsets: list[float] | None = None,
    auto_shift: bool = False,
    max_shift_beats: float = 1.0,
    shift_candidates: int | None = None,
) -> dict[str, Any]:
    reference_notes = build_intervals(reference_rows, "reference")
    build_intervals(soulxsinger_rows, "soulxsinger")
    syllables = project_syllables(soulxsinger_rows)
    matchable_notes = []
    phrase_index = 0
    for index, note in enumerate(reference_notes):
        if note["token"] == REST_TOKEN:
            phrase_index += 1
            continue
        matchable_notes.append((index, {**note, "phrase_index": phrase_index}))
    selected_shift, groups, tested_shifts = _select_shift(
        syllables,
        [note for _, note in matchable_notes],
        shift_seconds,
        beat_onsets,
        auto_shift,
        max_shift_beats,
        shift_candidates,
    )
    shifted = _shifted_syllables(syllables, selected_shift)
    assignments = []
    for group in groups:
        source_index = group[0][0]
        assigned_note_count = len(group)
        for group_index, (_, compact_note_index) in enumerate(group):
            note_index, note = matchable_notes[compact_note_index]
            pair_cost, onset_error, overlap = _pair_cost(shifted[source_index], note)
            if assigned_note_count == 1:
                assignment_type, rendered_token = (
                    "one_to_one",
                    shifted[source_index]["text"],
                )
            elif group_index == 0:
                assignment_type, rendered_token = (
                    "melisma_start",
                    shifted[source_index]["text"] + "_",
                )
            else:
                assignment_type, rendered_token = "melisma_continuation", "."
            assignments.append(
                {
                    "syllable": shifted[source_index],
                    "note_index": note_index,
                    "type": assignment_type,
                    "assigned_note_count": assigned_note_count,
                    "rendered_token": rendered_token,
                    "overlap": overlap,
                    "onset_error": onset_error,
                    "time_cost": pair_cost
                    if group_index == 0
                    else _extension_cost(shifted[source_index], note),
                }
            )
    assignments_by_note = {
        assignment["note_index"]: assignment for assignment in assignments
    }
    rendered_reference, rendered_aligned = [], []
    for note_index, note in enumerate(reference_notes):
        if note["token"] == REST_TOKEN:
            reference_token = aligned_token = REST_TOKEN
        elif note_index in assignments_by_note:
            reference_token = note["token"]
            aligned_token = assignments_by_note[note_index]["rendered_token"]
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
                assignment["type"] == "melisma_continuation"
                for assignment in assignments
            ),
            "rests": sum(note["token"] == REST_TOKEN for note in reference_notes),
            "unassigned_notes": sum(
                note["token"] != REST_TOKEN and index not in assignments_by_note
                for index, note in enumerate(reference_notes)
            ),
        },
    }


def normalise_key(key: str) -> str:
    match = PREFIXED_KEY.match(key)
    return (match.group("video_id") if match else key).strip("_")


def _reference_payload(
    value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[float]] | None]:
    if value.get("format_version") == 2:
        records, beat_onsets = value.get("records"), value.get("beat_onsets")
        if not isinstance(records, dict) or not isinstance(beat_onsets, dict):
            raise ValueError(
                "v2 reference must contain object-valued records and beat_onsets"
            )
        return records, beat_onsets
    return value, None


def align_data(
    soulxsinger_data: dict[str, Any], reference_data: dict[str, Any], **kwargs: Any
) -> tuple[dict[str, Any], dict[str, str]]:
    reference_records, beat_data = _reference_payload(reference_data)
    normalised_soul = {
        normalise_key(key): rows for key, rows in soulxsinger_data.items()
    }
    normalised_reference = {
        normalise_key(key): rows for key, rows in reference_records.items()
    }
    normalised_beats = {
        normalise_key(key): values for key, values in (beat_data or {}).items()
    }
    results, errors = {}, {}
    for key in sorted(set(normalised_soul) & set(normalised_reference)):
        try:
            results[key] = align_record(
                normalised_soul[key],
                normalised_reference[key],
                kwargs.get("shift_seconds", 0.0),
                normalised_beats.get(key),
                kwargs.get("auto_shift", False),
                kwargs.get("max_shift_beats", 1.0),
                kwargs.get("shift_candidates"),
            )
        except Exception as error:
            errors[key] = str(error)
    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("soulxsinger")
    parser.add_argument("reference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shift-seconds", type=float, default=0.0)
    parser.add_argument(
        "--auto-shift", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-shift-beats", type=float, default=1.0)
    parser.add_argument(
        "--shift-candidates",
        type=int,
        help="number of evenly spaced shifts to test per record; default uses a 10 ms grid",
    )
    arguments = parser.parse_args()
    output_path = Path(arguments.output)
    if output_path.exists() and not arguments.overwrite:
        raise SystemExit(f"output exists; use --overwrite: {output_path}")
    records, errors = align_data(
        json.loads(Path(arguments.soulxsinger).read_text()),
        json.loads(Path(arguments.reference).read_text()),
        shift_seconds=arguments.shift_seconds,
        auto_shift=arguments.auto_shift,
        max_shift_beats=arguments.max_shift_beats,
        shift_candidates=arguments.shift_candidates,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=".align-"
    )
    os.close(descriptor)
    Path(temporary_name).write_text(
        json.dumps(
            {
                "base_shift_seconds": arguments.shift_seconds,
                "records": records,
                "errors": errors,
            },
            indent=2,
        )
        + "\n"
    )
    Path(temporary_name).replace(output_path)
    print(f"aligned {len(records)} records; {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
