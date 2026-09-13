import re

LETTER_INDEX = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
}

# Semitone offsets for degrees 1–7 in a major scale.
MAJOR_SCALE_OFFSETS = [0, 2, 4, 5, 7, 9, 11]


def pitch_class(letter: str, accidental: str) -> int:
    natural_pcs = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }
    return (natural_pcs[letter] + accidental.count("#") - accidental.count("b")) % 12


def pitch_bass_to_scale_degree(chord_label: str) -> str:
    """
    Examples:
        C:maj/E       -> C:maj/3
        A:min/C       -> A:min/b3
        C:aug/G#      -> C:aug/#5
        C:aug/Ab      -> C:aug/b6
        F#:maj/A#     -> F#:maj/3
    """
    root_match = re.match(r"^([A-G])([#b]*)", chord_label)
    bass_match = re.search(r"/([A-G])([#b]*)$", chord_label)

    if not root_match or not bass_match:
        raise ValueError(f"Invalid chord label: {chord_label!r}")

    root_letter, root_accidental = root_match.groups()
    bass_letter, bass_accidental = bass_match.groups()

    root_letter = root_letter.upper()
    bass_letter = bass_letter.upper()

    # Diatonic degree determined by the note letters.
    degree = (LETTER_INDEX[bass_letter] - LETTER_INDEX[root_letter]) % 7 + 1

    root_pc = pitch_class(root_letter, root_accidental)
    bass_pc = pitch_class(bass_letter, bass_accidental)

    expected_pc = (root_pc + MAJOR_SCALE_OFFSETS[degree - 1]) % 12

    difference = (bass_pc - expected_pc) % 12

    if difference in [1, -11]:
        accidental = "#"
    elif difference in [-1, 11]:
        accidental = "b"
    elif difference == 0:
        accidental = ""
    else:
        raise ValueError(
            f"Too many accidentals: {chord_label}, would give {difference} accidentals"
        )

    return chord_label[: bass_match.start()] + f"/{accidental}{degree}"
