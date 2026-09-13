"""Syllabify English spelling from a g2p_en pronunciation.

The pronunciation determines how many syllables a word has.  English spelling
does not provide a one-to-one phone alignment, so character boundaries are
chosen with a small set of productive spelling rules and Pyphen's dictionary
as a soft prior.  This deliberately keeps pronunciation and spelling concerns
separate: callers can inspect the phone syllables when a projected boundary is
unexpected.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import combinations

import inflect
import pyphen
from pronunciations import PronunciationMode, pronunciation

_VOWEL = re.compile(r"^(?:AA|AE|AH|AO|AW|AX|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[012]?$")
_LAX_VOWELS = {"AE", "EH", "IH", "AH", "UH"}
_LEGAL_ONSETS = {
    "b",
    "bl",
    "br",
    "c",
    "ch",
    "cl",
    "cr",
    "d",
    "dr",
    "f",
    "fl",
    "fr",
    "g",
    "gl",
    "gr",
    "h",
    "j",
    "k",
    "kl",
    "kr",
    "l",
    "m",
    "n",
    "p",
    "pl",
    "pr",
    "qu",
    "r",
    "s",
    "sc",
    "scr",
    "sh",
    "sk",
    "sl",
    "sm",
    "sn",
    "sp",
    "spl",
    "spr",
    "st",
    "str",
    "sw",
    "t",
    "th",
    "tr",
    "tw",
    "v",
    "w",
    "wh",
    "wr",
    "y",
    "z",
}
_HYPHENATOR = pyphen.Pyphen(lang="en_US", left=2, right=2)
_NUMBER_TO_WORDS = inflect.engine()
_NUMBER_WORD_RE = re.compile(r"[A-Za-z]+")
_SPELLING_OVERRIDES = {
    ("aphrodite", 4): "aph-ro-di-te",
    ("else's", 2): "el-se's",
    ("everybody's", 4): "eve-ry-bod-y's",
    ("ideal", 2): "i-deal",
    ("laughter", 2): "laugh-ter",
    ("paranoid", 3): "par-a-noid",
    ("pleases", 2): "plea-ses",
    ("remember", 3): "re-mem-ber",
    ("slavery", 2): "slav-ery",
    ("slavery", 3): "slav-er-y",
    ("swingset", 2): "swing-set",
    ("telephone", 3): "tel-e-phone",
}


def _is_vowel(phone: str) -> bool:
    return bool(_VOWEL.fullmatch(phone))


def _phone_base(phone: str) -> str:
    return phone.rstrip("012")


def phone_syllable_spans(phones: Iterable[str]) -> list[tuple[int, int]]:
    """Return half-open ARPABET phone spans grouped by vowel nucleus."""
    phone_list = list(phones)
    nuclei = [index for index, phone in enumerate(phone_list) if _is_vowel(phone)]
    if not nuclei:
        return [(0, len(phone_list))] if phone_list else []

    starts = [0]
    for left, right in zip(nuclei, nuclei[1:]):
        consonants = phone_list[left + 1 : right]
        onset_size = 0
        for size in range(min(3, len(consonants)), 0, -1):
            onset = "".join(_phone_base(phone).lower() for phone in consonants[-size:])
            if onset in _LEGAL_ONSETS:
                onset_size = size
                break
        if (
            len(consonants) == 1
            and _phone_base(phone_list[left]) in _LAX_VOWELS
            and phone_list[right] != "IY0"
        ):
            onset_size = 0
        starts.append(right - onset_size)
    starts.append(len(phone_list))
    return list(zip(starts, starts[1:]))


def syllabify_phones(phones: Iterable[str]) -> list[list[str]]:
    """Group ARPABET phones into syllables using a maximal-onset rule.

    This produces phonetic syllables only.  Orthographic conventions may merge
    reduced vowels differently, which is why :func:`syllabify_word` performs a
    separate spelling projection.
    """
    phone_list = list(phones)
    return [phone_list[start:end] for start, end in phone_syllable_spans(phone_list)]


def _hyphen_positions(word: str) -> set[int]:
    inserted = _HYPHENATOR.inserted(word.lower())
    positions: set[int] = set()
    character_index = 0
    for character in inserted:
        if character == "-":
            positions.add(character_index)
        else:
            character_index += 1
    return positions


def _boundary_scores(word: str, phones: list[str]) -> dict[int, int]:
    """Return spelling-boundary preferences; positions are before a character."""
    lower = word.lower()
    scores = {position: -12 for position in range(1, len(word))}
    for position in _hyphen_positions(word):
        scores[position] += 20

    for position in range(1, len(lower)):
        before, after = lower[position - 1], lower[position]
        if before in "aeiouy" and after not in "aeiouy":
            scores[position] += 5
        elif before not in "aeiouy" and after not in "aeiouy":
            scores[position] += 2
        if lower[position - 1 : position + 1] in {
            "bb",
            "cc",
            "dd",
            "ff",
            "gg",
            "ll",
            "mm",
            "nn",
            "pp",
            "rr",
            "ss",
            "tt",
        }:
            scores[position] += 5

    # Do not sever an initial consonant from its vowel merely because a word
    # lacks a Pyphen entry (telephone, survivor, and similar borrowings).
    if lower[0] not in "aeiouy" and lower[1] in "aeiouy":
        scores[1] -= 30

    # An initial reduced vowel is normally a prefix syllable: a-bout, a-way.
    if (
        len(phones) > 1
        and phones[0] in {"AH0", "ER0"}
        and lower.startswith("a")
        and len(word) > 2
    ):
        scores[1] += 28

    # Productive suffixes have reliable orthographic boundaries, including in
    # compounds and forms absent from a hyphenation dictionary.
    for suffix in (
        "ingly",
        "ness",
        "ment",
        "tion",
        "sion",
        "able",
        "ible",
        "less",
        "ship",
    ):
        if lower.endswith(suffix) and len(lower) > len(suffix):
            scores[len(lower) - len(suffix)] += 12
    for suffix in ("ing",):
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            scores[len(lower) - len(suffix)] += 12
    for suffix in ("ly", "er"):
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            scores[len(lower) - len(suffix)] += 24
    for suffix in ("ic", "or", "ute"):
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            scores[len(lower) - len(suffix)] += 12
    if lower.endswith("y") and len(lower) > 2:
        scores[len(lower) - 1] += 18
    if lower.endswith("y's") and len(lower) > 3:
        scores[len(lower) - 3] += 18

    # The pronoun stems are productive in compounds (any-body, every-thing),
    # but their spelling boundary is absent from dictionary hyphenation.
    if lower.startswith("any"):
        scores[2] += 24
    if lower.startswith("every"):
        scores[3] += 24

    # Elision retains the preceding spelling stem: ev'-ry, prob'-bly.
    apostrophe = lower.find("'")
    if 0 < apostrophe < len(lower) - 1:
        scores[apostrophe + 1] += 24

    # Prefer hiatus boundaries unless it is a common single-vowel grapheme.
    for position in range(1, len(lower)):
        if lower[position - 1] in "aeiouy" and lower[position] in "aeiouy":
            pair = lower[position - 1 : position + 1]
            if pair not in {
                "ai",
                "au",
                "ay",
                "ea",
                "ee",
                "ei",
                "eu",
                "ie",
                "oa",
                "oe",
                "oi",
                "oo",
                "ou",
                "ow",
                "oy",
                "ue",
                "ui",
            }:
                scores[position] += 4
    return scores


def project_syllable_count(
    word: str, syllable_count: int, phones: list[str] | None = None
) -> str:
    """Project a known syllable count onto a word's spelling."""
    # Existing lexical hyphens are explicit user spelling and must never turn
    # into a doubled delimiter during projection.
    if "-" in word:
        return word
    phones = phones or []
    boundary_count = max(0, syllable_count - 1)
    if boundary_count == 0 or len(word) < 2:
        return word
    override = _SPELLING_OVERRIDES.get((word.casefold(), syllable_count))
    if override is not None:
        return override
    scores = _boundary_scores(word, phones)
    candidates = range(1, len(word))
    choices = combinations(candidates, boundary_count)
    vowel_positions = {
        index for index, character in enumerate(word.lower()) if character in "aeiouy"
    }
    valid_choices = []
    for positions in choices:
        starts = (0,) + positions
        ends = positions + (len(word),)
        if all(
            any(start <= vowel < end for vowel in vowel_positions)
            for start, end in zip(starts, ends)
        ):
            valid_choices.append(positions)
    if valid_choices:
        positions = max(
            valid_choices,
            key=lambda choice: (
                sum(scores[position] for position in choice),
                tuple(-position for position in choice),
            ),
        )
    else:
        positions = tuple(
            sorted(scores, key=lambda position: (-scores[position], position))[
                :boundary_count
            ]
        )
    return "".join(
        ("-" if index in positions else "") + character
        for index, character in enumerate(word)
    )


def split_projected_syllables(spelling: str) -> list[str]:
    """Split a projection while retaining boundary hyphens on both sides."""
    parts = spelling.split("-")
    if len(parts) == 1:
        return parts
    return [
        ("-" if index else "") + part + ("-" if index < len(parts) - 1 else "")
        for index, part in enumerate(parts)
    ]


def _syllabify_simple_word(
    word: str, mode: PronunciationMode
) -> tuple[str, list[str], list[list[str]]]:
    """Syllabify one spelling unit without numeric or compound structure."""
    phones = pronunciation(word, mode=mode)
    phone_syllables = syllabify_phones(phones)
    return (
        project_syllable_count(word, len(phone_syllables), phones),
        phones,
        phone_syllables,
    )


def _combine_syllabified_words(
    values: Iterable[tuple[str, list[str], list[list[str]]]],
) -> tuple[str, list[str], list[list[str]]]:
    """Join independently syllabified spelling units into one alignment token."""
    spellings: list[str] = []
    phones: list[str] = []
    phone_syllables: list[list[str]] = []
    for spelling, value_phones, value_syllables in values:
        spellings.append(spelling)
        phones.extend(value_phones)
        phone_syllables.extend(value_syllables)
    return "-".join(spellings), phones, phone_syllables


def syllabify_word(
    word: str, *, mode: PronunciationMode = "canonical"
) -> tuple[str, list[str], list[list[str]]]:
    """Return orthographic syllables, ARPABET phones, and phone syllables.

    Digit-only tokens are expanded with g2p_en's ``inflect`` dependency. Each
    hyphen-delimited component is then processed independently, which also
    makes explicit letter sequences such as ``A-M-A-N`` individual syllables.
    """
    if word.isdigit():
        number_words = _NUMBER_WORD_RE.findall(
            _NUMBER_TO_WORDS.number_to_words(int(word))
        )
        return _combine_syllabified_words(
            syllabify_word(number_word, mode=mode) for number_word in number_words
        )
    if "-" in word:
        components = word.split("-")
        if all(components):
            return _combine_syllabified_words(
                syllabify_word(component, mode=mode) for component in components
            )
    return _syllabify_simple_word(word, mode)
