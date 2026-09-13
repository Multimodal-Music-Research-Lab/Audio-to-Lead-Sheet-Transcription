"""English pronunciation variants used by lyric syllabification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import cmudict
from g2p_en import G2p

PronunciationMode = Literal["canonical", "reduced"]


@dataclass(frozen=True)
class PronunciationVariant:
    """One ordered ARPABET pronunciation and where it came from."""

    phones: tuple[str, ...]
    source: Literal["cmudict", "curated", "g2p"]


_VOWEL = re.compile(r"^(?:AA|AE|AH|AO|AW|AX|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[012]?$")
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})

# AHD-based choices where CMUdict's first entry is not the canonical lyric
# reading used by this project.
_CANONICAL_PRONUNCIATIONS = {
    "every": ("EH1", "V", "R", "IY0"),
    "learned": ("L", "ER1", "N", "D"),
    "our": ("AW1", "R"),
    "fyi": ("EH1", "F", "W", "AY1", "AY1"),
}

# Pronunciations which must supplement CMUdict, including initialisms that
# CMUdict expands into their full phrase.
_CURATED_PRONUNCIATIONS = {
    "fyi": ("EH1", "F", "W", "AY1", "AY1"),
}

# Reductions not supplied by CMUdict. Keep this deliberately reviewed rather
# than applying schwa deletion to every superficially similar word.
_CURATED_REDUCTIONS = {
    "slavery": ("S", "L", "EY1", "V", "R", "IY0"),
}


def _normalize_word(word: str) -> str:
    return word.translate(_APOSTROPHES).casefold()


def vowel_count(phones: tuple[str, ...] | list[str]) -> int:
    """Return the number of vowel nuclei in an ARPABET pronunciation."""
    return sum(bool(_VOWEL.fullmatch(phone.upper())) for phone in phones)


@lru_cache(maxsize=1)
def _cmudict() -> dict[str, list[list[str]]]:
    return cmudict.dict()


@lru_cache(maxsize=1)
def _g2p() -> G2p:
    return G2p()


@lru_cache(maxsize=4096)
def pronunciation_variants(word: str) -> tuple[PronunciationVariant, ...]:
    """Return all known pronunciations for one word in deterministic order."""
    normalized = _normalize_word(word)
    variants: list[PronunciationVariant] = []
    seen: set[tuple[str, ...]] = set()

    for phones in _cmudict().get(normalized, []):
        value = tuple(phones)
        if value not in seen:
            variants.append(PronunciationVariant(value, "cmudict"))
            seen.add(value)

    curated_pronunciation = _CURATED_PRONUNCIATIONS.get(normalized)
    if curated_pronunciation is not None and curated_pronunciation not in seen:
        variants.append(PronunciationVariant(curated_pronunciation, "curated"))
        seen.add(curated_pronunciation)

    curated = _CURATED_REDUCTIONS.get(normalized)
    if curated is not None and curated not in seen:
        variants.append(PronunciationVariant(curated, "curated"))
        seen.add(curated)

    if not variants:
        predicted = tuple(phone for phone in _g2p()(normalized) if phone != " ")
        if predicted:
            variants.append(PronunciationVariant(predicted, "g2p"))
    return tuple(variants)


def select_pronunciation(
    word: str, mode: PronunciationMode = "canonical"
) -> PronunciationVariant:
    """Select the stable canonical or shortest reviewed spoken variant."""
    if mode not in {"canonical", "reduced"}:
        raise ValueError(f"Unsupported pronunciation mode {mode!r}")
    variants = pronunciation_variants(word)
    if not variants:
        raise ValueError(f"Could not generate a pronunciation for {word!r}")

    normalized = _normalize_word(word)
    if mode == "canonical":
        preferred = _CANONICAL_PRONUNCIATIONS.get(normalized)
        if preferred is not None:
            for variant in variants:
                if variant.phones == preferred:
                    return variant
            raise ValueError(
                f"Canonical pronunciation for {word!r} is absent from its inventory"
            )
        return variants[0]

    # Curated reductions win an equal-syllable tie; otherwise CMUdict order is
    # retained. The primary criterion is the number of audible vowel nuclei.
    return min(
        variants,
        key=lambda variant: (
            vowel_count(variant.phones),
            variant.source != "curated",
            variants.index(variant),
        ),
    )


def pronunciation(word: str, *, mode: PronunciationMode = "canonical") -> list[str]:
    """Return the selected ARPABET phones for one word."""
    return list(select_pronunciation(word, mode).phones)


def curated_reductions() -> dict[str, tuple[str, ...]]:
    """Return a copy of reductions that should augment aligner dictionaries."""
    return dict(_CURATED_REDUCTIONS)
