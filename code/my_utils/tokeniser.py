"""Untokenise model token sequences back into Kern text."""

from __future__ import annotations

from my_utils.consts import (
    HYPHEN_TOKEN,
    LYRIC_TOKEN_PREFIX,
    MELISMA_TOKEN,
    TEXT_TOKEN_PREFIX,
)
from my_utils.lyrics_bpe_tokeniser import LyricsBPETokeniser


def untokenize(tokens, lyrics_bpe_tokeniser: LyricsBPETokeniser | None = None):
    """Untokenizes a list of tokens into a string."""
    if lyrics_bpe_tokeniser is not None:
        rendered = []
        bpe_tokens = []
        for token in tokens:
            if lyrics_bpe_tokeniser.unwrap(token) is not None:
                bpe_tokens.append(token)
                continue
            if bpe_tokens:
                rendered.append(lyrics_bpe_tokeniser.decode_wrappers(bpe_tokens))
                bpe_tokens = []
            rendered.append(token)
        if bpe_tokens:
            rendered.append(lyrics_bpe_tokeniser.decode_wrappers(bpe_tokens))
        tokens = rendered
    return (
        "".join(tokens)
        .replace("<t>", "\t")
        .replace("<n>", "\n")
        .replace("<s>", " ")
        .replace("<chord-pitch>", "")
        .replace("<chord-extension>", "")
        .replace(HYPHEN_TOKEN, "-")
        .replace(MELISMA_TOKEN, "_")
        .replace(LYRIC_TOKEN_PREFIX, "")
        .replace(TEXT_TOKEN_PREFIX, "")
    )
