"""Text normalisation shared with the lyric metrics."""

from __future__ import annotations


def normalize(text: str) -> str:
    """Normalize text for scoring after generated-text validation."""
    text = text.translate(
        str.maketrans(
            {
                "\u2018": "'",  # left single quotation mark
                "\u2019": "'",  # right single quotation mark
                "\uff07": "'",  # fullwidth apostrophe
                "\u2032": "'",  # slanted apostrophe
                "\uff08": "(",  # fullwidth parenthesis left
                "\uff09": ")",  # fullwidth parenthesis left
                "\ufffd": " ",  # unknown unicode char
                "\uff0c": ",",  # fullwidth comma
                "\u2014": "-",  # em dash
                "\u2026": "...",  # triple dots
                "\uff1f": "?",  # fullwidth question mark
                "\u3002": ".",  # chinese dot
            }
        )
    )
    text = text.translate(
        str.maketrans(
            {
                ".": " ",  # convert dot to space to keep acronyms separate
            }
        )
    )
    text = text.translate(str.maketrans("", "", ':;,!?"()\\[]'))
    return " ".join(text.lower().split())
