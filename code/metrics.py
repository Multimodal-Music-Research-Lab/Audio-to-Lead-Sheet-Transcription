import pathlib
import re
from collections import defaultdict

import fire
import my_utils.metrics as M
from harte_matching import pitch_bass_to_scale_degree
from modular.lyrics_normalization import normalize
from tqdm import tqdm

ERROR_METRIC_NAMES = (
    "sym-er",
    "char-er",
    "hard_sym-er",
    "hard_char-er",
    "melody_sym_er",
    "chords_sym_er",
    "lyrics_sym_er",
    "melody_hard_sym_er",
    "chords_hard_sym_er",
    "lyrics_hard_sym_er",
    "melody_hard_char_er",
    "chords_hard_char_er",
    "lyrics_hard_char_er",
    "melody_char_er",
    "chords_char_er",
    "lyrics_char_er",
)

HARD_IGNORE_PREFIXES = ("!", "*", "=")
HARD_IGNORE_TOKENS = (".", "\t", "\n")


def _is_ignored_kern_token(token: str) -> bool:
    return token in HARD_IGNORE_TOKENS or token.startswith(HARD_IGNORE_PREFIXES)


def _filter_kern_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if not _is_ignored_kern_token(token)]


def _hard_ed_metrics(y_true: list[list[str]], y_pred: list[list[str]]) -> dict[str, float]:
    """Sym-ER and char-ER over sequences with structural kern tokens removed.

    Ignored tokens (``!``/``*``/``=`` prefixes, bare ``.`` and the tab/newline
    delimiters) are removed entirely before the remaining tokens are assembled
    into the character sequences used for the char-ER.
    """
    filtered_true = [_filter_kern_tokens(seq) for seq in y_true]
    filtered_pred = [_filter_kern_tokens(seq) for seq in y_pred]
    if sum(len(seq) for seq in filtered_true) == 0:
        return {"sym-er": 0.0, "char-er": 0.0}
    ed_metrics = M.compute_ed_metrics(filtered_true, filtered_pred)
    return {"sym-er": ed_metrics["sym-er"], "char-er": ed_metrics["char-er"]}


def _hard_spine_ed_ers(ref: list[list[str]], pred: list[list[str]]) -> dict[str, float]:
    """Per-spine sym-ER and char-ER with structural kern tokens removed."""
    ref_spines = [
        M.split_spines_from_tokens(seq, use_literal_delimiters=True) for seq in ref
    ]
    pred_spines = [
        M.split_spines_from_tokens(seq, use_literal_delimiters=True) for seq in pred
    ]
    metrics = {}
    for spine_name, index in (("melody", 0), ("chords", 1), ("lyrics", 2)):
        ed_metrics = _hard_ed_metrics(
            [spines[index] for spines in ref_spines],
            [spines[index] for spines in pred_spines],
        )
        metrics[f"{spine_name}_hard_sym_er"] = ed_metrics["sym-er"]
        metrics[f"{spine_name}_hard_char_er"] = ed_metrics["char-er"]
    return metrics


def _extract_id(name: str, ground_truth_kern: bool) -> str:
    """Return a Kern record ID without assuming a fixed ID length."""
    if ground_truth_kern:
        return pathlib.Path(name).stem
    else:
        return name[1:12]


def _str_to_kern(file: str) -> list[str]:
    out = re.split(r"([\t\n])", file)
    if out and out[-1] == "":
        out.pop()
    return out


def _load_pred(id, PRED):
    with open(PRED / f"_{id}__prediction.krn", "r", encoding="utf-8", newline="") as f:
        text = f.read()
    text = _drop_comments(text)
    krn = _str_to_kern(text)
    return krn


def match_headers(pred_text: str) -> str:
    """Canonicalize prediction header cdata for any spine count.

    Non-kern spines represent every interpretation row (clef, key signature,
    tonic and meter) by a single ``*``; duplicated interpretation rows emitted
    by the decoder are dropped.  The ``*above`` interpretation row is
    canonicalized immediately before the first barline.  Lyric-free
    predictions have two spines, lyric models three.
    """
    pred_lines = pred_text.splitlines(keepends=True)

    first_bar = next(
        (
            index
            for index, line in enumerate(pred_lines)
            if line.rstrip("\r\n").split("\t")[0].startswith("=")
        ),
        len(pred_lines),
    )
    spine_count = next(
        (
            len(line.rstrip("\r\n").split("\t"))
            for line in pred_lines
            if line.startswith("**")
        ),
        3,
    )
    above_row = "\t".join(["*", "*above"] + ["*"] * (spine_count - 2))
    output = []
    inserted = False
    seen_interpretations = set()
    for index, line in enumerate(pred_lines):
        content = line.rstrip("\r\n")
        ending = line[len(content) :]
        columns = content.split("\t")
        if (
            index < first_bar
            and not content.startswith("!")
            and columns[0].startswith("*")
        ):
            row_key = tuple(columns)
            if row_key in seen_interpretations:
                continue
            seen_interpretations.add(row_key)
            if len(columns) >= 2 and not columns[0].startswith("**"):
                if columns[0] == "*" and "*above" in columns[1:]:
                    continue
                for spine_index in range(1, len(columns)):
                    columns[spine_index] = "*"
                line = "\t".join(columns) + ending
        if index == first_bar and not inserted:
            output.append(above_row + "\n")
            inserted = True
        output.append(line)
    if not inserted:
        output.append(above_row + "\n")
    return "".join(output)


def _drop_comments(text: str) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(line for line in lines if not line.startswith("!"))


def _normalize_predicted_lyrics(text: str) -> str:
    """Normalize only cells in the predicted ``**text`` spine."""
    lines = text.splitlines(keepends=True)
    header = None
    for line in lines:
        content = line.rstrip("\r\n")
        columns = content.split("\t")
        if "**text" in columns:
            header = columns
            break
    if header is None:
        return text

    normalized_lines = []
    for line in lines:
        content = line.rstrip("\r\n")
        ending = line[len(content) :]
        columns = content.split("\t")
        if len(columns) != len(header) or content.startswith(("!", "*")):
            normalized_lines.append(line)
            continue
        for index, spine in enumerate(header):
            if spine == "**text" and columns[index] not in {".", "="}:
                columns[index] = normalize(columns[index])
        normalized_lines.append("\t".join(columns) + ending)
    return "".join(normalized_lines)


def _normalize_predicted_chords(text: str) -> str:
    """Canonicalize pitch-name chord basses in predicted chord spines.

    Harte labels represent inversions as scale degrees (for example,
    ``C:maj/E`` becomes ``C:maj/3``).  Labels that cannot be parsed are
    preserved so an invalid model prediction remains visible in the metrics.
    """
    lines = text.splitlines(keepends=True)
    header = None
    for line in lines:
        columns = line.rstrip("\r\n").split("\t")
        if "**cdata" in columns or "**harm" in columns:
            header = columns
            break
    if header is None:
        return text

    chord_spines = {
        index for index, spine in enumerate(header) if spine in {"**cdata", "**harm"}
    }
    normalized_lines = []
    for line in lines:
        content = line.rstrip("\r\n")
        ending = line[len(content) :]
        columns = content.split("\t")
        if len(columns) != len(header) or content.startswith(("!", "*", "=")):
            normalized_lines.append(line)
            continue
        for index in chord_spines:
            chord_label = columns[index]
            if re.search(r"/[A-G][#b]*$", chord_label):
                try:
                    columns[index] = pitch_bass_to_scale_degree(chord_label)
                except ValueError:
                    pass
        normalized_lines.append("\t".join(columns) + ending)
    return "".join(normalized_lines)


def _load_ref(id, REF):
    with open(REF / f"{id}.krn", "r", encoding="utf-8", newline="") as f:
        text = f.read()
    text = _drop_comments(text)
    krn = _str_to_kern(text)
    return krn


def main(
    pred_kern_path: str,
    ref_kern_path: str,
    max_count: int = -1,
    ground_truth_kern: bool = False,
    skip_errors: bool = False,
):
    PRED = pathlib.Path(pred_kern_path)
    REF = pathlib.Path(ref_kern_path)
    files = sorted(PRED.glob("*.krn"))
    if max_count != -1:
        if max_count == 0 or max_count < -1:
            raise ValueError(
                "Cannot process 0 or less samples. Use '-1' to process the full data"
            )
        files = files[:max_count]
    ids = [_extract_id(path.name, ground_truth_kern) for path in files]
    metric_totals = defaultdict(float)
    processed_count = 0
    for id in tqdm(ids):
        try:
            ref = _load_ref(id, REF)
            pred_name = f"{id}.krn" if ground_truth_kern else f"_{id}__prediction.krn"
            with open(PRED / pred_name, "r", encoding="utf-8", newline="") as f:
                pred_text = f.read()
            with open(REF / f"{id}.krn", "r", encoding="utf-8", newline="") as f:
                ref_text = f.read()
            if not ground_truth_kern:
                pred_text = match_headers(pred_text)
            pred_text = _normalize_predicted_lyrics(pred_text)
            if not ground_truth_kern:
                pred_text = _normalize_predicted_chords(pred_text)
            pred = _str_to_kern(_drop_comments(pred_text))
            metrics = M.compute_ed_metrics([ref], [pred])
            metrics.update(
                M.compute_separated_spine_metrics(
                    [ref], [pred], use_literal_delimiters=True
                )
            )
            hard_ed = _hard_ed_metrics([ref], [pred])
            metrics["hard_sym-er"] = hard_ed["sym-er"]
            metrics["hard_char-er"] = hard_ed["char-er"]
            metrics.update(_hard_spine_ed_ers([ref], [pred]))
        except Exception as error:
            print(f"Error evaluating {id}: {error}")
            if skip_errors:
                continue
            metrics = {metric_name: 100.0 for metric_name in ERROR_METRIC_NAMES}

        for metric_name, value in metrics.items():
            metric_totals[metric_name] += value
        processed_count += 1

    if processed_count:
        average_metrics = {
            metric_name: total / processed_count
            for metric_name, total in metric_totals.items()
        }
        print(f"Total average ({processed_count} files): {average_metrics}")
    else:
        print("Total average: no files processed")


if __name__ == "__main__":
    fire.Fire(main)
