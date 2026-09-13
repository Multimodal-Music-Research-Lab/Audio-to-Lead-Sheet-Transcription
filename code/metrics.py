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
    "melody_sym_er",
    "chords_sym_er",
    "lyrics_sym_er",
    "melody_char_er",
    "chords_char_er",
    "lyrics_char_er",
)


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


def match_headers(pred_text: str, ref_text: str) -> str:
    """Align prediction header cdata with the reference kern header.

    Key/tonic and meter interpretations are intentionally represented by a
    single ``*`` in the cdata spine.  The ``*above`` interpretation row is
    canonicalized immediately before the first barline.
    """
    pred_lines = pred_text.splitlines(keepends=True)
    ref_lines = ref_text.splitlines()

    def header_kind(token: str) -> str | None:
        if token.startswith("*clef"):
            return "clef"
        if re.match(r"^\*k\[.*\]$", token):
            return "key_signature"
        if re.match(r"^\*[A-Ga-g][#n-]*:$", token):
            return "tonic"
        if token.startswith("*M"):
            return "meter"
        return None

    ref_values = {}
    for line in ref_lines:
        columns = line.split("\t")
        if len(columns) < 2 or columns[0].startswith(("!", "=")):
            continue
        kind = header_kind(columns[0])
        if kind is not None:
            ref_values[kind] = columns[0]

    first_bar = next(
        (
            index
            for index, line in enumerate(pred_lines)
            if line.rstrip("\r\n").split("\t")[0].startswith("=")
        ),
        len(pred_lines),
    )
    output = []
    inserted = False
    for index, line in enumerate(pred_lines):
        content = line.rstrip("\r\n")
        ending = line[len(content) :]
        columns = content.split("\t")
        if index < first_bar and len(columns) == 3 and not content.startswith("!"):
            kern_token = columns[0]
            kind = header_kind(kern_token)
            if kind is not None:
                cdata = ref_values.get(kind, kern_token)
                if (
                    re.match(r"^\*k\[.*\]$", cdata)
                    or re.match(r"^\*[A-Ga-g][#n-]*:$", cdata)
                    or cdata.startswith("*M")
                ):
                    cdata = "*"
                columns[1] = cdata
            if columns[0] == "*" and columns[1] == "*above":
                continue
            line = "\t".join(columns) + ending
        if index == first_bar and not inserted:
            output.append("*\t*above\t*\n")
            inserted = True
        output.append(line)
    if not inserted:
        output.append("*\t*above\t*\n")
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
                pred_text = match_headers(pred_text, ref_text)
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
