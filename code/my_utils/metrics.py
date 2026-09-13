"""Edit-distance metrics over Kern token sequences."""

VOICE_CHANGE_TOKEN = "<t>"
STEP_CHANGE_TOKEN = "<n>"


def compute_ed_metrics(y_true, y_pred):
    def levenshtein(a, b):
        n, m = len(a), len(b)

        if n > m:
            a, b = b, a
            n, m = m, n

        current = range(n + 1)
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]

    ed_acc = 0
    length_acc = 0
    for t, h in zip(y_true, y_pred):
        ed_acc += levenshtein(t, h)
        length_acc += len(t)

    def as_characters(sequence):
        characters = []
        for token in sequence:
            if token in {"\t", "\n"}:
                characters.append(token)
            else:
                characters.extend(token)
        return characters

    char_ed_acc = 0
    char_length_acc = 0
    for t, h in zip(y_true, y_pred):
        char_ed_acc += levenshtein(as_characters(t), as_characters(h))
        char_length_acc += len(as_characters(t))

    return {
        "sym-er": 0.0 if length_acc == 0 else 100.0 * ed_acc / length_acc,
        "char-er": 0.0
        if char_length_acc == 0
        else 100.0 * char_ed_acc / char_length_acc,
    }


def split_spines_from_tokens(tokens, *, use_literal_delimiters: bool = False):
    mel_tokens = []
    cho_tokens = []
    lyr_tokens = []
    current_voice = 0

    for token in tokens:
        if token == VOICE_CHANGE_TOKEN or (use_literal_delimiters and token == "\t"):
            current_voice += 1
        elif token == STEP_CHANGE_TOKEN or (use_literal_delimiters and token == "\n"):
            current_voice = 0
        else:
            if current_voice == 0:
                mel_tokens.append(token)
            elif current_voice == 1:
                cho_tokens.append(token)
            elif current_voice == 2:
                lyr_tokens.append(token)

    return mel_tokens, cho_tokens, lyr_tokens


def compute_separated_spine_metrics(
    y_true, y_pred, *, use_literal_delimiters: bool = False
):
    y_true_mel, y_true_cho, y_true_lyr = [], [], []
    y_pred_mel, y_pred_cho, y_pred_lyr = [], [], []

    for t in y_true:
        mel, cho, lyr = split_spines_from_tokens(
            t, use_literal_delimiters=use_literal_delimiters
        )
        y_true_mel.append(mel)
        y_true_cho.append(cho)
        y_true_lyr.append(lyr)

    for h in y_pred:
        mel, cho, lyr = split_spines_from_tokens(
            h, use_literal_delimiters=use_literal_delimiters
        )
        y_pred_mel.append(mel)
        y_pred_cho.append(cho)
        y_pred_lyr.append(lyr)

    melody_metrics = compute_ed_metrics(y_true_mel, y_pred_mel)
    chords_metrics = compute_ed_metrics(y_true_cho, y_pred_cho)
    lyrics_metrics = compute_ed_metrics(y_true_lyr, y_pred_lyr)
    metrics = {
        "melody_sym_er": melody_metrics["sym-er"],
        "chords_sym_er": chords_metrics["sym-er"],
        "lyrics_sym_er": lyrics_metrics["sym-er"],
    }
    metrics.update(
        {
            "melody_char_er": melody_metrics["char-er"],
            "chords_char_er": chords_metrics["char-er"],
            "lyrics_char_er": lyrics_metrics["char-er"],
        }
    )
    return metrics
