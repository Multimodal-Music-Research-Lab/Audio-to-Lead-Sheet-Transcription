"""Greedily export flat-model test predictions for ``metrics.py``."""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import fire
import torch
from datasets import load_from_disk
from my_utils.consts import (
    EOS_TOKEN,
    LYRIC_BPE_TOKENISER,
    LYRIC_CHARACTER_TOKENISER,
    LYRIC_TOKENISER,
    NO_LYRICS_TOKENISER,
    SOS_TOKEN,
)
from my_utils.lyrics_bpe_tokeniser import LyricsBPETokeniser
from my_utils.tokeniser import untokenize
from networks.transformer.model import A2STransformer
from tqdm import tqdm

VALID_LYRICS_TOKENISERS = frozenset(
    {
        NO_LYRICS_TOKENISER,
        LYRIC_TOKENISER,
        LYRIC_CHARACTER_TOKENISER,
        LYRIC_BPE_TOKENISER,
    }
)


def _safe_id(value: str, fallback: int) -> str:
    value = Path(str(value)).stem or str(value)
    # Underscores are part of SheetSage IDs; preserve leading/trailing ones.
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value or f"sample_{fallback}"


def _paper_output_ids(test_split) -> list[str]:
    if "file_name" not in test_split.column_names:
        raise ValueError("The test split must include a file_name column.")
    ids = [
        _safe_id(value, index) for index, value in enumerate(test_split["file_name"])
    ]
    invalid_ids = [file_id for file_id in ids if len(file_id) != 11]
    if invalid_ids:
        raise ValueError(
            "metrics.py requires exactly 11-character record IDs; "
            f"got {invalid_ids[0]!r}."
        )
    duplicate_ids = {file_id for file_id in ids if ids.count(file_id) > 1}
    if duplicate_ids:
        raise ValueError(
            f"Test split has duplicate record ID {sorted(duplicate_ids)[0]!r}."
        )
    return ids


def _greedy_decode(
    model, features: torch.Tensor, start_token: int, end_token: int
) -> list[int]:
    y_in = torch.zeros(
        1, model.max_seq_len + 1, dtype=torch.long, device=features.device
    )
    y_in[0, 0] = start_token
    with torch.inference_mode():
        memory = model.encoder(x=features)
        tokens = []
        for step in range(model.max_seq_len):
            logits = model.decoder(
                tgt=y_in[:, : step + 1], memory=memory, memory_len=None
            )
            token_id = logits[0, :, -1].argmax(dim=-1).item()
            if token_id == end_token:
                return tokens
            tokens.append(token_id)
            y_in[0, step + 1] = token_id
    return tokens


def _render_kern(
    tokens: list[str],
    record_id: str,
    lyrics_tokeniser: str,
    lyrics_bpe_tokeniser: LyricsBPETokeniser | None = None,
) -> str:
    body = untokenize(tokens, lyrics_bpe_tokeniser)
    body_lines = [line for line in body.splitlines() if line.strip()]
    expected_spine_count = 2 if lyrics_tokeniser == NO_LYRICS_TOKENISER else 3
    exclusive_index = next(
        (
            index
            for index, line in enumerate(body_lines)
            if line.split("\t")[0].startswith("**")
        ),
        None,
    )
    if exclusive_index is not None:
        spine_count = len(body_lines[exclusive_index].split("\t"))
        body_lines = [
            line
            for line in body_lines
            if not line.startswith("**") and line.split("\t") != ["*-"] * spine_count
        ]
        if body_lines and all(
            cell.startswith("=") for cell in body_lines[-1].split("\t")
        ):
            body_lines.pop()
    else:
        spine_count = max((len(line.split("\t")) for line in body_lines), default=0)
    if not body_lines or spine_count != expected_spine_count:
        raise ValueError(f"{record_id}: greedy decoder emitted no Kern content.")
    if any(len(line.split("\t")) != spine_count for line in body_lines):
        raise ValueError(
            f"{record_id}: greedy decoder emitted malformed "
            f"{expected_spine_count}-spine Kern."
        )
    spine_types = ["**kern", "**cdata", "**text"][:expected_spine_count]
    clefs = ["*clefG2"] + ["*"] * (expected_spine_count - 1)
    footer = ["=="] * expected_spine_count
    terminators = ["*-"] * expected_spine_count
    return "\n".join(
        [
            "!! Model prediction",
            "\t".join(spine_types),
            "\t".join(clefs),
            *body_lines,
            "\t".join(footer),
            "\t".join(terminators),
        ]
    )


@torch.inference_mode()
def infer(
    checkpoint_path: str,
    ds_location: str,
    output_dir: str,
    max_samples: int = -1,
    lyrics_tokeniser: str = NO_LYRICS_TOKENISER,
) -> None:
    """Export flat-model predictions with optional lyric-spine decoding."""
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be positive or -1 for the full test split.")
    if lyrics_tokeniser not in VALID_LYRICS_TOKENISERS:
        choices = ", ".join(sorted(VALID_LYRICS_TOKENISERS))
        raise ValueError(f"lyrics_tokeniser must be one of: {choices}")

    dataset = load_from_disk(ds_location)
    if "test" not in dataset:
        raise ValueError("Dataset must contain a test split.")
    test_split = dataset["test"]
    required_columns = {"features", "file_name"}
    missing_columns = required_columns - set(test_split.column_names)
    if missing_columns:
        raise ValueError(
            f"Test split is missing columns: {', '.join(sorted(missing_columns))}"
        )
    output_ids = _paper_output_ids(test_split)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if list(output_path.glob("*.krn")):
        raise ValueError("output_dir must contain no existing .krn files.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint["hyper_parameters"]
    bpe_metadata = hyper_parameters.get("lyrics_bpe")
    if bpe_metadata is not None and lyrics_tokeniser != LYRIC_BPE_TOKENISER:
        raise ValueError("lyrics_tokeniser conflicts with the checkpoint BPE tokenizer")
    if lyrics_tokeniser == LYRIC_BPE_TOKENISER:
        if not isinstance(bpe_metadata, dict) or not isinstance(
            bpe_metadata.get("model_name"), str
        ):
            raise ValueError(
                "BPE inference requires lyrics_bpe metadata in the checkpoint"
            )
        lyrics_bpe_tokeniser = LyricsBPETokeniser(bpe_metadata["model_name"])
    else:
        lyrics_bpe_tokeniser = None
    model = A2STransformer.load_from_checkpoint(checkpoint_path, strict=True)
    model.requires_grad_(False).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_token = model.w2i[SOS_TOKEN]
    end_token = model.w2i[EOS_TOKEN]
    total = len(test_split) if max_samples == -1 else min(len(test_split), max_samples)
    failure_path = output_path / "failures.jsonl"
    failures = 0
    with failure_path.open("w", encoding="utf-8") as failure_file:
        for index in tqdm(range(total), desc="greedy inference"):
            record_id = output_ids[index]
            token_ids: list[int] = []
            tokens: list[str] = []
            try:
                features = torch.as_tensor(
                    test_split[index]["features"], dtype=torch.float32, device=device
                )
                token_ids = _greedy_decode(model, features, start_token, end_token)
                tokens = [model.i2w[token_id] for token_id in token_ids]
                kern = _render_kern(
                    tokens,
                    record_id,
                    lyrics_tokeniser,
                    lyrics_bpe_tokeniser,
                )
                (output_path / f"_{record_id}__prediction.krn").write_text(
                    kern, encoding="utf-8"
                )
            except Exception as error:
                failures += 1
                json.dump(
                    {
                        "index": index,
                        "record_id": record_id,
                        "error": f"{type(error).__name__}: {error}",
                        "token_ids": token_ids,
                        "tokens": tokens,
                        "traceback": traceback.format_exc(),
                    },
                    failure_file,
                )
                failure_file.write("\n")
                failure_file.flush()
    print(
        f"Wrote {total - failures} metrics.py-compatible predictions to {output_path}; "
        f"recorded {failures} failures in {failure_path}"
    )


if __name__ == "__main__":
    fire.Fire(infer)
