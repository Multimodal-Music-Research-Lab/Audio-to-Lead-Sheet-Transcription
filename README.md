# Audio-to-Lead-Sheet Transcription (A2LS)

> **⚠️ Temporary repository for a paper under review.**
> The audio files cannot be shared for copyright reasons; instead, a subset of the
> test split with pre-computed features is provided so that code and results can be
> inspected without the audio. Data is released under
> [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), code under
> the [MIT licence](LICENSE).



[![Static Badge](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/framework-PyTorch-%23EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Static Badge](https://img.shields.io/badge/code%20licence-MIT-green)](LICENSE)
[![Static Badge](https://img.shields.io/badge/data%20licence-CC--BY--NC--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Static Badge](https://img.shields.io/badge/demo-github_pages-%23825df5?logo=github)](https://multimodal-music-research-lab.github.io/Audio-to-Lead-Sheet-Transcription/)
[![Static Badge](https://img.shields.io/badge/Dataset-A2LS-%23FFD21E?logo=huggingface)](https://huggingface.co/datasets/MMR-Lab/SheetSage-A2LS-subset100)
[![Static Badge](https://img.shields.io/badge/Model-A2LS-%23FF9D00?logo=huggingface)](https://huggingface.co/MMR-Lab/SheetSage-A2LS-model)

Public release of our audio-to-lead-sheet transcription research. From a complete popular music recording with a main vocal melody, the models transcribe a Humdrum `**kern` lead sheet with three spines:
the sung melody (`**kern`), the chord progression (`**cdata`) and the lyrics
(`**text`). This repository contains the inference and evaluation code of the
retrained models, the modular lyrics-alignment code, a 100-sample test subset with
pre-computed [MuQ](https://github.com/tencent-ailab/MuQ) features, and a live
[GitHub Pages demo](https://multimodal-music-research-lab.github.io/Audio-to-Lead-Sheet-Transcription/)
that renders all generated scores in the browser.

```bibtex
@online{dhoogeA2LS,
  title = {Complete {{Audio-to-Lead-Sheet Transcription Using Modular}} and {{End-to-End Approaches}} on the {{New Sheetsage-A2LS Dataset}}},
  author = {D’Hooge, Alexandre and Cummins, Eoin and Huang, Zhongyi and Wu, Zhiyong and Ju, Yaolong},
  date = {2026},
  pubstate = {prepublished},
  note = {Under review for ICASSP 2027}
}
```


## Setup

**Option A — Docker (recommended).**

```bash
docker build -t a2ls .
docker run -it --gpus all --ipc=host -v /path/to/this/repo:/workspace a2ls bash
```

**Option B — Conda / pip.** Requires Python ≥ 3.11 and a CUDA-capable GPU for
inference (evaluation works on CPU too). The system packages `ffmpeg`,
`fluidsynth` and (for data augmentation only) `rubberband-cli` should be
installed beforehand.

```bash
conda create -n a2ls python=3.12 -y
conda activate a2ls
pip install -r requirements.txt
```

The retrained checkpoints are too heavy to commit; download them from the
Hugging Face Hub and place them in `code/weights/`:

```bash
hf download MMR-Lab/SheetSage-A2LS --local-dir code/weights
```

## Data

`data/a2ls-100` is a Hugging Face dataset with 100 random samples of the test
split (seed 42) and pre-computed MuQ features, one row per test sample:

| column        | content                             |
| ------------- | ----------------------------------- |
| `file_name`   | SheetSage record ID                 |
| `features`    | pre-computed MuQ features           |
| `beat_onsets` | beat onset times in seconds         |
| `length`      | number of feature frames            |
| `transcript`  | reference token sequence            |
| `kern`        | ground-truth lead sheet in `**kern` |

The same 100 ground-truth lead sheets are also stored as plain files in
`data/kern/` (with lyrics) and `data/kern-without-lyrics/` (without the lyric
spine). The corresponding generations from every model compared in the paper are
in `krn/`.

The dataset is also hosted on the Hugging Face Hub at
[MMR-Lab/SheetSage-A2LS-subset100](https://huggingface.co/datasets/MMR-Lab/SheetSage-A2LS-subset100).
To download it directly instead of cloning this repository:

```bash
hf download MMR-Lab/SheetSage-A2LS-subset100 \
    --repo-type dataset \
    --local-dir data/a2ls-100
```

## Usage

### 1-A. End-to-End inference

Greedily decode the test split of the bundled 100-sample dataset
(`data/a2ls-100`) and export `metrics.py`-compatible Kern predictions:

```bash
cd code
python inference.py \
    --checkpoint_path weights/char_tok.ckpt \
    --ds_location ../data/a2ls-100 \
    --output_dir ../predictions/character \
    --lyrics_tokeniser character
```

Valid `--lyrics_tokeniser` values: `none`, `word`, `character`, `bpe`
(must match the tokeniser the checkpoint was trained with). Full-test-set runs
simply point `--ds_location` to the complete dataset.

### 1-B. Modular systems

The modular approaches first transcribe the audio with
ASR systems (Qwen3-ASR via wav2vec2 character timestamps, or SoulXSinger),
syllabifying and aligning the syllables to the ground-truth notes, and appending
a `**text` spine to the lyric-free Kern files.

Start from lyric-free Kern predictions (for example from the `cummins_retrained`
checkpoint with `--lyrics_tokeniser none`) and export their note onsets as a
timestamped transcript. The beat onsets come from the ground-truth annotation
JSON, and the lyric reference supplies the placeholder tokens that the aligners
later replace with syllables:

```bash
cd code
python export_kern_timestamped_transcripts.py \
    --kern-files ../predictions/no-lyrics/*.krn \
    --beat-onsets-json ../data/alignments/GT-word-timestamps-100.json \
    --lyric-reference ../data/alignments/GT-word-timestamps-100.json \
    --output ../data/alignments/no-lyrics-timestamps.json
```

This writes `no-lyrics-timestamps.json` plus a `-with_rest` variant, in the same
format as the files the aligners consume. Then align the ASR syllables to those
notes and append the `**text` spine:

```bash
cd code
# Qwen3-ASR transcripts + wav2vec2 character timestamps -> syllable alignment
python modular/align_wav2vec2_syllables.py \
    ../data/alignments/qwen3-asr-wav2vec2-character-timestamps-100.json \
    ../data/alignments/GT-word-timestamps-with_rest-100.json \
    --output ../data/alignments/wav2vec2_aligned-gt-100.json \
    --shift-candidates 16

# SoulXSinger timestamped transcripts -> syllable alignment
python modular/align_soulxsinger_syllables.py \
    ../data/alignments/soulxsinger-timestamped-transcripts-100.json \
    ../data/alignments/GT-word-timestamps-with_rest-100.json \
    --output ../data/alignments/soulx_aligned-gt-100.json \
    --shift-candidates 16

# Append the aligned lyrics as a **text spine
python add_aligned_lyrics_to_kern.py \
    ../data/alignments/wav2vec2_aligned-gt-100.json \
    ../data/kern-without-lyrics \
    ../krn-with-aligned-lyrics/
```

The alignment inputs/outputs shipped in `data/alignments/` cover the 100 demo
samples; the same commands were used on the full test set.

### 2. Metrics

Compare a prediction folder against the ground-truth lead sheets:

```bash
cd code
python metrics.py \
    --pred_kern_path ../predictions/character \
    --ref_kern_path ../data/kern
```


## Repository layout

```
├── code/                      # Inference, evaluation and alignment code
│   ├── inference.py     #   greedy inference for the retrained models
│   ├── metrics.py       #   per-file metric computation
│   ├── add_aligned_lyrics_to_kern.py  # append an aligned **text spine to Kern files
│   ├── export_kern_timestamped_transcripts.py  # beat-aligned lyric cells from Kern files
│   ├── my_utils/, networks/   #   model and tokeniser code
│   ├── modular/               #   modular lyrics alignment (wav2vec2 & SoulXSinger)
│   ├── tokenizers/gpt2/       #   vendored BPE tokeniser (offline-friendly)
│   └── weights/               #   model checkpoints (downloaded, not committed)
├── data/
│   ├── a2ls-100/              # Hugging Face dataset: 100 test samples (+ kern column)
│   ├── kern/                  # ground-truth lead sheets of the 100 samples
│   ├── kern-without-lyrics/   # same, without the lyric spine
│   ├── alignments/            # alignment inputs/outputs for the 100 samples
│   └── samples.json           # sample list used by the demo site
├── krn/                       # scores of the 100 samples shown on the demo site
├── metrics/                   # per-sample error rates used by the demo site
├── mei/                       # generated by convert_kern_to_mei.py for the demo site
├── assets/                    # demo website (Verovio, styles, app)
└── index.html                 # demo website entry point
```

The demo site is served by GitHub Pages from the repository root.

## Licence

Code is released under the [MIT licence](LICENSE); the data (scores, features,
generations) is released under
[CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Audio
files are not distributed in this repository.
