# Local Speech-to-Text

Private, offline speech-to-text on CPU. Two model backends are available — pick one based on your hardware and accuracy needs.

| | Voxtral Mini 4B | Parakeet TDT 0.6B |
|---|---|---|
| Parameters | 4B | 0.6B |
| Weights | ~8 GB (BF16) | ~2.5 GB (FP32) |
| Inference | Batch or Streaming | Batch |
| Delay tuning | Yes (`--del`) | No |
| Toolkit | HuggingFace Transformers | NVIDIA NeMo |
| Windows native | Yes | WSL2 best |

---

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.10+.

```powershell
git clone https://github.com/ruschenpohler/speech-2-sum.git
cd speech-2-sum

# Interactive setup — choose Voxtral, Parakeet, or both
python install.py
```

`install.py` will create a `.venv`, install the right dependencies for your chosen backend, and print the next steps.

### Manual install (alternative)

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1

uv pip install -e ".[voxtral]"   # Mistral Voxtral
uv pip install -e ".[parakeet]"  # NVIDIA Parakeet
uv pip install -e ".[all]"       # both
```

### Download model weights

Run once before first use. Weights are cached for all future transcriptions.

```powershell
python download_voxtral.py          # ~8 GB
python download_parakeet.py         # ~2.5 GB

# Optional: custom cache location
python download_voxtral.py  --cache-dir D:\models
python download_parakeet.py --cache-dir D:\models
```

---

## Quick Start

```powershell
.\.venv\Scripts\Activate.ps1

# Transcribe a file
python transcribe.py --voxtral  audio.mp3
python transcribe.py --parakeet audio.mp3

# Record from microphone
python transcribe.py --voxtral  --mic --dur 30
python transcribe.py --parakeet --mic --dur 30
```

---

## Usage

### All flags

| Flag | Values | Applies to |
|---|---|---|
| `--voxtral` / `--parakeet` | — | `transcribe.py` dispatcher |
| `--mic` | — | both |
| `--mic-loop` | — | both |
| `--kHz N` | integer (default: 16) | both |
| `audio.mp3` | any audio file | both |
| `--dur N` | minutes (default: 60) | both |
| `--del N` | ms: 80, 480, 960, 2400 (default: 960) | Voxtral only |
| `--cache-dir PATH` | directory | both |

`--del` controls Voxtral's transcription delay: higher values improve accuracy at the cost of latency. It is accepted but ignored by Parakeet for CLI consistency.

### Model-specific scripts (also work standalone)

```powershell
python transcribe_voxtral.py  --mic --dur 60 --del 960
python transcribe_parakeet.py --mic --dur 30
```

---

## Output

Transcripts are saved to `transcripts/` as timestamped Markdown files:

```
transcripts/2026-04-08_14h-32m--5min.md
transcripts/2026-04-08_14h-32m--30sec.md
```

Filename format: `YYYY-MM-DD_HHh-MMm--[duration].[ext]`
- `[duration]` is in seconds (`30sec`) for recordings under 1 minute, or minutes (`5min`) for 1 minute and over.

Each file header includes the model name, [delay parameter for Voxtral,] timestamp, approximate location (from IP), WiFi network, and audio source.

---

## Notes

- **Parakeet on Windows:** NVIDIA NeMo is not officially supported on native Windows (>> may use WSL2)
- **Voxtral on CPU:** Expect ~0.5–2× real-time factor depending on CPU
- **Supported audio formats:** mp3, wav, flac, ogg, and others

- **Fallback:** If Transformers is too slow for Voxtral, [antirez/voxtral.c](https://github.com/antirez/voxtral.c) is a faster pure-C implementation. May benchmark later

---

## Summarization

Local summarization using Qwen2.5-7B-Instruct (GGUF Q8_0) via llama.cpp. Produces a hybrid output: abstractive summary + extracted verbatim quotes. Requires ~12 GB RAM.

### Setup

```powershell
uv pip install -e ".[summarize]"
python download_summarizer.py   # ~6.7 GB
```

### Usage

```powershell
# Standalone — summarize a specific transcript
python summarize.py transcripts/2026-04-08_14h-32m--5min.md

# Auto-summarize immediately after transcription finishes
python transcribe.py --voxtral  --mic --sum
python transcribe.py --parakeet --mic --sum
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--keep` | `last` | Truncation strategy when input exceeds `--max-cont`: `first` (lectures/presentations) or `last` (meetings/conversations) |
| `--max-cont` | 4000 | Max input tokens before truncation |
| `--max-out` | 500 | Max output tokens |
| `--cache-dir` | `./models` | Directory containing the GGUF file |

### Output

Saved to `summaries/[transcript_stem]_SUM.md`, including summary, key quotes, and inference metrics.
