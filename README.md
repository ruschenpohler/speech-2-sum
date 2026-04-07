# Voxtral Mini 4B Realtime - Local CPU Deployment

Local speech-to-text using Mistral's Voxtral Mini 4B Realtime model.

## Quick Start

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Transcribe an audio file
python transcribe.py path/to/audio.mp3
```

## Setup

```powershell
# Create virtual environment (if starting fresh)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=5.2.0" "mistral-common[audio]>=1.9.0" librosa soxr soundfile
```

## Usage

```powershell
# Basic usage
python transcribe.py audio.mp3

# Specify a different model path or local cache
python transcribe.py audio.wav --cache-dir D:\models

# Help
python transcribe.py --help
```

## Notes

- **CPU inference**: Expect ~0.5-2x real-time factor depending on CPU
- **Supported audio formats**: mp3, wav, flac, ogg, and other formats supported by librosa/soundfile
- **Model size**: ~8GB disk space for BF16 weights

## Fallback: voxtral.c

If Transformers is too slow, see `fallbacks/voxtral.c/` for a faster pure-C implementation.
