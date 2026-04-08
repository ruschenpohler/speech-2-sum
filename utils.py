"""
Shared utilities for local speech-to-text transcription.
Used by transcribe_voxtral.py and transcribe_parakeet.py.
"""

import datetime
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import psutil


def print_ram(label=""):
    """Print current process RAM usage."""
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    total = psutil.virtual_memory()
    prefix = f"[RAM] {label} " if label else "[RAM] "
    print(
        f"{prefix}Process: {mem.rss / 1024**3:.1f}GB | "
        f"System: {total.used / 1024**3:.1f}GB / {total.total / 1024**3:.1f}GB "
        f"({total.percent}% used)"
    )


def get_wifi_name():
    """Get current WiFi network name on Windows."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "SSID" in line and "BSSID" not in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown"


def get_location():
    """Get approximate location and timezone from public IP geolocation."""
    try:
        req = urllib.request.Request(
            "http://ip-api.com/json/", headers={"User-Agent": "transcribe"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        parts = []
        for key in ["city", "regionName", "country"]:
            if data.get(key):
                parts.append(data[key])
        location = ", ".join(parts) if parts else "Unknown"
        timezone = data.get("timezone", "")
        return location, timezone
    except Exception:
        return "Unknown", ""


def save_transcript(
    transcript: str,
    source: str,
    start_time: datetime.datetime,
    duration_sec: float,
    model_name: str = "Unknown",
    model_params: dict = None,
):
    """Save transcript to transcripts/ folder as a timestamped .md file.

    model_params: optional dict of quality-relevant parameters to log,
    e.g. {"delay_ms": 960} for Voxtral. Keys/values are appended to the
    header as-is, so only pass parameters that meaningfully affect output.
    """
    out_dir = Path("transcripts")
    out_dir.mkdir(exist_ok=True)

    wifi = get_wifi_name()
    location, timezone = get_location()

    ts = start_time.strftime("%Y-%m-%d_%Hh-%Mm")
    ts_with_tz = start_time.strftime("%Y-%m-%d %H:%M:%S")
    if timezone:
        ts_with_tz += f" {timezone}"

    # Format: [xx]sec for <1 min, [xx]min for >=1 min
    if duration_sec < 60:
        dur_str = f"{round(duration_sec)}sec"
    else:
        dur_str = f"{round(duration_sec / 60)}min"
    filename = out_dir / f"{ts}--{dur_str}.md"

    params_str = ""
    if model_params:
        params_str = " | " + ", ".join(f"{k}: {v}" for k, v in model_params.items())

    header = (
        f"# {ts_with_tz} | "
        f"Model: {model_name}{params_str} | "
        f"{location} | WiFi: {wifi} | Source: {source}"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        f.write(transcript + "\n")

    print(f"\nSaved to: {filename}")
    return filename
