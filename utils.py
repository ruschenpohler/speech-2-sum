"""
Shared utilities for local speech-to-text transcription.
Used by transcribe_voxtral.py and transcribe_parakeet.py.
"""

import datetime
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import psutil


def get_loopback_device(verbose: bool = True, api_filter: str = None):
    """
    Find the WASAPI loopback device corresponding to the default audio output.
    Returns the sounddevice device index, or None if not found.
    On Windows, also checks for "PC Speaker" and "Stereo Mix" via WDM-KS.

    Args:
        verbose: if True, print all detected input devices for debugging
        api_filter: if set, only return devices matching this API (e.g., "WASAPI" or "WDM-KS")
    """
    import sounddevice as sd

    try:
        default_out = sd.query_devices(kind="output")
        default_out_name = default_out["name"]
        if verbose:
            print(f"  Default output: {default_out_name}")

            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0 and dev.get("hostapi") is not None:
                    host_api = sd.query_hostapis(dev["hostapi"])
                    print(
                        f"  Device {i}: {dev['name']} | inputs: {dev['max_input_channels']} | API: {host_api['name']}"
                    )

        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] == 0:
                continue

            dev_name = dev["name"].lower()
            host_api = sd.query_hostapis(dev["hostapi"])
            api_name = host_api["name"]

            # Filter by API if requested
            if api_filter and api_filter not in api_name:
                continue

            # Match if device name contains default output name OR is PC Speaker/Stereo Mix
            # Priority: Stereo Mix first (often more reliable), then PC Speaker, then generic match
            is_loopback = (
                "stereo mix" in dev_name
                or "pc speaker" in dev_name
                or default_out_name.lower() in dev_name
                or "loopback" in dev_name
            )

            # Accept WASAPI or WDM-KS (WDM-KS is how PC Speaker appears on Windows)
            is_windows_api = "WASAPI" in api_name or "WDM-KS" in api_name

            if (
                is_loopback
                and is_windows_api
                and i != sd.query_devices(kind="input")["index"]
            ):
                print(
                    f"  Found loopback device: {dev['name']} (index {i}, API: {api_name})"
                )
                return i
    except Exception as e:
        print(f"  Error detecting loopback: {e}")
    return None


def record_until_esc_or_timeout(duration: float):
    """
    Block for up to `duration` seconds, returning early if ESC is pressed.
    Uses the `keyboard` package for non-blocking key detection.
    Falls back to plain sleep (Ctrl+C to stop) if `keyboard` is unavailable
    or lacks OS permissions.
    """
    try:
        import keyboard

        print("Press ESC to stop recording early.\n")
        start = time.time()
        while time.time() - start < duration:
            if keyboard.is_pressed("esc"):
                print("\nESC pressed — stopping recording.")
                break
            time.sleep(0.05)
    except Exception:
        print("Press Ctrl+C to stop recording early.\n")
        time.sleep(duration)


def record_audio(
    mic: bool,
    loopback: bool,
    duration_sec: float,
    kHz: int = 16,
    wait_for_loopback: int = 30,
) -> tuple[np.ndarray, str]:
    """
    Record audio from mic, system loopback, or both simultaneously.

    Args:
        mic: record physical microphone
        loopback: record system audio via WASAPI loopback
        duration_sec: max recording length in seconds
        kHz: sample rate in kHz (default: 16)
        wait_for_loopback: seconds to wait for loopback device to appear (default: 30)

    Returns:
        (audio_array, source_label)
        - audio_array: float32 numpy array, mono, at kHz*1000 Hz
        - source_label: "Microphone", "System Audio (loopback)", or "Microphone + System Audio"
    """
    import sounddevice as sd

    if not mic and not loopback:
        raise ValueError("At least one of mic or loopback must be True")

    sample_rate = kHz * 1000

    loopback_device = None
    if loopback:
        # First try WASAPI loopback (works with headphones/external audio)
        loopback_device = get_loopback_device(verbose=True, api_filter="WASAPI")
        # If not found, try WDM-KS devices (PC Speaker / Stereo Mix for built-in speakers)
        if loopback_device is None:
            print("  No WASAPI loopback found, trying WDM-KS devices...")
            loopback_device = get_loopback_device(verbose=True, api_filter="WDM-KS")

        if loopback_device is None and wait_for_loopback > 0:
            print(f"  Waiting up to {wait_for_loopback}s for system audio to start...")
            start = time.time()
            while time.time() - start < wait_for_loopback:
                time.sleep(2)
                # Try WDM-KS during wait (WASM API probably won't appear mid-recording)
                loopback_device = get_loopback_device(
                    verbose=False, api_filter="WDM-KS"
                )
                if loopback_device is not None:
                    print(
                        f"  Detected system audio (device: {sd.query_devices(loopback_device)['name']})"
                    )
                    break
                print(f"  Still waiting... ({int(time.time() - start)}s)")
            if loopback_device is None:
                print(
                    f"  No system audio detected after {wait_for_loopback}s — falling back to mic."
                )
                loopback = False
                if not mic:
                    mic = True

    if mic and loopback:
        source_label = "Microphone + System Audio"
    elif loopback:
        source_label = "System Audio (loopback)"
    else:
        source_label = "Microphone"

    print(f"  Source: {source_label}")

    # Check and adjust sample rate for loopback device if needed
    if loopback:
        try:
            device_info = sd.query_devices(loopback_device)
            device_rate = int(device_info.get("default_sample_rate", sample_rate))
            if device_rate != sample_rate:
                print(f"  Adjusting to device sample rate: {device_rate}Hz")
                sample_rate = device_rate
        except Exception as e:
            print(f"  Could not query device sample rate: {e}")

    mic_chunks = []
    loop_chunks = []

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", file=__import__("sys").stderr)
        mic_chunks.append(indata.copy())

    def loop_callback(indata, frames, time_info, status):
        if status:
            print(f"[loop] {status}", file=__import__("sys").stderr)
        loop_chunks.append(indata.copy())

    streams = []
    if mic:
        streams.append(
            sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                callback=mic_callback,
            )
        )
    if loopback:
        streams.append(
            sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=loopback_device,
                callback=loop_callback,
            )
        )

    class MultiStream:
        def __init__(self, streams):
            self.streams = streams

        def __enter__(self):
            for s in self.streams:
                s.__enter__()
            return self

        def __exit__(self, *args):
            for s in self.streams:
                s.__exit__(*args)

    with MultiStream(streams):
        record_until_esc_or_timeout(duration_sec)

    if mic and loopback and mic_chunks and loop_chunks:
        mic_arr = np.concatenate(mic_chunks).flatten()
        loop_arr = np.concatenate(loop_chunks).flatten()
        min_len = min(len(mic_arr), len(loop_arr))
        mic_arr = mic_arr[:min_len]
        loop_arr = loop_arr[:min_len]
        mic_max = np.abs(mic_arr).max()
        loop_max = np.abs(loop_arr).max()
        if mic_max > 0:
            mic_arr = mic_arr / mic_max
        if loop_max > 0:
            loop_arr = loop_arr / loop_max
        audio = (mic_arr + loop_arr) / 2.0
    elif mic and mic_chunks:
        audio = np.concatenate(mic_chunks).flatten()
    elif loopback and loop_chunks:
        audio = np.concatenate(loop_chunks).flatten()
    else:
        audio = np.array([], dtype=np.float32)
        print(
            "  Warning: No audio recorded. System audio device may not be capturing data."
        )
        source_label = "Microphone (loopback failed)"

    return audio, source_label


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
