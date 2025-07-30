from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_CACHE: Dict[str, Any] | None = None


def load_config(path: str | Path = "config.json") -> Dict[str, Any]:
    """Load JSON config file."""
    cfg_path = Path(path)
    print(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------- ffmpeg helpers -----------------


def parse_ffmpeg_encoders(ffmpeg_bin: str = "ffmpeg") -> set[str]:
    """Return a set of encoder names that ffmpeg reports as available."""
    import subprocess, re

    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    lines = proc.stdout.splitlines()
    encoders: set[str] = set()
    pattern = re.compile(r"^[\sA-Z\.]+ ([\w\-]+) ")
    for line in lines:
        m = pattern.match(line)
        if m:
            encoders.add(m.group(1))

    # fallback: if encoder later查询时未在列表，可用此函数 encoder_available(name)
    return encoders


def encoder_available(name: str, ffmpeg_bin: str = "ffmpeg", debug: bool = False) -> bool:
    """Return True if ffmpeg recognizes encoder. If debug, print stderr when unavailable."""
    import subprocess, re
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-h", f"encoder={name}"],
        capture_output=True,
        text=True,
    )
    not_rec = re.compile(rf"Codec '.+' is not recognized by FFmpeg", re.I)
    err = proc.stdout.strip()
    avail = not bool(not_rec.search(err))
    if debug:
        print(f"[ffmpeg stderr] encoder={name}:\n{err}\n")
    return avail
