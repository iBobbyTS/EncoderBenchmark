from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


CONFIG_CACHE: Dict[str, Any] | None = None
general_config = json.load(open('config/general.json', 'r'))


def _probe_pixfmt(file: Path) -> str:
    cmd = [
        Path(general_config['ffmpeg_path']) / "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=pix_fmt",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(file),
    ]
    res = _run_subprocess(cmd)
    pix = res.stdout.strip()
    return pix

def load_config(path: str | Path = "config.json") -> Dict[str, Any]:
    """Load JSON config file."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _run_subprocess(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    """Wrapper for subprocess.run with common flags."""
    return subprocess.run(cmd, capture_output=True, text=True, check=False)



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


def find_usable_pixfmts(video, enc_name):
    encoder_supported_pix_fmts = general_config.get("avaliable_pixfmt", {}).get(enc_name, [])
    pix_fmt = _probe_pixfmt(video)
    target_pix_fmts = general_config.get("pix_fmt_downsample", {}).get(pix_fmt, [])
    target_pix_fmts = [fmt for fmt in target_pix_fmts if fmt in encoder_supported_pix_fmts]
    return target_pix_fmts
