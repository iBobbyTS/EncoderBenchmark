from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from EncoderBenchmark.utils import ensure_dir, _run_subprocess


@dataclass
class EncoderTask:
    src: Path
    dst: Path
    encoder: str
    pix_fmt: str
    preset: Optional[list[str]] = None
    qparam_name: Optional[str] = None
    qvalue: Optional[int] = None
    extra_args: Optional[list[str]] = None  # for custom parameters


class EncoderRunner:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        raw_path: str = cfg.get("ffmpeg_path", "") or ""
        # If raw_path is empty -> use binaries found in PATH; else treat as directory and append binary names.
        if raw_path == "":
            self.ffmpeg = "ffmpeg"
            self.ffprobe = "ffprobe"
        else:
            self.ffmpeg = str(Path(raw_path) / "ffmpeg")
            self.ffprobe = str(Path(raw_path) / "ffprobe")
        self.threads = cfg.get("threads", 0)
        self.thread_rules: dict[str, list[str]] = cfg.get("threading_rules", {})
        self.dry_run = False  # 外部设置

    # ---------------- public -----------------
    def run(self, task: EncoderTask) -> Dict[str, float | str]:
        """Run encoding + VMAF, return metrics dict."""
        cmd = self.build_cmd(task)
        print("CMD:", " ".join(cmd))
        if self.dry_run:
            return {}

        start = time.perf_counter()
        proc = _run_subprocess(cmd)
        elapsed = time.perf_counter() - start

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr}\nCmd: {' '.join(cmd)}")

        # TODO: parse bitrate from ffmpeg output or probe output file
        bitrate_kbps = self._extract_bitrate(task.dst)  # placeholder

        # Compute VMAF
        vmaf = self._calc_vmaf(task)
        # PSNR removed as per latest requirements

        result = {
            "elapsed": elapsed,
            "bitrate_kbps": bitrate_kbps,
            "vmaf": vmaf,
        }

        print(f"RESULT | enc={task.encoder} {task.qparam_name or 'preset'}={task.qvalue or task.preset} vmaf={vmaf:.6f}")
        return result

    # ---------------- internal -----------------
    def build_cmd(self, task: EncoderTask) -> List[str]:
        """Assemble ffmpeg command based on task description."""
        out_dir = ensure_dir(task.dst.parent)
        quality_args: List[str] = []
        if task.qparam_name and task.qvalue is not None:
            quality_args = [f"-{task.qparam_name}", str(task.qvalue)]
        cmd = [
            self.ffmpeg,
            "-y",  # overwrite
            "-i", str(task.src),
            "-pix_fmt", task.pix_fmt,  # set pixel format
            "-map", "v",  # only map video streams, skip audio
            "-c:v", task.encoder,
            *(task.preset if task.preset is not None else []),  # add preset if provided
            *quality_args,
        ]

        # threading parameters
        if task.encoder in self.thread_rules:
            t_val = self.threads
            if task.encoder == "libaom-av1":
                t_val = min(t_val, 8)
            custom = [self.thread_rules[task.encoder][0], self.thread_rules[task.encoder][1].format(threads=t_val)]
            cmd += custom

        cmd.append(str(out_dir / task.dst.name))

        if task.extra_args:
            # insert before output file path (last element)
            # we added output at end; place extra before it
            cmd = cmd[:-1] + task.extra_args + cmd[-1:]

        return cmd

    # ---------------- bitrate & vmaf -----------------
    def _extract_bitrate(self, outfile: Path) -> float:
        """Use ffprobe to read stream bit_rate (kbps)."""
        cmd = [
            self.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(outfile),
        ]
        proc = _run_subprocess(cmd)
        if proc.returncode == 0 and proc.stdout.strip().isdigit():
            return round(int(proc.stdout.strip()) / 1000, 2)
        # fallback: compute by file size / duration
        size_bytes = outfile.stat().st_size
        duration = self._probe_duration(outfile)
        if duration > 0:
            return round((size_bytes * 8) / 1000 / duration, 2)
        return 0.0

    def _probe_duration(self, file: Path) -> float:
        cmd = [
            self.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(file),
        ]
        proc = _run_subprocess(cmd)
        try:
            return float(proc.stdout.strip())
        except ValueError:
            return 0.0

    def _calc_vmaf(self, task: EncoderTask) -> float:
        """Run ffmpeg libvmaf comparing encoded file to source, return VMAF score."""
        cfg_model = self.cfg.get("vmaf_model", "")
        t = self.threads
        base = f"n_threads={t}"
        if cfg_model:
            filter_opts = f"{base}:{cfg_model}"
        else:
            filter_opts = base

        # probe dst pixel format
        pix_fmt = self._probe_pixfmt(task.dst)
        fmt_convert = f"[0:v]format=pix_fmts={pix_fmt}[ref];[ref][1:v]libvmaf={filter_opts}"

        filter_str = fmt_convert

        cmd = [
            self.ffmpeg,
            "-i",
            str(task.src),
            "-i",
            str(task.dst),
            "-lavfi",
            filter_str,
            "-f",
            "null",
            "-",
        ]
        print("CMD:", " ".join(cmd))
        proc = _run_subprocess(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"VMAF calculation failed: {proc.stderr}")

        import re
        m = re.search(r"VMAF score[:=]\s*([0-9\.]+)", proc.stderr)
        if m:
            return float(m.group(1))

        # failed to parse vmaf
        raise RuntimeError("Failed to extract VMAF score. FFmpeg output:\n" + proc.stderr)

    def _probe_pixfmt(self, file: Path) -> str:
        cmd = [
            self.ffprobe,
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
