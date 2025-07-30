"""
Speed test module for encoder benchmark
Tests different presets/speeds for each encoder
"""

import json
from pathlib import Path
from typing import Dict, Any
from EncoderBenchmark.encoder_runner import EncoderRunner, EncoderTask
from EncoderBenchmark.utils import encoder_available, load_config, find_usable_pixfmts


def run_speed_test(general_config: Dict[str, Any], speed_config: Dict[str, Any], 
                   dry_run: bool = False, debug_encoder_check: bool = False) -> None:
    """
    Run speed test (Step1) for all encoders and presets
    
    Args:
        general_config: General configuration
        speed_config: Speed test configuration
        dry_run: If True, only print commands without execution
        debug_encoder_check: If True, print debug info for encoder availability check
    """
    # Initialize encoder runner
    runner = EncoderRunner(general_config)
    runner.dry_run = dry_run
    
    # Get source videos
    source_dir = Path(general_config["source_videos_dir"])
    videos = list(source_dir.glob("*.mov")) + list(source_dir.glob("*.mp4")) + list(source_dir.glob("*.mkv")) + \
             list(source_dir.glob("*.y4m")) + list(source_dir.glob("*.yuv"))
    
    if not videos:
        print(f"No source videos found in {source_dir}")
        return
    
    # Get output directory
    out_root = Path(general_config["output_root"])
    out_root.mkdir(exist_ok=True)
    
    # Load completed tasks
    def load_done(video: Path):
        done_cache = {"step1": set()}
        result_file = out_root / f"{video.stem}_step1.jsonl"
        if result_file.exists():
            with open(result_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("step") == "step1":
                        key = (data["encoder"], data.get("preset"), data.get("param"), data.get("q"), data.get("pix_fmt"))
                        done_cache["step1"].add(key)
        return done_cache
    
    def is_done(video: Path, step: str, task: EncoderTask):
        done_cache = load_done(video)
        if step == "step1":
            key = (task.encoder, task.preset, task.qparam_name, task.qvalue, task.pix_fmt)
            return key in done_cache["step1"]
        return False
    
    def _append_result(cfg: Dict[str, Any], src_name: str, step: str, task: EncoderTask, metrics: Dict[str, Any]):
        result_file = out_root / f"{src_name}_{step}.jsonl"
        result = {
            "video": src_name,
            "encoder": task.encoder,
            "preset": task.preset,
            "param": task.qparam_name,
            "q": task.qvalue,
            "step": step,
            "pix_fmt": task.pix_fmt,
            **metrics
        }
        with open(result_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    # Get encoder filter
    run_limit = general_config.get("run_encoders", [])
    debug_enc = debug_encoder_check
    
    # Get presets configuration
    presets = speed_config["presets"]
    
    # Run speed tests
    for enc_name, rule in presets.items():
        if run_limit and enc_name.lower() not in run_limit:
            continue
        if not encoder_available(enc_name, runner.ffmpeg, debug=debug_enc):
            print(f"[Skip] encoder {enc_name} not available")
            continue
        for video in videos:
            target_pix_fmts = find_usable_pixfmts(video, enc_name)
            for pix_fmt in target_pix_fmts:
                for preset in rule["values"]:
                    if rule["method"] == "preset":
                        out_name = f"{video.stem}_{enc_name}_{rule['name']}_{preset}_{pix_fmt}.mp4"
                        task = EncoderTask(
                            src=video,
                            dst=out_root / out_name,
                            pix_fmt=pix_fmt,
                            encoder=enc_name,
                            preset=['-'+rule['name'], preset],
                            extra_args=rule.get("additional_params", [])
                        )
                    elif rule["method"] == "parm":
                        out_name = f"{video.stem}_{enc_name}_{'-'.join(preset)}_{pix_fmt}.mp4"
                        task = EncoderTask(
                            src=video,
                            dst=out_root / out_name,
                            pix_fmt=pix_fmt,
                            encoder=enc_name,
                            extra_args=preset + rule.get("additional_params", [])
                        )
                    else:
                        print(f"[Skip] Unsupported method {rule['method']} for encoder {enc_name}")
                        continue
                    if is_done(video, "step1", task):
                        continue
                    metrics = runner.run(task)
                    if not dry_run:
                        _append_result(general_config, video.name, "step1", task, metrics)


def main():
    """Main function for speed test"""
    # Load configurations
    general_config = load_config("config/general.json")
    speed_config = load_config("config/speed_test.json")
    
    # Get settings from general config
    dry_run = general_config.get("dry_run", False)
    debug_encoder_check = general_config.get("debug_encoder_check", False)
    
    print("=== Speed Test (Step1) ===")
    run_speed_test(general_config, speed_config, dry_run, debug_encoder_check)


if __name__ == "__main__":
    main()
