from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from EncoderBenchmark.encoder_runner import EncoderRunner, EncoderTask
from EncoderBenchmark.utils import ensure_dir, load_config, encoder_available


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video encoder benchmark tool")
    p.add_argument("--config", default="config.json", help="Config JSON path")
    return p.parse_args()


def iterate_source_videos(src_dir: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".mkv", ".y4m", ".yuv"}
    return sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = load_config(args.config)

    runner = EncoderRunner(cfg)
    dry_run = bool(cfg.get("dry_run", False))
    debug_enc = bool(cfg.get("debug_encoder_check", False))
    runner.dry_run = dry_run

    src_dir = Path(cfg["source_videos_dir"])
    out_root = ensure_dir(cfg["output_root"])

    videos = iterate_source_videos(src_dir)
    if not videos:
        print(f"No source videos found in {src_dir}")
        sys.exit(1)

    # preload existing results into a set for fast skip
    def load_done(video: Path):
        rec_file = out_root / f"{video.name}.jsonl"
        step1_done = set()
        step3_done = set()
        if rec_file.exists():
            for line in rec_file.read_text().splitlines():
                rec = json.loads(line)
                step = rec.get("step")
                if step == "step1":
                    key = (rec.get("step"), rec.get("encoder"), rec.get("preset"), rec.get("param"), rec.get("q"))
                    step1_done.add(key)
                elif step == "step3":
                    # Step3的唯一标识是 {encoder, vmaf_target}
                    key = (rec.get("encoder"), rec.get("vmaf_target"))
                    step3_done.add(key)
        return {"step1": step1_done, "step3": step3_done}

    done_cache: dict[str, dict] = {v.name: load_done(v) for v in videos}

    def is_done(video: Path, step: str, task: EncoderTask):
        if step == "step1":
            key = (step, task.encoder, task.preset, task.qparam_name, task.qvalue)
            return key in done_cache[video.name]["step1"]
        return False  # Step3 uses different logic

    # ---- Step-1 presets test ----
    step1_preset_cfg = cfg["step1"]["presets"]
    run_limit = set(e.lower() for e in cfg.get("run_encoders", []) if e)
    for enc_name, rule in step1_preset_cfg.items():
        if run_limit and enc_name.lower() not in run_limit:
            continue
        if not encoder_available(enc_name, runner.ffmpeg, debug=debug_enc):
            print(f"[Skip] encoder {enc_name} not available")
            continue
        method = rule["method"]
        if method == "preset":
            param_name = rule.get("name", "preset")
            for preset_val in rule["values"]:
                for video in videos:
                    out_name = f"{video.stem}_{enc_name}_{param_name}_{preset_val}.mp4"
                    task = EncoderTask(
                        src=video,
                        dst=out_root / out_name,
                        encoder=enc_name,
                        preset=None if param_name != "preset" else str(preset_val),
                        qparam_name=None if param_name == "preset" else param_name,
                        qvalue=preset_val if param_name != "preset" else None,
                        extra_args=rule.get("additional_params"),
                    )
                    if is_done(video, "step1", task):
                        continue
                    metrics = runner.run(task)
                    if not dry_run:
                        _append_result(cfg, video.name, "step1", task, metrics)
        elif method == "parm":
            for plist in rule["params"]:
                for video in videos:
                    suffix = "default" if not plist else "_".join([p.strip('-') for p in plist])
                    out_name = f"{video.stem}_{enc_name}_mode_{suffix}.mp4"

                    base_args: list[str] = list(rule.get("additional_params", []))
                    if plist:
                        base_args.extend(plist)
                    task = EncoderTask(
                        src=video,
                        dst=out_root / out_name,
                        encoder=enc_name,
                        extra_args=base_args or None,
                    )
                    # 无 preset / quality param 的纯附加参数模式
                    if plist:
                        task.preset = None
                        task.qparam_name = None
                    if is_done(video, "step1", task):
                        continue
                    metrics = runner.run(task)
                    if not dry_run:
                        _append_result(cfg, video.name, "step1", task, metrics)

    # ---- Step-3 quality search ----
    step3_section = cfg["step3"]
    targets = step3_section.get("target_vmafs", [90, 95, 99])
    qcontrols = step3_section["quality_controls"]

    for enc_name, qcfg in qcontrols.items():
        if run_limit and enc_name.lower() not in run_limit:
            continue
        if not encoder_available(enc_name, runner.ffmpeg, debug=debug_enc):
            continue
        for video in videos:
            # 检查哪些target已经完成
            video_step3_done = done_cache[video.name]["step3"]
            remaining_targets = [t for t in targets if (enc_name, t) not in video_step3_done]
            
            if not remaining_targets:
                print(f"[Skip] {video.name} + {enc_name} all targets already completed")
                continue
                
            best_dict = _search_multi_targets(runner, video, out_root,
                                              enc_name, qcfg, remaining_targets, dry_run)
            if dry_run:
                continue
            # write results
            for tv, (task, metrics) in best_dict.items():
                _append_result(cfg, video.name, "step3", task,
                               metrics | {"vmaf_target": tv})


def _append_result(cfg: Dict[str, Any], src_name: str, step: str, task: EncoderTask, metrics: Dict[str, Any]):
    out_root = Path(cfg["output_root"])
    ensure_dir(out_root)
    record_file = out_root / f"{src_name}.jsonl"
    row = {
        "video": src_name,
        "encoder": task.encoder,
        "preset": task.preset,
        "param": task.qparam_name,
        "q": task.qvalue,
        "step": step,
        **metrics,
    }
    with record_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


# ---------------- search helper -----------------


def _search_quality(runner: EncoderRunner, video: Path, out_root: Path, enc_name: str, qcfg: dict, target_vmaf: float, dry: bool):
    """Linear search around initial_quality to get closest vmaf."""
    min_q = qcfg["min_quality"]
    max_q = qcfg["max_quality"]
    cur_q = qcfg["initial_quality"]
    param_name = qcfg["quality_param"]
    higher_is_better = qcfg["higher_quality_when"] == "value_lower"

    best_delta = 999.0
    best_quality = None
    best_metrics = {}

    step = 2  # coarse step first
    tried = set()
    direction = 0

    while min_q <= cur_q <= max_q and cur_q not in tried:
        tried.add(cur_q)
        out_name = f"{video.stem}_{enc_name}_{param_name}_{cur_q}.mp4"
        task = EncoderTask(src=video, dst=out_root / out_name, encoder=enc_name,
                           qparam_name=param_name, qvalue=cur_q,
                           extra_args=qcfg.get("additional_params"))
        if dry:
            # Dry-run: build command then exit after first iteration
            runner.run(task)  # will only print command
            return None, {}

        metrics = runner.run(task)
        delta = abs(metrics["vmaf"] - target_vmaf)
        if delta < best_delta:
            best_delta = delta
            best_quality = cur_q
            best_metrics = metrics | {"task": task, "target_vmaf": target_vmaf, "delta": delta}

        # decide next q
        if metrics["vmaf"] < target_vmaf:
            # need better quality
            cur_q = cur_q - step if higher_is_better else cur_q + step
        elif metrics["vmaf"] > target_vmaf:
            # need worse quality
            cur_q = cur_q + step if higher_is_better else cur_q - step
        else:
            break

        if step > 1 and len(tried) >= 6:
            # switch to fine step
            step = 1

    return best_quality, best_metrics

# ---------------- new multi target search -----------------


def _search_multi_targets(runner: EncoderRunner, video: Path, out_root: Path, enc_name: str,
                          qcfg: dict, targets: list[float], dry: bool):
    """New search algorithm: start from worst quality, step by 5, refine when crossing targets."""
    min_q = qcfg["min_quality"]
    max_q = qcfg["max_quality"]
    higher_is_better = qcfg["higher_quality_when"] == "value_higher"
    
    # 1. 从低画质开始
    start_q = min_q if higher_is_better else max_q
    step_direction = 5 if higher_is_better else -5
    
    # 2. 粗扫阶段：从低画质向高画质走
    remaining_targets = targets.copy()  # [90, 95, 99]
    vmaf_results = {}  # {quality_param: (task, metrics)}
    
    cur_q = start_q
    while remaining_targets:
        # 边界处理
        if higher_is_better and cur_q > max_q:
            cur_q = max_q
            refine_step_back = cur_q - (cur_q - 5)  # 用于细扫起点
        elif not higher_is_better and cur_q < min_q:
            cur_q = min_q  
            refine_step_back = cur_q - (cur_q + 5)
        else:
            refine_step_back = 5
        
        # Step3不再单独跳过质量参数测试，因为我们需要完整的搜索过程
            
        # 编码并计算VMAF
        out_name = f"{video.stem}_{enc_name}_{qcfg['quality_param']}_{cur_q}.mp4"
        task = EncoderTask(src=video,
                           dst=out_root / out_name,
                           encoder=enc_name,
                           qparam_name=qcfg["quality_param"],
                           qvalue=cur_q,
                           extra_args=qcfg.get("additional_params"))
        if dry:
            runner.run(task)
            return {}
            
        metrics = runner.run(task)
        vmaf = metrics["vmaf"]
        vmaf_results[cur_q] = (task, metrics)
        
        # 检查是否超过当前最低目标
        if vmaf >= remaining_targets[0]:
            target_reached = remaining_targets.pop(0)  # 移除已达到的目标
            
            # 往回细扫：从上一个粗扫点的下一步开始到当前点的前一步
            prev_coarse_q = cur_q - step_direction  # 上一个粗扫点
            
            if higher_is_better:
                # 对于 higher_is_better，从 prev_coarse_q + 1 到 cur_q - 1
                refine_start = prev_coarse_q + 1
                refine_end = cur_q - 1
                refine_range = range(max(min_q, refine_start), min(max_q, refine_end) + 1)
            else:
                # 对于 value_lower，从 prev_coarse_q - 1 到 cur_q + 1
                refine_start = prev_coarse_q - 1
                refine_end = cur_q + 1
                refine_range = range(min(max_q, refine_start), max(min_q, refine_end) - 1, -1)
                
            for refine_q in refine_range:
                if refine_q in vmaf_results:
                    continue
                    
                r_name = f"{video.stem}_{enc_name}_{qcfg['quality_param']}_{refine_q}.mp4"
                r_task = EncoderTask(src=video,
                                     dst=out_root / r_name,
                                     encoder=enc_name,
                                     qparam_name=qcfg["quality_param"],
                                     qvalue=refine_q,
                                     extra_args=qcfg.get("additional_params"))
                if dry:
                    runner.run(r_task)
                    continue
                    
                r_metrics = runner.run(r_task)
                vmaf_results[refine_q] = (r_task, r_metrics)
        
        # 到边界就结束
        if (higher_is_better and cur_q >= max_q) or (not higher_is_better and cur_q <= min_q):
            break
            
        cur_q += step_direction
    
    # 3. 后处理：为每个目标找最接近的quality_param
    sorted_results = sorted(vmaf_results.items(), 
                           key=lambda x: x[0], 
                           reverse=not higher_is_better)
    
    final_results = {}
    targets_copy = targets.copy()  # [90, 95, 99]
    
    for q_param, (task, metrics) in sorted_results:
        if not targets_copy:
            break
        target = targets_copy[0]
        vmaf = metrics["vmaf"]
        
        if vmaf >= target:
            final_results[target] = (task, metrics)
            targets_copy.pop(0)
    
    # 处理未找到的目标：使用最高质量的参数
    if targets_copy and vmaf_results:
        # 找到最高质量的参数（按VMAF排序）
        best_quality_result = max(vmaf_results.items(), key=lambda x: x[1][1]["vmaf"])
        best_q_param, (best_task, best_metrics) = best_quality_result
        
        for remaining_target in targets_copy:
            print(f"[Warning] Target VMAF {remaining_target} not reached, using best quality q={best_q_param} vmaf={best_metrics['vmaf']:.2f}")
            final_results[remaining_target] = (best_task, best_metrics)
    
    return final_results 