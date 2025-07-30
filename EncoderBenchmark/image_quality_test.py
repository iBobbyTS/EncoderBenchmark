"""
Image quality test module for encoder benchmark
Tests quality parameters to find closest VMAF to target values
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from .encoder_runner import EncoderRunner, EncoderTask
from .utils import encoder_available, load_config


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


def run_image_quality_test(general_config: Dict[str, Any], quality_config: Dict[str, Any], 
                          dry_run: bool = False, debug_encoder_check: bool = False) -> None:
    """
    Run image quality test (Step3) for all encoders and target VMAFs
    
    Args:
        general_config: General configuration
        quality_config: Image quality test configuration
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
        done_cache = {"step3": set()}
        result_file = out_root / f"{video.stem}_step3.jsonl"
        if result_file.exists():
            with open(result_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("step") == "step3":
                        key = (data["encoder"], data.get("vmaf_target"))
                        done_cache["step3"].add(key)
        return done_cache
    
    def _append_result(cfg: Dict[str, Any], src_name: str, step: str, task: EncoderTask, metrics: Dict[str, Any]):
        result_file = out_root / f"{src_name}_{step}.jsonl"
        result = {
            "video": src_name,
            "encoder": task.encoder,
            "preset": task.preset,
            "param": task.qparam_name,
            "q": task.qvalue,
            "step": step,
            **metrics
        }
        with open(result_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    # Get encoder filter
    run_limit = general_config.get("run_encoders", [])
    debug_enc = debug_encoder_check
    
    # Get quality test configuration
    targets = quality_config.get("target_vmafs", [90, 95, 99])
    qcontrols = quality_config["quality_controls"]
    
    # Run quality tests
    for enc_name, qcfg in qcontrols.items():
        if run_limit and enc_name.lower() not in run_limit:
            continue
        if not encoder_available(enc_name, runner.ffmpeg, debug=debug_enc):
            print(f"[Skip] encoder {enc_name} not available")
            continue
            
        for video in videos:
            # 检查哪些target已经完成
            video_step3_done = load_done(video)["step3"]
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
                _append_result(general_config, video.name, "step3", task,
                               metrics | {"vmaf_target": tv})


def main():
    """Main function for image quality test"""
    # Load configurations
    general_config = load_config("config/general.json")
    quality_config = load_config("config/image_quality_test.json")
    
    # Get settings from general config
    dry_run = general_config.get("dry_run", False)
    debug_encoder_check = general_config.get("debug_encoder_check", False)
    
    print("=== Image Quality Test (Step3) ===")
    run_image_quality_test(general_config, quality_config, dry_run, debug_encoder_check)


if __name__ == "__main__":
    main() 