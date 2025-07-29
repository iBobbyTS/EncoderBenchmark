import json
from pathlib import Path
from EncoderBenchmark.utils import load_config
from EncoderBenchmark.encoder_runner import EncoderRunner, EncoderTask


def main():
    cfg = load_config()
    runner = EncoderRunner(cfg)
    runner.dry_run = False  # must execute

    results_dir = Path(cfg["output_root"])
    src_dir = Path(cfg["source_videos_dir"])

    for jfile in results_dir.glob("*.jsonl"):
        updated = []
        changed = False
        for line in jfile.read_text().splitlines():
            rec = json.loads(line)
            if "psnr" in rec:
                updated.append(rec)
                continue
            enc_path = results_dir / rec["encoder"]  # wrong; we saved filename only
            enc_path = results_dir.parent / rec.get("step", "?")  # skip complexity
            vid_name = rec["video"]
            src = src_dir / vid_name
            dst = Path(rec_file_path(rec, results_dir))
            if not dst.exists():
                updated.append(rec)
                continue
            task = EncoderTask(src=src, dst=dst, encoder=rec["encoder"])
            try:
                rec["psnr"] = runner._calc_psnr(task)
                changed = True
            except Exception as e:
                print("PSNR failed", dst, e)
            updated.append(rec)
        if changed:
            with jfile.open("w", encoding="utf-8") as f:
                for r in updated:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rec_file_path(rec, results_dir):
    # reconstruct file name consistent with earlier naming
    if rec.get("step") == "step2" and rec.get("param") == "crf":
        return results_dir / f"{Path(rec['video']).stem}_{rec['encoder']}_crf{rec['q']}.mp4"
    if rec.get("step") == "step1":
        if rec.get("param"):
            return results_dir / f"{Path(rec['video']).stem}_{rec['encoder']}_{rec['param']}_{rec['q']}.mp4"
        elif rec.get("preset") is None:
            return results_dir / f"{Path(rec['video']).stem}_{rec['encoder']}_mode_{rec['q']}.mp4"
        else:
            return results_dir / f"{Path(rec['video']).stem}_{rec['encoder']}_preset_{rec['preset']}.mp4"
    if rec.get("step") == "step3":
        return results_dir / f"{Path(rec['video']).stem}_{rec['encoder']}_{rec['param']}_{rec['q']}.mp4"
    return ""


if __name__ == "__main__":
    main() 