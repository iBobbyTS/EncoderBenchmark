"""
Main benchmark module for encoder benchmark
Orchestrates speed test and image quality test
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List
from .utils import load_config
from .speed_test import run_speed_test
from .image_quality_test import run_image_quality_test


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Encoder Benchmark")
    parser.add_argument("config", help="Path to general config file")
    return parser.parse_args()


def main() -> None:
    """Main benchmark function"""
    args = parse_args()
    
    # Load configurations
    general_config = load_config(args.config)
    speed_config = load_config("config/speed_test.json")
    quality_config = load_config("config/image_quality_test.json")
    
    # Get settings from general config
    dry_run = general_config.get("dry_run", False)
    debug_encoder_check = general_config.get("debug_encoder_check", False)
    
    print("=== Encoder Benchmark ===")
    
    # Run speed test (Step1)
    print("\n--- Speed Test (Step1) ---")
    run_speed_test(general_config, speed_config, dry_run, debug_encoder_check)
    
    # Run image quality test (Step3)
    print("\n--- Image Quality Test (Step3) ---")
    run_image_quality_test(general_config, quality_config, dry_run, debug_encoder_check)
    
    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main() 