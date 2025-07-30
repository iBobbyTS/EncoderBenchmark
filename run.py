#!/usr/bin/env python3
"""
New entry point for encoder benchmark with separated config files
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from EncoderBenchmark.benchmark import main

if __name__ == "__main__":
    # Override sys.argv to use the new config path
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "config/general.json"]
    
    main() 