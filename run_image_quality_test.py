#!/usr/bin/env python3
"""
Standalone image quality test runner
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from EncoderBenchmark.image_quality_test import main

if __name__ == "__main__":
    main() 