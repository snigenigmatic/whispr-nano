#!/usr/bin/env bash
set -euo pipefail
# Pinned worker dependencies, including the runtime image's torch version.
python -m pip install --disable-pip-version-check \
  'torch==2.8.0' \
  'transformers==4.57.6' 'numpy==2.2.6' 'scipy==1.15.3' 'soundfile==0.13.1' \
  'jiwer==4.0.0' 'huggingface-hub==0.36.2' 'pyarrow==21.0.0' \
  'qwen-asr==0.0.6' 'accelerate==1.12.0'
python experiments/hf_run.py
