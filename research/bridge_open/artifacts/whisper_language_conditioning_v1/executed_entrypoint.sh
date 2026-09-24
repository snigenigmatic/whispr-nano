#!/usr/bin/env bash
set -euo pipefail
python -m pip install --disable-pip-version-check 'torch==2.8.0' 'transformers==4.57.6' 'numpy==2.2.6' 'scipy==1.15.3' 'soundfile==0.13.1' 'jiwer==4.0.0' 'huggingface-hub==0.36.2' 'openai-whisper==20250625' 'numba==0.61.2' 'tiktoken==0.9.0' 'more-itertools==10.7.0'
python experiments/audit.py
