#!/usr/bin/env bash
set -euo pipefail
# Fixed on every HF node; model and decoding changes live in committed JSON.
python -m pip install --disable-pip-version-check 'transformers==4.57.6' 'numpy==2.2.6' 'scipy==1.15.3' 'soundfile==0.13.1' 'jiwer==4.0.0' 'huggingface-hub==0.36.2' 'pyarrow==21.0.0'
python experiments/hf_run.py
