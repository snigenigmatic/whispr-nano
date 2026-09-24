#!/usr/bin/env bash
set -euo pipefail
# Fixed ORX/HF command: bash experiments/hf_entrypoint.sh
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git libsndfile1 build-essential cmake
python -m pip install --disable-pip-version-check \
  'torch==2.8.0' 'torchaudio==2.8.0' 'fairseq2==0.6.0' 'fairseq2n==0.6' \
  'numpy==1.26.4' 'scipy==1.15.3' 'soundfile==0.13.1' 'jiwer==4.0.0' \
  'huggingface-hub==0.36.2' \
  'omnilingual-asr @ git+https://github.com/facebookresearch/omnilingual-asr.git@81f51e224ce9e74b02cc2a3eaf21b2d91d743455'
python experiments/hf_run.py
