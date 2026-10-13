#!/usr/bin/env bash
set -euo pipefail

# Fixed command for the portable Lahaja cloud tree. Experiment changes live in
# committed Python/config files, never in this command or the job environment.
python -m pip install --disable-pip-version-check \
  'transformers==5.16.1' 'pyarrow==25.0.1' 'soundfile==0.14.0' \
  'jiwer==4.0.0' 'numpy==2.5.2' 'huggingface-hub==1.29.0'
python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; print("cuda_gpu=" + torch.cuda.get_device_name(0), flush=True)'
python experiments/publication_run.py
