"""On-policy distillation of whisper-tiny from whisper-large-v3.

Core, Modal-agnostic training/eval logic lives in this package; `modal_app.py`
at the repo root is a thin wrapper that runs it on Modal's CUDA GPUs.
"""

__version__ = "0.1.0"
