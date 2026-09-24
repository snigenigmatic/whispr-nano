# Whispr BRIDGE study: Hugging Face GPU execution

This is the portable GPU execution component of C. Kaustubh's open-model ASR
audit. It preserves the prior Modal experiment tree and its Mac-specific run
command. Source lineage is recorded in PROVENANCE.json. This is an engineering
development study, not an exact reproduction of the commercial BRIDGE audit.

Every HF experiment runs `bash experiments/hf_entrypoint.sh` in the same pinned
PyTorch image on one A100. Changes to the method live in committed configuration.
The first baseline repeats Bengali conversation humyn-01-01 with the existing
native Whisper Turbo recipe. This checks the provider/runtime transition before
testing compression-triggered temperature fallback as a child experiment.

The user authorized a separate $50 HF ceiling. `experiments/launch_hf.py` reserves
$1.50 per 30-minute A100 job and calls ORX. The current provider rate is $0.041667
per minute; the reservation includes headroom. Reservations are not released
based on runtime estimates. Modal's separate $15 ledger is not changed.

The runner stages the exact frozen public WAV and reference files, verifies
their hashes against the frozen manifest, pins model weights, and
records raw generated sequences and seek traces. Artifacts are compressed into
hash-checked log chunks so ORX retains them after the ephemeral machine stops.
Recovered predictions must pass the independent local validator before use.

Data: humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR, revision
2e4879d569bd57f675c227a41fecf4a2e56f467c, CC-BY-4.0. The exposed development
conversation must not be presented as a held-out benchmark or fairness result.
