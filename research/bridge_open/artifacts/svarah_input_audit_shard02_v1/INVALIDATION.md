# Invalidated: do not use for inference

The v1 limited-shard bundle has a reference-binding defect. In partial mode it
recorded the Arrow shard-local scan offset as `dataset_index`; bundle creation
then read `prior[dataset_index]`, which can bind an utterance audio/UID to a
different global Svarah reference. This affects every v1 reference binding and
its derived manifest/config hashes.

No GPU inference was launched from v1. Preserve v1 for diagnosis only. Use
`svarah_native_shard02_v2` and `svarah_input_audit_shard02_v2` only after their
global metadata ID, source basename, and reference SHA binding checks pass.
