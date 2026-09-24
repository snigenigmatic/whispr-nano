# Independent beam-state review

The fallback run has 31 native windows and 176 attempts. Window 0 at temperature 0 is the sole beam 5 attempt. The other 175 attempts, including the first temperature 0 attempt of every later window, use beam 1.

Transformers 4.57.6 `WhisperGenerationMixin.generate_with_fallback` mutates the shared per-call generation configuration. Sampling sets num_beams=1; returning to temperature 0 resets do_sample but does not restore the beam count. The same configuration is reused across native windows, although the outer generate call initially deep-copies the model configuration.

This is an implementation quirk evidenced by both code and the recorded trace. The completed comparison evaluates the whole native fallback recipe, including this state change. It does not isolate temperature fallback at a constant beam width. No code fix or restoration experiment has been run.
