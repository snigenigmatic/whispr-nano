# Listening lab

Two-reviewer dual transcription of the frozen 120-clip Indian-English set.
Blind to the Svarah reference and the whisper-large-v3 teacher until both
reviewers finish.

```bash
cd listen
npm install
npm run dev -- --port 43147 --hostname 127.0.0.1
```

Open [http://127.0.0.1:43147](http://127.0.0.1:43147). Choose Reviewer A or B,
transcribe, export JSON. Reconciliation imports both exports, then optionally
loads `data/research_reset/listening_set.json` to unseal reference and teacher
transcripts.

Audio wavs are not in git (Svarah is gated):

```bash
PYTHONPATH=src python -m research_reset.fetch_audio --out listen/public/audio
```
