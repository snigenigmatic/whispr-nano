---
pretty_name: L2-ARCTIC v5.0
license: cc-by-nc-4.0
task_categories:
- automatic-speech-recognition
- audio-classification
language:
- en
tags:
- speech
- pronunciation
- phoneme
- l2
- mispronunciation-detection
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: "*.zip"
---

# L2-ARCTIC v5.0

L2-ARCTIC is a non-native English speech corpus intended for research in
pronunciation assessment, accent conversion, voice conversion, and
mispronunciation detection.

This Hub dataset mirrors the `v5.0` release as distributed locally:
speaker-level zip archives, the suitcase corpus archive, prompts, the
original README, and the license.

## Summary

- 24 non-native English speakers
- 26,867 utterances
- 27.1 hours of scripted speech
- Manual phone-level annotations for 3,599 utterances
- Additional `suitcase_corpus` spontaneous-speech subset
- Native language groups: Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese

## Structure

Each speaker archive contains:

- `wav/`: 44.1 kHz WAV audio
- `transcript/`: orthographic transcripts
- `textgrid/`: forced-aligned phoneme transcriptions
- `annotation/`: manual pronunciation annotations where available

## License

This dataset is distributed under `CC BY-NC 4.0` as provided in the
original release. Redistribution is for non-commercial use only.

## Citation

```bibtex
@inproceedings{zhao2018l2arctic,
  author={Guanlong {Zhao} and Sinem {Sonsaat} and Alif {Silpachai}
          and Ivana {Lucic} and Evgeny {Chukharev-Hudilainen}
          and John {Levis} and Ricardo {Gutierrez-Osuna}},
  title={L2-ARCTIC: A Non-native English Speech Corpus},
  year=2018,
  booktitle={Proc. Interspeech},
  pages={2783--2787},
  doi={10.21437/Interspeech.2018-1110}
}
```

## Notes

- This card is a Hub-specific wrapper around the original release metadata.
- Refer to `README.original.md` for the full upstream release notes.
- The original `LICENSE` file is included unchanged.
