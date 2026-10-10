# Night 1 public-corpus data card

[OBSERVED] Generated from `scripts/termeval/probe_data.py` with seed 20261011. Exact machine-readable counts are in `data_summary.json`; full transcripts and audio stay under ignored `data/`.

## MUCS Hindi-English test

[OBSERVED] Source: [OpenSLR SLR104](https://www.openslr.org/104/), CC BY-SA 4.0. The downloaded `Hindi-English_test.tar.gz` has SHA-256 `93e358b3bf8233a897fcd353c1f4f98fdda6b8c01b7eed17a70c7dd26e984b37`.
[OBSERVED] Layout: `test/*.wav` plus `test/transcripts/{text,segments,utt2spk,spk2utt,wav.scp,spkr_list}`; 30 long recordings, 3136 sentence segments, 30 speaker IDs. The source is 16-kHz audio; segments were cut into mono 16-kHz clips.
[OBSERVED] Segment hours 5.178; duration min/median/95th/max 1.00/5.00/12.00/41.00 seconds. The `segments` file is required to turn long recordings into utterances.
[OBSERVED] Normalized reference token scripts: DEV: 28,164 (75.1%), LAT: 9,095 (24.3%), MIX: 55 (0.1%), NUM: 186 (0.5%).
[OBSERVED] Speaker and recording IDs are present. `utt2spk` is used for cluster bootstrap, with 30 clusters.

### 20 seed-selected MUCS transcript lines

1. निम्न link पर उपलब्ध video देखें यह spoken tutorial को सारांशित करता है
2. options टैब पर click करें
3. अब अपने document में columns प्रविष्ट करें
4. प्रत्येक भाषा के लिए spell check विशेषता भिन्न होती है
5. हम देख सकते हैं कि window का name भी sample dot c में बदल गया है
6. आप प्रिव्यू के लिए एक से भी अधिक animation चुन सकते हैं
7. सभी चीजों याने ऑब्जेक्ट्स के की पोइंट्स भी दिखाई देते है
8. हम निम्न भी सीखेंगे
9. यह दिशा इंगित करता है जिसमें control point moved आकार में हेरफेर करने के लिए स्थानांतरित किया जा सकता है
10. अट्रिब्यूट पैनल में से  text just बटन पर क्लिक करे सेंटर अलाइन्मन्ट याने मध्य संरेखण को चुनते है पहले बॉक्स के सेंटर में क्लिक करते है ओह मैंने सही जगह नहीं चुनी है कर्सर को हटाने के लिए मैं दूसरी जगह पर क्लिक करती हूँ फिर मैं सही जगह पर क्लिक करुँगी
11. इस ट्यूटोरियल में क्लिकिंग याने  लेफ्ट माउस बटन क्लिक करके छोड़ना उसी तरह किसी बटन को चुनना याने आपको उस पर लेफ्ट माउस बटन से क्लिक करना होगा अगर कुछ और ज़रुरी है तो मैं बता दूंगी
12. इस व्यापक नियत कार्य की कोशिश करें
13. अब हम कुछ plugins को सक्षम करेंगे
14. यहाँ आप के लिए एक नियतकार्य है
15. slide 1 चुनें
16. अतः चलिए एक उदाहरण के साथ देखते हैं कि वे कार्य कैसे करते हैं
17. इस तरह document में लिखे गये विविध प्वॉइंट्स के बीच भेद कर सकते हैं
18. कंटेंट का font 36 और color blue रखें
19. एक निमंत्रण पत्र type करें
20. कॉन्टेक्स्ट मेन्यू देखने के लिए दायाँ क्लिक करें और delete पर क्लिक करें

## HiACC

[OBSERVED] Source: [Zenodo record 15551669](https://zenodo.org/records/15551669), CC BY 4.0. Downloaded `Corpus.zip` MD5 `dd6cc9354e1dee5e2f25bc5243df88ac` and SHA-256 `fd83414b3aba28a4ae86c110f4aab0981621f9f3a1b105c161bf2dfd14737ccf` both verified.
[OBSERVED] 5176 mono 16-kHz utterance WAVs from 44 speaker IDs, 5.274 hours by WAV header duration. Normalized reference token scripts: DEV: 30,694 (60.6%), LAT: 19,977 (39.4%), MIX: 2 (0.0%), OTHER: 1 (0.0%).
[OBSERVED] Folder structure: `Corpus/{adult,children}/audio/{train,val,test}_split/`, transcripts, `metadata/{speaker_info.csv,sentence_stats.csv}`, and `annotations/code_switched_labels.json`.
[OBSERVED] The annotation JSON has one row per utterance with an utterance-level `label` (`none`, `intra-sential`, `inter-sential`); the released JSON does not contain word-level language tags, despite the bundled README describing word-level annotations.
[OBSERVED] The bundled README calls the splits speaker-disjoint. Every adult speaker ID and every child speaker ID occurs in each of train, validation, and test. These splits are therefore **not speaker-disjoint** in the downloaded archive; do not claim unseen-speaker test performance from them.
[OBSERVED] `AD63` appears in 119 adult WAV/transcript rows but not in `adult/metadata/speaker_info.csv`; `AD65` has a speaker-info row but no WAV. Overall ASR scoring retains AD63. Analyses by age/gender/L1 must exclude its missing values and state the reduced support.

### Adult

[OBSERVED] 3318 utterances, 24 audio speaker IDs, 3.217 hours; split utterance counts {'test': 664, 'train': 2322, 'val': 332}; speaker overlap counts {'train_val': 24, 'train_test': 24, 'val_test': 24}.
[OBSERVED] Metadata columns: `speaker_info.csv` ['PID', 'Gender', 'Age', 'L1'] (24 rows); `sentence_stats.csv` ['audio', 'sentence', 'total_words', 'num_english_words', 'num_hindi_words', 'CMI', 'duration_sec', 'WPM', 'speech_rate', 'code_switch_count', 'hi_to_en_switch', 'en_to_hi_switch'] (3318 rows); `code_switched_labels.json` ['audio', 'label', 'transcription'] (3318 rows).
[OBSERVED] Speaker metadata distributions: {'Gender': {'F': 10, 'M': 14}, 'Age': {'19': 1, '22': 3, '25': 2, '27': 1, '29': 1, '30': 1, '31': 2, '32': 3, '34': 1, '35': 2, '36': 3, '38': 3, '42': 1}, 'L1': {'Hindi': 24}}. Annotation labels: {'inter-sential': 113, 'intra-sential': 1095, 'none': 2110}.
[OBSERVED] Normalized reference token scripts: DEV: 16,196 (49.7%), LAT: 16,395 (50.3%), MIX: 2 (0.0%).

#### 20 seed-selected adult transcript lines

1. probably
2. the बसंत पंचमी
3. lunch time था तो मां बाप ने पकड़ के जबरदस्ती बिठा दिया
4. खूब सारा खाना बनाया था fruits लेके आए थे
5. and
6. मैं हूं गुफा का guardian।
7. all time favourite series of TV shows
8. face
9. rainy season में तो ऐसा लगता है कि हम clouds के ऊपर आ गए हैं।
10. mountains है उधर peak है
11. I can see various other things like
12. the secret of the cave began to unfold before them
13. ये गुफा उन दोनों के लिए है जो दोस्ती में प्यार में और साथ में हमेश साथ रहते हैं
14. तो NCR है तो there is no
15. किसी को airport पर लेने आया है शायद और वो
16. दोस्ती और साथ में रहना ही असली ख़ुशी है। मिलते हैं अगली कहानी में तब तक अपना ख्याल रखो और दोस्ती को हमेशा
17. Literally I enjoyed a lot because I am a water person
18. and one is crying
19. It's the modern family. Now I remember the name
20. दोस्ती और साथ में रहना ही असली ख़ुशी है। मिलते हैं अगली कहानी में

### Children

[OBSERVED] 1858 utterances, 20 audio speaker IDs, 2.057 hours; split utterance counts {'test': 372, 'train': 1300, 'val': 186}; speaker overlap counts {'train_val': 20, 'train_test': 20, 'val_test': 20}.
[OBSERVED] Metadata columns: `speaker_info.csv` ['PID', 'Gender', 'Age', 'L1'] (20 rows); `sentence_stats.csv` ['audio', 'sentence', 'path', 'total_words', 'num_english_words', 'num_hindi_words', 'CMI', 'duration_sec', 'WPM', 'speech_rate', 'code_switch_count', 'hi_to_en_switch', 'en_to_hi_switch'] (186 rows); `code_switched_labels.json` ['audio_filepath', 'label', 'transcription'] (1858 rows).
[OBSERVED] Speaker metadata distributions: {'Gender': {'F': 10, 'M': 10}, 'Age': {'10': 1, '11': 4, '12': 6, '13': 4, '14': 5}, 'L1': {'Garhwali': 2, 'Hindi': 17, 'Nepali': 1}}. Annotation labels: {'inter-sential': 32, 'intra-sential': 1092, 'none': 734}.
[OBSERVED] Normalized reference token scripts: DEV: 14,498 (80.2%), LAT: 3,582 (19.8%), OTHER: 1 (0.0%).

#### 20 seed-selected children transcript lines

1. वीर और दिया ने एक छुपी cave खोजने की यातरा पर जाने का निर्णय लिया
2. अपने bag में कुछ snack और water pack
3. असली magic उनके दिलों में होता है और भाई-बहन का bond सबसे strong magic है
4. जहाँ पे मैं रहती हु वहांपे
5. symbols को समझने में बिताए
6. तो हम वो देखने जाते हैं गर्मियों की छुट्टियों में अच्छा लगता है वहां पे जाना मेरी favourite place है juddo
7. तो फिर वो उसके बाद उसपे मारपीट पे
8. purple,yellow,brwon और वहां पे एक आदमी भी है वहां पे एक oven है
9. लेकिन उन्होने हिम्मत नहीं हारी
10. वो सब पे इतना अत्याचार करता है मतलब hospitals में
11. की अटारी की खोज कर रहे थे,उन्हें एक dust old book मिली
12. in the heart of a city दो sibling रहते थे
13. Secret थोड़ा और समझ में आने लगा धीरे धीरे
14. जैसे ही वे forest से गुजर रहे थे उन्हें लगा जैसे कोई उन्हें देख रहा है
15.  recommend something from your place to everyone.
16. Deciphering the symbols
17. हमेशा पास,हमेशा साथ। एक दिन,Ram और Shyam ने सुना कि उनके पास के forest में एक cave है। यह सुनकर,दोनों ने सोचा कि
18. With bated breath,they explored the old house,discovering a hidden passage
19. abandoned house खोज निकाला।
20. मंदिर भी है और

## Access checks on remaining planned sources

[OBSERVED] [Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja) exposes public metadata but its Parquet content returned HTTP 401 without a locally available accepted-access token. No Lahaja audio, rows, group counts, or transcripts were inspected; it is excluded from this run.
[OBSERVED] [VITB-HEBiC](https://huggingface.co/datasets/PalashJain/VITB-HEBiC_Code-switch_corpus) returned HTTP 401 for its dataset API and file tree. No claims from its paper/card were verified against files; it is excluded from this run.

## Scope limits

[ASSUMED] All released HiACC splits are pooled for a corpus feasibility audit. The original split label is retained in the local manifest so split-specific scores can be reported. The pooled result must not be framed as a held-out speaker benchmark.
[OBSERVED] MUCS speaker IDs and HiACC filename prefixes are available for speaker-cluster bootstrap; demographic metadata is unavailable for MUCS.
[UNVERIFIED] Model training-data overlap with either corpus is unknown.
