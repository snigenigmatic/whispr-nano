"""Maps Svarah's 19 accent labels to 3 Indian language families.

============================================================================
DISCLAIMER: This mapping was independently reconstructed from public
linguistic classification for a feasibility check. It has not been verified
against the source project's authoritative mapping file and MUST be
cross-checked before use in any publication-track output.
============================================================================

Provenance of the label set
----------------------------
The public `ai4bharat/svarah` dataset on Hugging Face (gated; requires an
authenticated `huggingface` Modal secret / HF token to load) stores the
accent label in a column called **`primary_language`** -- confirmed by
directly loading the dataset and inspecting `dataset.column_names` plus
sample rows on 2026-08-28 (see `audit/opd_bias_extension/modal_svarah_audit.py`'s pre-flight
schema check). This is the speaker's self-reported native/mother-tongue
language, which the Svarah paper (Javed et al., "Svarah: Evaluating English
ASR Systems on Indian Accents," Interspeech 2023, arXiv:2305.15760) uses as
the accent identity for its per-accent breakdown (Table 2 of the paper,
using ISO 639 codes). The full dataset (`split="test"`, the only split
Svarah ships) contains 6,656 utterances and exactly 19 unique
`primary_language` values, which line up 1:1 with the paper's Table 2 accent
codes:

    ne=Nepali, brx=Bodo, as=Assamese, doi=Dogri, sd=Sindhi, ml=Malayalam,
    kn=Kannada, mr=Marathi, bn=Bengali, pa=Punjabi, ur=Urdu, te=Telugu,
    ks=Kashmiri, kok=Konkani, or=Odia, guj=Gujarati, hi=Hindi, mai=Maithili,
    ta=Tamil

The Svarah paper's own text states its 19 languages belong to "4 different
language families" (Section 1) -- notably *not* 3. The most likely 4th
family, beyond Indo-Aryan / Dravidian / Sino-Tibetan, is **Dardic**
(Kashmiri's actual sub-branch of Indo-Iranian, distinct from Indo-Aryan in
strict genetic classification, e.g. Glottolog/recent Ethnologue editions).
Since this task requires exactly 3 buckets {indo_aryan, dravidian,
sino_tibetan} (matching the prior capstone project's scheme), Kashmiri is
folded into `indo_aryan` below -- a standard coarse-grained simplification
(older Ethnologue editions and most non-specialist treatments group Dardic
under the Indo-Aryan umbrella).

Composition check against the prior project's reported numbers
-----------------------------------------------------------------
This reconstruction yields:

    indo_aryan:    14 languages  (Nepali, Assamese, Dogri, Sindhi, Marathi,
                   Bengali, Punjabi, Urdu, Kashmiri, Konkani, Odia,
                   Gujarati, Hindi, Maithili)
    dravidian:      4 languages  (Malayalam, Kannada, Telugu, Tamil)
    sino_tibetan:   1 language   (Bodo)
    ---------------------------------------------------------------
    total:         19 languages

The task brief states the prior (unseen) capstone project reported "15 IA /
4 Dravidian / 1 ST" -- note that 15+4+1=20, which is already inconsistent
with Svarah's 19 total accents, so that figure appears to be an approximate
recollection rather than an exact count. Our reconstruction (14/4/1=19) is
"close to" that recollection and matches it exactly on the two most
confident buckets (Dravidian=4, Sino-Tibetan=1, with the ST language being
Bodo, exactly as expected). The one-off discrepancy on Indo-Aryan (14 vs.
"15") is most plausibly explained by whether Kashmiri (Dardic) is folded
into Indo-Aryan (our choice, giving 14) or kept as the Svarah paper's
implicit 4th family (which would give 13 "pure" Indo-Aryan languages plus a
separate 1-language Dardic bucket) -- neither reading produces exactly 15,
so this is flagged explicitly as a judgment call requiring cross-check
against the source project's authoritative file, per the disclaimer above.

Family assignment rationale (standard genetic/areal classification)
---------------------------------------------------------------------
- Indo-Aryan (Indo-European > Indo-Iranian > Indo-Aryan): the majority
  language family of northern, western, central, and eastern India,
  descended from Sanskrit-derived Prakrits. All the Hindi-belt, Bihari,
  Rajasthani-adjacent, Gujarati/Marathi (western), Punjabi (northwestern),
  Bengali/Odia/Assamese/Maithili (eastern) languages fall here, per
  standard classification.
- Dravidian: the major South Indian family (Tamil, Telugu, Kannada,
  Malayalam are its four literary/scheduled languages), genetically
  unrelated to Indo-Aryan.
- Sino-Tibetan: Northeast Indian languages (Bodo, Manipuri/Meitei, several
  Naga languages, Mizo, Khasi-type -- though Khasi itself is actually
  Austroasiatic, a common point of confusion) trace to the Sino-Tibetan
  family via the Tibeto-Burman branch. Bodo (spoken mainly in Assam) is
  Tibeto-Burman / Sino-Tibetan and is Svarah's sole representative here.
"""

from __future__ import annotations

INDO_ARYAN = "indo_aryan"
DRAVIDIAN = "dravidian"
SINO_TIBETAN = "sino_tibetan"

# Keys are the exact strings found in the `primary_language` column of
# `ai4bharat/svarah` (split="test"). Values are one of the three family
# constants above.
SVARAH_LANGUAGE_TO_FAMILY: dict[str, str] = {
    # --- Indo-Aryan (14) -----------------------------------------------
    "Hindi": INDO_ARYAN,        # Hindi belt (Indo-Aryan core)
    "Marathi": INDO_ARYAN,      # Western Indo-Aryan
    "Gujarati": INDO_ARYAN,     # Western Indo-Aryan
    "Punjabi": INDO_ARYAN,      # Northwestern Indo-Aryan
    "Bengali": INDO_ARYAN,      # Eastern Indo-Aryan
    "Odia": INDO_ARYAN,         # Eastern Indo-Aryan
    "Assamese": INDO_ARYAN,     # Eastern Indo-Aryan (northeast, but the
                                # language itself is Indo-Aryan, not
                                # Sino-Tibetan -- do not confuse with Bodo)
    "Maithili": INDO_ARYAN,     # Bihari group, Eastern Indo-Aryan
    "Nepali": INDO_ARYAN,       # Northern Indo-Aryan (Pahari group)
    "Konkani": INDO_ARYAN,      # Southern Indo-Aryan (coastal Goa/Konkan)
    "Sindhi": INDO_ARYAN,       # Northwestern Indo-Aryan
    "Dogri": INDO_ARYAN,        # Northwestern Indo-Aryan (Jammu region)
    "Urdu": INDO_ARYAN,         # Hindustani register, Indo-Aryan
    "Kashmiri": INDO_ARYAN,     # Judgment call: genetically Dardic (a
                                # distinct Indo-Iranian branch, sister to
                                # Indo-Aryan), folded into Indo-Aryan here
                                # per the common coarse-grained convention
                                # -- see the module docstring's composition
                                # discussion. FLAG: verify against the
                                # source project's treatment of Kashmiri.
    # --- Dravidian (4) ---------------------------------------------------
    "Tamil": DRAVIDIAN,
    "Telugu": DRAVIDIAN,
    "Kannada": DRAVIDIAN,
    "Malayalam": DRAVIDIAN,
    # --- Sino-Tibetan (1) --------------------------------------------------
    "Bodo": SINO_TIBETAN,       # Tibeto-Burman, Assam (northeast India)
}

# ISO 639 codes used in the Svarah paper's Table 2, included for
# cross-reference/traceability back to the paper.
SVARAH_LANGUAGE_TO_ISO639 = {
    "Nepali": "ne", "Bodo": "brx", "Assamese": "as", "Dogri": "doi",
    "Sindhi": "sd", "Malayalam": "ml", "Kannada": "kn", "Marathi": "mr",
    "Bengali": "bn", "Punjabi": "pa", "Urdu": "ur", "Telugu": "te",
    "Kashmiri": "ks", "Konkani": "kok", "Odia": "or", "Gujarati": "guj",
    "Hindi": "hi", "Maithili": "mai", "Tamil": "ta",
}

FAMILIES = (INDO_ARYAN, DRAVIDIAN, SINO_TIBETAN)


def get_family(primary_language: str) -> str:
    """Look up the language family for a Svarah `primary_language` value.

    Raises KeyError with the full known-label list if `primary_language`
    isn't one of the 19 values this mapping was built from -- fail loudly
    rather than silently mis-bucketing an utterance, since the whole point
    of this script is a trustworthy per-family breakdown.
    """
    try:
        return SVARAH_LANGUAGE_TO_FAMILY[primary_language]
    except KeyError:
        raise KeyError(
            f"Unmapped Svarah primary_language {primary_language!r}. "
            f"Known labels ({len(SVARAH_LANGUAGE_TO_FAMILY)}): "
            f"{sorted(SVARAH_LANGUAGE_TO_FAMILY)}"
        ) from None


def composition_summary() -> dict[str, int]:
    """Count of languages per family -- used as a sanity check at import
    time and printed by the audit script's smoke entrypoint."""
    counts = {f: 0 for f in FAMILIES}
    for family in SVARAH_LANGUAGE_TO_FAMILY.values():
        counts[family] += 1
    return counts


if __name__ == "__main__":
    counts = composition_summary()
    print(f"{len(SVARAH_LANGUAGE_TO_FAMILY)} languages mapped:")
    for family in FAMILIES:
        print(f"  {family}: {counts[family]}")
    assert sum(counts.values()) == 19, "expected exactly 19 Svarah languages"
