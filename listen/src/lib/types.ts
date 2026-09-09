export type ReviewerId = "A" | "B";

export type BlindClip = {
  utt_id: string;
  svarah_index: number;
  primary_language: string;
  family: string;
  gender: string | null;
  duration_s: number;
  clip_index: number;
};

export type ClipTranscript = {
  text: string;
  notes: string;
  updatedAt: string;
};

export type ReviewerBundle = {
  reviewer: ReviewerId;
  name: string;
  protocolVersion: string;
  transcripts: Record<string, ClipTranscript>;
};

export type ReconcileLabel =
  | "teacher_mistake"
  | "reference_mistake"
  | "both_wrong"
  | "normalization"
  | "genuine_variation"
  | "other";

export const LABELS: { id: ReconcileLabel; title: string; hint: string }[] = [
  { id: "teacher_mistake", title: "Teacher mistake", hint: "Audio matches the reference, not the teacher." },
  { id: "reference_mistake", title: "Reference mistake", hint: "Audio matches the teacher, not the reference." },
  { id: "both_wrong", title: "Both wrong", hint: "Neither transcript matches what was said." },
  { id: "normalization", title: "Normalization", hint: "Same words; punctuation, casing, numerals, spelling." },
  { id: "genuine_variation", title: "Genuine variation", hint: "Both are plausible hearings of the same audio." },
  { id: "other", title: "Other", hint: "Write a note." },
];

export const PROTOCOL_VERSION = "2026-09-09";
export const STORAGE_KEYS = {
  active: "listenlab:activeReviewer",
  name: (id: ReviewerId) => `listenlab:name:${id}`,
  bundle: (id: ReviewerId) => `listenlab:bundle:${id}`,
} as const;
