"use client";

import { PROTOCOL_VERSION, STORAGE_KEYS, type ReviewerBundle, type ReviewerId } from "./types";

export function emptyBundle(reviewer: ReviewerId, name = ""): ReviewerBundle {
  return {
    reviewer,
    name,
    protocolVersion: PROTOCOL_VERSION,
    transcripts: {},
  };
}

export function loadBundle(reviewer: ReviewerId): ReviewerBundle {
  if (typeof window === "undefined") return emptyBundle(reviewer);
  const raw = localStorage.getItem(STORAGE_KEYS.bundle(reviewer));
  if (!raw) return emptyBundle(reviewer, localStorage.getItem(STORAGE_KEYS.name(reviewer)) ?? "");
  try {
    const parsed = JSON.parse(raw) as ReviewerBundle;
    return {
      ...emptyBundle(reviewer),
      ...parsed,
      reviewer,
      transcripts: parsed.transcripts ?? {},
    };
  } catch {
    return emptyBundle(reviewer);
  }
}

export function saveBundle(bundle: ReviewerBundle) {
  localStorage.setItem(STORAGE_KEYS.bundle(bundle.reviewer), JSON.stringify(bundle));
  if (bundle.name) localStorage.setItem(STORAGE_KEYS.name(bundle.reviewer), bundle.name);
}

export function loadActiveReviewer(): ReviewerId | null {
  if (typeof window === "undefined") return null;
  const v = localStorage.getItem(STORAGE_KEYS.active);
  return v === "A" || v === "B" ? v : null;
}

export function setActiveReviewer(id: ReviewerId) {
  localStorage.setItem(STORAGE_KEYS.active, id);
}

export function completedCount(bundle: ReviewerBundle) {
  return Object.values(bundle.transcripts).filter((t) => t.text.trim().length > 0).length;
}

export function downloadJson(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
