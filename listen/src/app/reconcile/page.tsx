"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ClipPlayer } from "@/components/clip-player";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { downloadJson, loadBundle } from "@/lib/storage";
import { LABELS, type BlindClip, type ReconcileLabel, type ReviewerBundle } from "@/lib/types";

type SealedClip = BlindClip & {
  reference_text?: string;
  teacher_text?: string;
  accepted?: boolean;
};

type Decision = {
  utt_id: string;
  agreed_transcript: string;
  label: ReconcileLabel | "";
  notes: string;
};

export default function ReconcilePage() {
  const [clips, setClips] = useState<BlindClip[]>([]);
  const [bundleA, setBundleA] = useState<ReviewerBundle | null>(null);
  const [bundleB, setBundleB] = useState<ReviewerBundle | null>(null);
  const [sealed, setSealed] = useState<Record<string, SealedClip> | null>(null);
  const [index, setIndex] = useState(0);
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [importError, setImportError] = useState<string | null>(null);

  useEffect(() => {
    setBundleA(loadBundle("A"));
    setBundleB(loadBundle("B"));
    fetch("/listening_set.blind.json")
      .then((r) => r.json())
      .then(setClips);
  }, []);

  function importReviewer(which: "A" | "B", file: File) {
    file.text().then((text) => {
      try {
        const parsed = JSON.parse(text) as ReviewerBundle;
        if (which === "A") setBundleA(parsed);
        else setBundleB(parsed);
        setImportError(null);
      } catch {
        setImportError("That file is not a reviewer JSON export.");
      }
    });
  }

  function importSealed(file: File) {
    file.text().then((text) => {
      try {
        const parsed = JSON.parse(text) as SealedClip[];
        const map: Record<string, SealedClip> = {};
        for (const row of parsed) map[row.utt_id] = row;
        setSealed(map);
        setImportError(null);
      } catch {
        setImportError("Sealed file must be listening_set.json.");
      }
    });
  }

  const clip = clips[index];
  const aText = clip ? bundleA?.transcripts[clip.utt_id]?.text ?? "" : "";
  const bText = clip ? bundleB?.transcripts[clip.utt_id]?.text ?? "" : "";
  const bothReady = Boolean(aText.trim() && bText.trim());
  const decision = clip
    ? decisions[clip.utt_id] ?? {
        utt_id: clip.utt_id,
        agreed_transcript: aText === bText ? aText : "",
        label: "" as const,
        notes: "",
      }
    : null;

  const labeled = useMemo(
    () => Object.values(decisions).filter((d) => d.label).length,
    [decisions],
  );

  function writeDecision(patch: Partial<Decision>) {
    if (!clip) return;
    setDecisions((prev) => ({
      ...prev,
      [clip.utt_id]: {
        utt_id: clip.utt_id,
        agreed_transcript: decision?.agreed_transcript ?? "",
        label: decision?.label ?? "",
        notes: decision?.notes ?? "",
        ...patch,
      },
    }));
  }

  const counts = useMemo(() => {
    const tally: Record<string, number> = {};
    for (const d of Object.values(decisions)) {
      if (d.label) tally[d.label] = (tally[d.label] ?? 0) + 1;
    }
    return tally;
  }, [decisions]);

  const rejectedSealed = sealed
    ? Object.values(sealed).filter((c) => c.accepted === false)
    : [];
  const badRefs = rejectedSealed.length
    ? Object.values(decisions).filter(
        (d) =>
          (d.label === "reference_mistake" || d.label === "both_wrong") &&
          sealed?.[d.utt_id]?.accepted === false,
      ).length
    : 0;
  const repairTrigger =
    rejectedSealed.length > 0 && badRefs / rejectedSealed.length > 0.15;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Reconciliation</h1>
        <p className="text-muted-foreground max-w-2xl">
          Sit together. Compare A and B first. Only then load the sealed file so
          you can see the reference and the teacher. Label every disagreement
          before anyone trains.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Reviewer A</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm text-muted-foreground">
              {bundleA?.name || "unnamed"} ·{" "}
              {bundleA ? Object.keys(bundleA.transcripts).length : 0} saved
            </p>
            <Label className="text-xs">Import A JSON from another machine</Label>
            <input
              type="file"
              accept="application/json"
              onChange={(e) => e.target.files?.[0] && importReviewer("A", e.target.files[0])}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Reviewer B</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm text-muted-foreground">
              {bundleB?.name || "unnamed"} ·{" "}
              {bundleB ? Object.keys(bundleB.transcripts).length : 0} saved
            </p>
            <Label className="text-xs">Import B JSON from another machine</Label>
            <input
              type="file"
              accept="application/json"
              onChange={(e) => e.target.files?.[0] && importReviewer("B", e.target.files[0])}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Sealed transcripts</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm text-muted-foreground">
              {sealed ? "Loaded. Reference and teacher are visible." : "Not loaded."}
            </p>
            <Label className="text-xs">Load data/research_reset/listening_set.json</Label>
            <input
              type="file"
              accept="application/json"
              onChange={(e) => e.target.files?.[0] && importSealed(e.target.files[0])}
            />
          </CardContent>
        </Card>
      </div>

      {importError ? (
        <Alert variant="destructive">
          <AlertTitle>Could not import</AlertTitle>
          <AlertDescription>{importError}</AlertDescription>
        </Alert>
      ) : null}

      {repairTrigger ? (
        <Alert variant="destructive">
          <AlertTitle>Stop before training</AlertTitle>
          <AlertDescription>
            More than 15% of the rejected cell is labelled reference_mistake or
            both_wrong ({badRefs} / {rejectedSealed.length}). Repair or drop bad
            references before the 12-run gate.
          </AlertDescription>
        </Alert>
      ) : null}

      {clip && decision ? (
        <Card>
          <CardHeader className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary">{clip.primary_language}</Badge>
              <span className="text-sm text-muted-foreground">
                {clip.utt_id} · clip {clip.clip_index} · {labeled} labelled
              </span>
            </div>
            <ClipPlayer uttId={clip.utt_id} />
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Reviewer A
                </p>
                <p className="rounded-md bg-muted p-3 text-sm whitespace-pre-wrap min-h-16">
                  {aText || "— empty —"}
                </p>
              </div>
              <div>
                <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Reviewer B
                </p>
                <p className="rounded-md bg-muted p-3 text-sm whitespace-pre-wrap min-h-16">
                  {bText || "— empty —"}
                </p>
              </div>
            </div>
            {aText && bText && aText.trim() === bText.trim() ? (
              <p className="text-sm">A and B already match.</p>
            ) : (
              <p className="text-sm text-muted-foreground">
                {bothReady
                  ? "Reviewers disagree. Replay, then write an agreed transcript."
                  : "Both reviewers need a transcript before you label this clip."}
              </p>
            )}

            {sealed?.[clip.utt_id] ? (
              <>
                <Separator />
                <div className="grid gap-3 sm:grid-cols-2">
                  <div>
                    <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Reference (sealed)
                    </p>
                    <p className="rounded-md border p-3 text-sm whitespace-pre-wrap">
                      {sealed[clip.utt_id].reference_text}
                    </p>
                  </div>
                  <div>
                    <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Teacher, large-v3 (sealed)
                    </p>
                    <p className="rounded-md border p-3 text-sm whitespace-pre-wrap">
                      {sealed[clip.utt_id].teacher_text}
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                Reference and teacher stay hidden until you load the sealed file.
              </p>
            )}

            <div className="space-y-1.5">
              <Label>Agreed transcript</Label>
              <Textarea
                value={decision.agreed_transcript}
                onChange={(e) => writeDecision({ agreed_transcript: e.target.value })}
                className="min-h-24"
              />
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {LABELS.map((lab) => (
                <Button
                  key={lab.id}
                  type="button"
                  variant={decision.label === lab.id ? "default" : "outline"}
                  className="h-auto justify-start whitespace-normal py-3 text-left"
                  onClick={() => writeDecision({ label: lab.id })}
                  disabled={!bothReady}
                >
                  <span>
                    <span className="block font-medium">{lab.title}</span>
                    <span className="block text-xs opacity-80">{lab.hint}</span>
                  </span>
                </Button>
              ))}
            </div>
            <Textarea
              value={decision.notes}
              onChange={(e) => writeDecision({ notes: e.target.value })}
              placeholder="Optional note"
            />
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setIndex(Math.max(0, index - 1))} disabled={index === 0}>
                Back
              </Button>
              <Button
                onClick={() => setIndex(Math.min(clips.length - 1, index + 1))}
                className="ml-auto"
              >
                Next clip
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : null}

      <div className="flex flex-wrap gap-2">
        {Object.entries(counts).map(([k, v]) => (
          <Badge key={k} variant="secondary">
            {k}: {v}
          </Badge>
        ))}
      </div>

      <Button
        variant="outline"
        onClick={() =>
          downloadJson("reconciliation.json", {
            reviewer_a: bundleA?.name,
            reviewer_b: bundleB?.name,
            protocol_version: "2026-09-09",
            clips: Object.values(decisions),
          })
        }
      >
        Export reconciliation.json
      </Button>
      <p className="text-sm text-muted-foreground">
        Drop the export at <code className="font-mono text-xs">data/research_reset/reconciliation.json</code>{" "}
        before anyone sets <code className="font-mono text-xs">RESEARCH_RESET_LAUNCH=yes</code>.
      </p>
      <Button asChild variant="ghost">
        <Link href="/review">Back to review</Link>
      </Button>
    </div>
  );
}
