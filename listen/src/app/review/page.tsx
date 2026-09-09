"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ClipPlayer } from "@/components/clip-player";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import {
  completedCount,
  downloadJson,
  loadActiveReviewer,
  loadBundle,
  saveBundle,
} from "@/lib/storage";
import type { BlindClip, ReviewerBundle, ReviewerId } from "@/lib/types";

export default function ReviewPage() {
  const [clips, setClips] = useState<BlindClip[]>([]);
  const [reviewer, setReviewer] = useState<ReviewerId | null>(null);
  const [bundle, setBundle] = useState<ReviewerBundle | null>(null);
  const [index, setIndex] = useState(0);
  const [text, setText] = useState("");
  const [notes, setNotes] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const id = loadActiveReviewer();
    setReviewer(id);
    if (id) setBundle(loadBundle(id));
    fetch("/listening_set.blind.json")
      .then((r) => {
        if (!r.ok) throw new Error("blind clip list missing");
        return r.json();
      })
      .then((data: BlindClip[]) => {
        setClips(data);
      })
      .catch((err: Error) => setLoadError(err.message));
  }, []);

  const clip = clips[index];
  const done = bundle ? completedCount(bundle) : 0;
  const incomplete = useMemo(() => {
    if (!bundle) return [];
    return clips
      .map((c, i) => ({ i, id: c.utt_id }))
      .filter(({ id }) => !bundle.transcripts[id]?.text.trim());
  }, [bundle, clips]);

  useEffect(() => {
    if (!clip || !bundle) return;
    const existing = bundle.transcripts[clip.utt_id];
    setText(existing?.text ?? "");
    setNotes(existing?.notes ?? "");
  }, [clip, bundle]);

  const persist = useCallback(
    (nextIndex: number) => {
      if (!bundle || !clip) return;
      const next: ReviewerBundle = {
        ...bundle,
        transcripts: {
          ...bundle.transcripts,
          [clip.utt_id]: {
            text: text.trim(),
            notes: notes.trim(),
            updatedAt: new Date().toISOString(),
          },
        },
      };
      saveBundle(next);
      setBundle(next);
      setIndex(Math.min(Math.max(nextIndex, 0), Math.max(clips.length - 1, 0)));
    },
    [bundle, clip, clips.length, notes, text],
  );

  if (loadError) {
    return <p className="text-destructive">{loadError}</p>;
  }
  if (!reviewer || !bundle) {
    return (
      <div className="space-y-3">
        <p>Choose Reviewer A or B on the lab home page first.</p>
        <Button asChild>
          <Link href="/">Go to lab home</Link>
        </Button>
      </div>
    );
  }
  if (!clip) {
    return <p className="text-muted-foreground">Loading the 120-clip list…</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm text-muted-foreground">
            Reviewer {reviewer}
            {bundle.name ? ` · ${bundle.name}` : ""}
          </p>
          <h1 className="text-2xl font-semibold tracking-tight">
            Clip {clip.clip_index} of {clips.length}
          </h1>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            onClick={() =>
              downloadJson(`reviewer_${reviewer.toLowerCase()}_transcripts.json`, bundle)
            }
          >
            Export JSON
          </Button>
          <Button variant="ghost" asChild>
            <Link href="/reconcile">Reconcile</Link>
          </Button>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm text-muted-foreground">
          <span>{done} typed</span>
          <span>{incomplete.length} remaining</span>
        </div>
        <Progress value={(done / 120) * 100} />
      </div>

      <Card>
        <CardHeader className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">{clip.primary_language}</Badge>
            <Badge variant="outline">{clip.family}</Badge>
            {clip.gender ? <Badge variant="outline">{clip.gender}</Badge> : null}
            <span className="text-sm text-muted-foreground">
              {clip.duration_s.toFixed(1)} s · {clip.utt_id}
            </span>
          </div>
          <CardTitle className="text-base font-medium">Play, then type what you hear</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <ClipPlayer uttId={clip.utt_id} />
          <div className="space-y-1.5">
            <Label htmlFor="transcript">Transcript</Label>
            <Textarea
              id="transcript"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type the words. Keep disfluencies. Use [inaudible] or [uncertain: guess] when needed."
              className="min-h-36 text-base"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="notes">Private notes (not a transcript)</Label>
            <Textarea
              id="notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Noise, talker overlap, anything that made the clip hard — not your best-guess words."
              className="min-h-20"
            />
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Button variant="outline" onClick={() => persist(index - 1)} disabled={index === 0}>
              Back
            </Button>
            <Button onClick={() => persist(index + 1)} className="sm:ml-auto">
              Save and next
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Reference and teacher transcripts are not on this page. If you can see them, stop and
            tell the other reviewer — the seal is broken.
          </p>
        </CardContent>
      </Card>

      <div className="flex flex-wrap gap-2">
        <Button
          variant="secondary"
          disabled={incomplete.length === 0}
          onClick={() => incomplete[0] && persist(incomplete[0].i)}
        >
          Jump to next empty
        </Button>
        <p className="self-center text-sm text-muted-foreground">
          {incomplete.length === 0
            ? "Every clip has text. Export JSON and wait for the other reviewer."
            : "Empty clips stay in worksheet order."}
        </p>
      </div>
    </div>
  );
}
