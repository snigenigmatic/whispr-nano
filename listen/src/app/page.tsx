"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { completedCount, loadActiveReviewer, loadBundle, saveBundle, setActiveReviewer } from "@/lib/storage";
import type { ReviewerId } from "@/lib/types";

export default function HomePage() {
  const [reviewer, setReviewer] = useState<ReviewerId | null>(null);
  const [nameA, setNameA] = useState("");
  const [nameB, setNameB] = useState("");
  const [doneA, setDoneA] = useState(0);
  const [doneB, setDoneB] = useState(0);

  useEffect(() => {
    const a = loadBundle("A");
    const b = loadBundle("B");
    setNameA(a.name);
    setNameB(b.name);
    setDoneA(completedCount(a));
    setDoneB(completedCount(b));
    setReviewer(loadActiveReviewer());
  }, []);

  function choose(id: ReviewerId, name: string) {
    const bundle = loadBundle(id);
    bundle.name = name.trim();
    saveBundle(bundle);
    setActiveReviewer(id);
    setReviewer(id);
  }

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Two reviewers, 120 recordings, no peeking.
        </h1>
        <p className="max-w-2xl text-muted-foreground text-base leading-relaxed">
          Before we train on teacher transcripts, we need to know what the
          disagreements actually are: teacher mistakes, reference mistakes, or
          punctuation. Each of you transcribes the same 120 Indian-English clips
          without seeing the Svarah reference, the whisper-large-v3 hypothesis,
          or each other.
        </p>
      </div>

      <Alert>
        <AlertTitle>Sealed transcripts</AlertTitle>
        <AlertDescription>
          This lab only loads the blind clip list. Do not open{" "}
          <code className="font-mono text-xs">listening_set.json</code> until
          both of you have finished. If more than 15% of the rejected cell is
          reference error, we repair the data before the 12-run gate.
        </AlertDescription>
      </Alert>

      <div className="grid gap-4 sm:grid-cols-2">
        {(["A", "B"] as ReviewerId[]).map((id) => {
          const name = id === "A" ? nameA : nameB;
          const setName = id === "A" ? setNameA : setNameB;
          const done = id === "A" ? doneA : doneB;
          const active = reviewer === id;
          return (
            <Card key={id} className={active ? "ring-2 ring-primary" : ""}>
              <CardHeader>
                <CardTitle>Reviewer {id}</CardTitle>
                <CardDescription>
                  {done} of 120 clips typed on this browser.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-1.5">
                  <Label htmlFor={`name-${id}`}>Your name</Label>
                  <Input
                    id={`name-${id}`}
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder={id === "A" ? "e.g. Aditi" : "e.g. Aditya"}
                  />
                </div>
                <Button onClick={() => choose(id, name)} className="w-full">
                  {active ? "Continue as reviewer " + id : "Work as reviewer " + id}
                </Button>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="flex flex-col gap-3 sm:flex-row">
        <Button asChild disabled={!reviewer}>
          <Link href="/review">Start this session</Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/protocol">Read the protocol</Link>
        </Button>
        <Button asChild variant="ghost">
          <Link href="/reconcile">Open reconciliation</Link>
        </Button>
        <Button asChild variant="ghost">
          <Link href="/practice">Check the audio player</Link>
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>What is in the 120</CardTitle>
          <CardDescription>
            Frozen sample, seed 20260909, from the scored whisper-large-v3 Svarah CSV.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-2 text-sm sm:grid-cols-3">
          <p>
            <span className="text-muted-foreground">L1 groups</span>
            <br />
            Hindi, Bengali, Marathi, Tamil, Telugu, Bodo
          </p>
          <p>
            <span className="text-muted-foreground">Balance</span>
            <br />
            10 accepted + 10 rejected per L1
          </p>
          <p>
            <span className="text-muted-foreground">Audio</span>
            <br />
            Fetch with <code className="font-mono text-xs">python -m research_reset.fetch_audio</code>{" "}
            after Hugging Face login. Until then, you can still type and export.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
