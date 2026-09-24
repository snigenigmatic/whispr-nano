"use client";

import { useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function ClipPlayer({ uttId }: { uttId: string }) {
  const src = `/audio/${uttId}.wav`;
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    setMissing(false);
  }, [src]);

  if (missing) {
    return (
      <Alert>
        <AlertTitle>Audio not on this machine</AlertTitle>
        <AlertDescription className="space-y-2">
          <p>Svarah is gated. After Hugging Face login, fetch only these 120 wavs:</p>
          <pre className="overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs">
            PYTHONPATH=src python -m research_reset.fetch_audio --out listen/public/audio
          </pre>
          <p>You can still type a transcript if you are listening from another player.</p>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <audio className="w-full" controls src={src} preload="metadata" onError={() => setMissing(true)}>
      Your browser cannot play this wav.
    </audio>
  );
}
