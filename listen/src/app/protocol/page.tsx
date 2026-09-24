import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function ProtocolPage() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Listening protocol</h1>
        <p className="text-muted-foreground max-w-2xl">
          Two independent transcripts, then a joint label. Full writeup:{" "}
          <code className="font-mono text-xs">docs/LISTENING_PROTOCOL.md</code> in the repo.
        </p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>While you transcribe</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm leading-relaxed">
          <p>Headphones. Quiet room. Worksheet order.</p>
          <p>Type what was said, including disfluencies you clearly hear.</p>
          <p>Do not “correct” Indian-English grammar or lexical choice.</p>
          <p>
            Unintelligible: <code className="font-mono">[inaudible]</code>. Unsure word:{" "}
            <code className="font-mono">[uncertain: guess]</code>.
          </p>
          <p>Keep numbers as you hear them — digits or words.</p>
          <p>Do not discuss a clip until both reviewers have submitted it.</p>
          <p>Do not open the dataset reference, the teacher hypothesis, or the audit CSV.</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>When you reconcile</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm leading-relaxed">
          <p>
            One primary label per clip: teacher mistake, reference mistake, both wrong,
            normalization, genuine variation, or other.
          </p>
          <p>
            If reference_mistake + both_wrong exceed 15% of the 60 rejected clips, stop.
            Repair the data before the 12-run label-source experiment.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
