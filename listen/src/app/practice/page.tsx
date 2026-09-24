import { ClipPlayer } from "@/components/clip-player";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function PracticePage() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">Player check</h1>
      <p className="text-muted-foreground max-w-2xl">
        A two-second tone so you can confirm the browser can play wavs. This is
        not Svarah and not part of the 120.
      </p>
      <Card>
        <CardHeader>
          <CardTitle>practice.wav</CardTitle>
          <CardDescription>440 Hz, 2 seconds, 16 kHz mono</CardDescription>
        </CardHeader>
        <CardContent>
          <ClipPlayer uttId="practice" />
        </CardContent>
      </Card>
    </div>
  );
}
