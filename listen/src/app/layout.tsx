import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { LabShell } from "@/components/lab-shell";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Listening lab — 120 Indian-English clips",
  description:
    "Two-reviewer dual transcription for the Indian-English ASR research reset. Transcripts stay hidden until both reviewers finish.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased font-sans`}>
        <LabShell>{children}</LabShell>
      </body>
    </html>
  );
}
