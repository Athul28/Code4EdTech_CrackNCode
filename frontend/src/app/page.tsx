"use client";

import { Navbar } from "~/components/Navbar";
import { Footer } from "~/components/Footer";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { CheckCircle, Upload, Brain, FileBarChart } from "lucide-react";
import { useRouter } from "next/navigation";

export default function Home() {
  const features = [
    {
      title: "Capture",
      desc: "Upload OMR sheets directly via mobile or scanner.",
      icon: Upload,
    },
    {
      title: "AI Detection",
      desc: "CV + ML ensures accurate bubble recognition.",
      icon: Brain,
    },
    {
      title: "Instant Scoring",
      desc: "Get per-subject & total scores in seconds.",
      icon: CheckCircle,
    },
    {
      title: "Analytics",
      desc: "Dashboard with downloadable reports & insights.",
      icon: FileBarChart,
    },
  ];

  const steps = [
    "Students fill OMR sheets during exams",
    "Sheets captured via mobile app or scanner",
    "Evaluator uploads sheets to web dashboard",
    "System detects answers & compares with key",
    "Subject-wise + total scores generated instantly",
    "Download results & analytics report",
  ];

  const router = useRouter();

  return (
    <main className="text-foreground flex min-h-screen flex-col bg-gradient-to-b from-purple-500 via-purple-600 to-black">
      <Navbar />

      {/* Hero */}
      <section className="container flex flex-col items-center justify-center gap-6 px-4 py-24 text-center">
        <h1 className="max-w-4xl text-5xl font-extrabold tracking-tight text-white sm:text-7xl">
          Automated <span className="text-yellow-300">OMR Evaluation</span> &
          Scoring System
        </h1>
        <p className="max-w-2xl text-lg text-white/90">
          Eliminate manual errors, reduce evaluation time, and empower educators
          with instant, accurate results for 3000+ OMR sheets in minutes.
        </p>
        <Button
          size="lg"
          className="cursor-pointer rounded-full px-10 shadow-lg transition hover:scale-105"
          onClick={() => router.push("/test")}
        >
          Get Started
        </Button>
      </section>

      {/* Features */}
      <section
        id="solution"
        className="container mx-auto space-y-12 px-4 py-20"
      >
        <h2 className="text-center text-4xl font-bold text-white">
          Our Solution
        </h2>
        <div className="grid gap-8 sm:grid-cols-2 md:grid-cols-4">
          {features.map((f) => (
            <Card
              key={f.title}
              className="rounded-2xl border-white/20 bg-white/10 text-white backdrop-blur transition hover:shadow-xl"
            >
              <CardHeader className="flex flex-col items-center gap-2">
                <f.icon className="h-10 w-10 text-yellow-300" />
                <CardTitle className="text-xl font-semibold">
                  {f.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="text-center text-white/80">
                {f.desc}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Workflow */}
      <section
        id="workflow"
        className="container mx-auto space-y-12 px-4 py-20"
      >
        <h2 className="text-center text-4xl font-bold text-white">Workflow</h2>
        <div className="grid gap-6 md:grid-cols-3">
          {steps.map((s, i) => (
            <Card
              key={i}
              className="rounded-2xl border-white/20 bg-white/10 text-white backdrop-blur transition hover:shadow-xl"
            >
              <CardHeader className="flex flex-col items-center gap-2">
                <CardTitle className="text-xl font-semibold">{`Step ${i + 1}`}</CardTitle>
              </CardHeader>
              <CardContent className="text-center text-white/80">
                {s}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
      <Footer />
    </main>
  );
}
