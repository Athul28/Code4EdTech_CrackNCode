"use client";

import { Navbar } from "~/components/Navbar";
import { Footer } from "~/components/Footer";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { CheckCircle, Upload, Brain, FileBarChart } from "lucide-react";
import { useRouter } from "next/navigation";

export default function Home() {
  const features = [
    { title: "Capture", desc: "Upload OMR sheets directly via mobile or scanner.", icon: Upload },
    { title: "AI Detection", desc: "CV + ML ensures accurate bubble recognition.", icon: Brain },
    { title: "Instant Scoring", desc: "Get per-subject & total scores in seconds.", icon: CheckCircle },
    { title: "Analytics", desc: "Dashboard with downloadable reports & insights.", icon: FileBarChart },
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
    <main className="flex min-h-screen flex-col bg-gradient-to-b from-purple-500 via-purple-600 to-purple-900">
      <Navbar />

      {/* Hero */}
      <section className="container mx-auto flex flex-col items-center justify-center gap-6 px-4 py-24 text-center">
        <h1 className="max-w-4xl text-4xl font-extrabold tracking-tight text-white sm:text-6xl lg:text-7xl">
          Automated <span className="text-yellow-300">OMR Evaluation</span> & Scoring System
        </h1>
        <p className="max-w-2xl text-lg text-white/90 sm:text-xl">
          Eliminate manual errors, reduce evaluation time, and empower educators with instant, accurate results for 3000+ OMR sheets in minutes.
        </p>
        <Button
          size="lg"
          className="mt-4 rounded-full bg-yellow-400 px-12 py-4 font-semibold shadow-lg hover:scale-105 hover:shadow-2xl transition-transform"
          onClick={() => router.push("/test")}
        >
          Get Started
        </Button>
      </section>

      {/* Features */}
      <section id="solution" className="container mx-auto space-y-12 px-4 py-20">
        <h2 className="text-center text-3xl font-bold text-white sm:text-4xl">Our Solution</h2>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((f) => (
            <Card
              key={f.title}
              className="flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-white/10 p-6 text-white backdrop-blur-md transition hover:scale-105 hover:shadow-2xl"
            >
              <CardHeader className="flex flex-col items-center gap-3">
                <f.icon className="h-12 w-12 text-yellow-300" />
                <CardTitle className="text-xl font-semibold text-center">{f.title}</CardTitle>
              </CardHeader>
              <CardContent className="text-center text-white/80">{f.desc}</CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Workflow */}
      <section id="workflow" className="container mx-auto space-y-12 px-4 py-20">
        <h2 className="text-center text-3xl font-bold text-white sm:text-4xl">Workflow</h2>
        <div className="grid gap-6 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
          {steps.map((s, i) => (
            <Card
              key={i}
              className="flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-white/10 p-6 text-white backdrop-blur-md transition hover:scale-105 hover:shadow-2xl"
            >
              <CardHeader className="flex flex-col items-center gap-2">
                <CardTitle className="text-lg font-semibold text-center sm:text-xl">{`Step ${i + 1}`}</CardTitle>
              </CardHeader>
              <CardContent className="text-center text-white/80">{s}</CardContent>
            </Card>
          ))}
        </div>
      </section>

      <Footer />
    </main>
  );
}
