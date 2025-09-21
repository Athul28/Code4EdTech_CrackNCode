import { Button } from "~/components/ui/button";

export function Hero() {
  return (
    <section className="flex flex-col items-center justify-center gap-6 py-24 text-center bg-gradient-to-b from-primary/20 via-background to-background">
      <h1 className="text-5xl sm:text-7xl font-extrabold tracking-tight max-w-3xl">
        Automated <span className="text-primary">OMR Evaluation</span> & Scoring System
      </h1>
      <p className="max-w-2xl text-lg text-muted-foreground">
        Eliminate manual errors, reduce evaluation time, and empower educators with instant, 
        accurate results for 3000+ OMR sheets in minutes.
      </p>
      <Button size="lg" className="rounded-full px-8">
        Get Started
      </Button>
    </section>
  );
}
