import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

const steps = [
  "Students fill OMR sheets during exams",
  "Sheets captured via mobile",
  "Evaluator uploads sheets on web app",
  "System detects bubbles & matches answer key",
  "Generates subject-wise + total scores",
  "Results dashboard & export options",
];

export function Workflow() {
  return (
    <section id="workflow" className="container mx-auto py-16 space-y-8">
      <h2 className="text-4xl font-bold text-center">Workflow</h2>
      <div className="grid gap-6 md:grid-cols-3">
        {steps.map((s, i) => (
          <Card key={i} className="relative group hover:scale-105 transition">
            <CardHeader>
              <CardTitle className="text-primary">Step {i + 1}</CardTitle>
            </CardHeader>
            <CardContent>{s}</CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
