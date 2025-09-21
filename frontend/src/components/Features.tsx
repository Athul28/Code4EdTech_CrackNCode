import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

const features = [
  { title: "Capture", desc: "Upload OMR sheets via mobile camera." },
  { title: "Preprocessing", desc: "Auto correct skew, rotation & lighting." },
  { title: "Bubble Detection", desc: "Robust CV + ML-based bubble classification." },
  { title: "Results", desc: "Instant per-subject and total scoring." },
];

export function Features() {
  return (
    <section id="solution" className="container mx-auto py-16 space-y-8">
      <h2 className="text-4xl font-bold text-center">Our Solution</h2>
      <div className="grid gap-6 sm:grid-cols-2 md:grid-cols-4">
        {features.map((f) => (
          <Card key={f.title} className="hover:shadow-lg transition">
            <CardHeader>
              <CardTitle>{f.title}</CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">{f.desc}</CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
