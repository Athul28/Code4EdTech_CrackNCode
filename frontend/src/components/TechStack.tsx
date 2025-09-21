export function TechStack() {
  return (
    <section id="tech" className="container mx-auto py-16 space-y-8 text-center">
      <h2 className="text-4xl font-bold">Tech Stack</h2>
      <p className="text-muted-foreground max-w-3xl mx-auto">
        Python + OpenCV for OMR evaluation · NumPy / SciPy for preprocessing · 
        ML classifiers (scikit-learn / TF Lite) · Flask/FastAPI backend · 
        Next.js + shadcn/ui frontend · PostgreSQL for secure storage
      </p>
    </section>
  );
}
