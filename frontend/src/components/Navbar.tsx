"use client";

import { useRouter } from "next/navigation";
import { Button } from "~/components/ui/button";

export function Navbar() {
    const router=useRouter()
  return (
    <nav className="sticky top-0 z-50 w-full border-b border-white/10 bg-purple-900/40 backdrop-blur">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <span className="text-2xl font-bold tracking-tight text-white">
          OMR<span className="text-yellow-300">AI</span>
        </span>
        <div className="flex items-center gap-6 text-white/90">
          <a href="#solution" className="transition hover:text-yellow-300">
            Solution
          </a>
          <a href="#workflow" className="transition hover:text-yellow-300">
            Workflow
          </a>
          <a href="#tech" className="transition hover:text-yellow-300">
            Tech
          </a>
          <Button
            className="bg-yellow-300 text-black hover:bg-yellow-400 cursor-pointer"
            onClick={() => router.push("/test")}
          >
            Start
          </Button>
        </div>
      </div>
    </nav>
  );
}
