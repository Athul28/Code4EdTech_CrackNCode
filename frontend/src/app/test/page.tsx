"use client";

import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { Input } from "~/components/ui/input";
import { Button } from "~/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "~/components/ui/card";
import Link from "next/link";

type TestData = {
  id: string;
  name: string;
  totalScore: number;
};

export default function Test() {
  const [testName, setTestName] = useState("");
  const [test, setTest] = useState<TestData[]>([]);
  const { data: session, status } = useSession();

  const createTest = async () => {
    if (!testName) return;
    await fetch("/api/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: testName, userId: session?.user.id }),
    });
    setTestName("");
    void getTest();
  };

  const getTest = async () => {
    const res = await fetch(`/api/test?userId=${session?.user.id}`);
    const data = (await res.json()) as TestData[];
    setTest(data);
  };

  useEffect(() => {
    if (session?.user.id) void getTest();
  }, [session]);

  if (status === "loading")
    return (
      <div className="flex h-screen items-center justify-center text-white text-xl">
        Loading...
      </div>
    );

  if (!session)
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-6 bg-gradient-to-b from-purple-500 via-purple-600 to-purple-900 text-white text-center px-4">
        <h1 className="text-3xl font-bold">Please log in to continue</h1>
        <Link
          href={session ? "/api/auth/signout" : "/api/auth/signin"}
          className="rounded-full bg-yellow-400 px-10 py-3 font-semibold transition hover:scale-105 hover:shadow-lg"
        >
          Sign In
        </Link>
      </div>
    );

  return (
    <main className="min-h-screen bg-gradient-to-b from-purple-500 via-purple-600 to-purple-900 px-4 py-12">
      <div className="mx-auto max-w-2xl space-y-6">
        {/* Sign Out */}
        <div className="flex justify-end">
          <Link
            href="/api/auth/signout"
            className="rounded-full bg-white/10 px-6 py-2 font-semibold text-white transition hover:bg-white/20"
          >
            Sign Out
          </Link>
        </div>

        {/* Create Test */}
        <Card className="bg-white/10 backdrop-blur-md border border-white/20 rounded-3xl shadow-md">
          <CardHeader>
            <CardTitle className="text-white text-2xl">Create Test</CardTitle>
          </CardHeader>
            <CardContent className="flex flex-col sm:flex-row gap-3">
              <Input
              placeholder="Enter test name"
              value={testName}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setTestName(e.target.value)
              }
              className="flex-1 bg-white/20 text-white border-white/20 focus:border-yellow-300 focus:ring-yellow-300 placeholder:text-white"
              />
              <Button
              onClick={createTest}
              className="bg-yellow-400 text-black hover:bg-yellow-500 transition shadow-lg"
              >
              Submit
              </Button>
            </CardContent>
        </Card>

        {/* Test List */}
        <div className="space-y-4">
          {test.map((item) => (
            <Card
              key={item.id}
              className="bg-white/10 backdrop-blur-md border border-white/20 rounded-3xl shadow-md hover:shadow-xl transition"
            >
              <CardHeader>
                <CardTitle className="text-white text-lg sm:text-xl">
                  <Link href={`/test/${item.id}`} className="hover:underline">
                    {item.name}
                  </Link>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col sm:flex-row items-center justify-between gap-2 text-white/90">
                {/* <span>Total Score: {item.totalScore ?? 0}</span> */}
                <Button
                  asChild
                  className="bg-yellow-400 text-black hover:bg-yellow-500 transition shadow-md"
                >
                  <Link href={`/test/${item.id}`}>Check Out</Link>
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </main>
  );
}
