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

  if (status === "loading") return <h1>Loading...</h1>;
  if (!session)
    return (
      <h1>
        Please log in to continue
        <Link
          href={session ? "/api/auth/signout" : "/api/auth/signin"}
          className="rounded-full bg-white/10 px-10 py-3 font-semibold no-underline transition hover:bg-white/20"
        >
          Sign In
        </Link>
      </h1>
    );

  return (
    <div className="mx-auto max-w-xl space-y-4">
      <Link
        href="/api/auth/signout"
        className="rounded-full bg-white/10 px-10 py-3 font-semibold no-underline transition hover:bg-white/20"
      >
        Sign Out
      </Link>
      <Card>
        <CardHeader>
          <CardTitle>Create Test</CardTitle>
        </CardHeader>
        <CardContent className="flex gap-2">
          <Input
            placeholder="Enter test name"
            value={testName}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setTestName(e.target.value)
            }
          />
          <Button onClick={createTest}>Submit</Button>
        </CardContent>
      </Card>

      <div className="space-y-2">
        {test.map((item) => (
          <Card key={item.id}>
            <CardHeader>
              <CardTitle>
                <Link href={`/test/${item.id}`} className="hover:underline">
                  {item.name}
                </Link>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex items-center justify-between">
              <span>Total Score: {item.totalScore ?? 0}</span>
              <Button asChild>
                <Link href={`/test/${item.id}`}>Check Out</Link>
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
