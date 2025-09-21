"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Input } from "~/components/ui/input";
import { Button } from "~/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "~/components/ui/card";

type Student = {
  id: string;
  name: string;
  usn: string;
  score: number | null;
};

export default function TestPage() {
  const { testId } = useParams<{ testId: string }>();
  const [students, setStudents] = useState<Student[]>([]);
  const [studentName, setStudentName] = useState("");
  const [usn, setUsn] = useState("");
  const [score, setScore] = useState<number | "">("");

  const createStudent = async () => {
    if (!studentName || !usn) return;
    await fetch("/api/student", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: studentName, usn, score: score || 0, testId }),
    });
    setStudentName("");
    setUsn("");
    setScore("");
    void getStudents();
  };

  const getStudents = async () => {
    const res = await fetch(`/api/student?testId=${testId}`);
    const data = (await res.json()) as Student[];
    setStudents(data);
  };

  useEffect(() => {
    if (testId) void getStudents();
  }, [testId]);

  return (
    <div className="max-w-xl mx-auto space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Add Student</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Input
            placeholder="Student name"
            value={studentName}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setStudentName(e.target.value)}
          />
          <Input
            placeholder="USN"
            value={usn}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUsn(e.target.value)}
          />
          {/* <Input
            placeholder="Score"
            type="number"
            value={score}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setScore(Number(e.target.value))}
          /> */}
          <Button onClick={createStudent}>Add</Button>
        </CardContent>
      </Card>

      <div className="space-y-2">
        {students.map((student) => (
          <Card key={student.id}>
            <CardHeader>
              <CardTitle>{student.name}</CardTitle>
            </CardHeader>
            <CardContent>
              <p>USN: {student.usn}</p>
              <p>Score: {student.score ?? 0}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
