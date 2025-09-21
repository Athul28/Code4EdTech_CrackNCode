"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Input } from "~/components/ui/input";
import { Button } from "~/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "~/components/ui/card";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogAction,
} from "~/components/ui/alert-dialog";

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
  const [selectedFiles, setSelectedFiles] = useState<Record<string, File | null>>({});
  const [processing, setProcessing] = useState<Record<string, boolean>>({});
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogStudent, setDialogStudent] = useState<Student | null>(null);

  const createStudent = async () => {
    if (!studentName || !usn) return;
    await fetch("/api/student", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: studentName, usn, testId }),
    });
    setStudentName("");
    setUsn("");
    void getStudents();
  };

  const uploadOMR = async (studentUsn: string) => {
    const file = selectedFiles[studentUsn];
    if (!file) return;

    setProcessing((prev) => ({ ...prev, [studentUsn]: true }));

    const formData = new FormData();
    formData.append("file", file);
    formData.append("usn", studentUsn);
    formData.append("testId", testId);

    await fetch("/api/omr", { method: "POST", body: formData });

    setSelectedFiles((prev) => ({ ...prev, [studentUsn]: null }));

    const res = await fetch(`/api/student?testId=${testId}`);
    const data = (await res.json()) as Student[];
    setStudents(data);

    const updated = data.find((s) => s.usn === studentUsn);
    if (updated) {
      setDialogStudent(updated);
      setDialogOpen(true);
    }

    setProcessing((prev) => ({ ...prev, [studentUsn]: false }));
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
    <main className="min-h-screen bg-purple-800 px-4 py-12">
      <div className="mx-auto max-w-2xl space-y-6">
        {/* Add Student */}
        <Card className="bg-white/10 backdrop-blur-md border border-white/20 rounded-3xl shadow-md">
          <CardHeader>
            <CardTitle className="text-white text-2xl">Add Student</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col sm:flex-row gap-3">
            <Input
              placeholder="Student Name"
              value={studentName}
              onChange={(e) => setStudentName(e.target.value)}
              className="flex-1 bg-white/20 text-white border-white/20 focus:border-yellow-300 focus:ring-yellow-300 placeholder:text-white"
            />
            <Input
              placeholder="USN"
              value={usn}
              onChange={(e) => setUsn(e.target.value)}
              className="flex-1 bg-white/20 text-white border-white/20 focus:border-yellow-300 focus:ring-yellow-300 placeholder:text-white"
            />
            <Button
              onClick={createStudent}
              className="bg-yellow-400 text-black hover:bg-yellow-500 transition shadow-lg"
            >
              Add
            </Button>
          </CardContent>
        </Card>

        {/* Student List */}
        <div className="space-y-4">
          {students.map((student) => (
            <Card
              key={student.id}
              className="bg-white/10 backdrop-blur-md border border-white/20 rounded-3xl shadow-md hover:shadow-xl transition"
            >
              <CardHeader>
                <CardTitle className="text-white text-lg sm:text-xl">{student.name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-white/90">USN: {student.usn}</p>
                <p className="text-white/90">Score: {student.score ?? 0}</p>

                {/* Upload section */}
                <div className="flex flex-col sm:flex-row items-center gap-2">
                  <Input
                    type="file"
                    accept="image/*"
                    onChange={(e) =>
                      setSelectedFiles((prev) => ({
                        ...prev,
                        [student.usn]: e.target.files?.[0] ?? null,
                      }))
                    }
                    className="bg-white/20 text-white placeholder-white/60 border-white/20 focus:border-yellow-300 focus:ring-yellow-300 flex-1"
                  />
                  <Button
                    onClick={() => uploadOMR(student.usn)}
                    disabled={!selectedFiles[student.usn] || processing[student.usn]}
                    className="bg-yellow-400 text-black hover:bg-yellow-500 transition shadow-lg flex-shrink-0"
                  >
                    {processing[student.usn] ? "Processing..." : "Upload & Process"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Score Popup */}
      <AlertDialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>OMR Processed</AlertDialogTitle>
            <AlertDialogDescription className="text-white/90">
              {dialogStudent
                ? `${dialogStudent.name} (${dialogStudent.usn}) scored ${dialogStudent.score ?? 0}`
                : "Score updated."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setDialogOpen(false)} className="bg-yellow-400 text-black hover:bg-yellow-500">
              OK
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </main>
  );
}
