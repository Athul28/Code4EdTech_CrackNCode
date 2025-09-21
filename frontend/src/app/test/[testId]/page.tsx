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
  const [selectedFiles, setSelectedFiles] = useState<
    Record<string, File | null>
  >({});
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

    await fetch("/api/omr", {
      method: "POST",
      body: formData,
    });

    setSelectedFiles((prev) => ({ ...prev, [studentUsn]: null }));

    // Fetch latest students and use the result directly
    const res = await fetch(`/api/student?testId=${testId}`);
    const data = (await res.json()) as Student[];
    setStudents(data);

    // Find updated student in the new data
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
    <div className="mx-auto max-w-xl space-y-4">
      {/* Add Student */}
      <Card>
        <CardHeader>
          <CardTitle>Add Student</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Input
            placeholder="Student name"
            value={studentName}
            onChange={(e) => setStudentName(e.target.value)}
          />
          <Input
            placeholder="USN"
            value={usn}
            onChange={(e) => setUsn(e.target.value)}
          />
          <Button onClick={createStudent}>Add</Button>
        </CardContent>
      </Card>

      {/* Student List with Upload */}
      <div className="space-y-2">
        {students.map((student) => (
          <Card key={student.id}>
            <CardHeader>
              <CardTitle>{student.name}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <p>USN: {student.usn}</p>
              <p>Score: {student.score ?? 0}</p>

              {/* Upload section for each student */}
              <div className="flex items-center gap-2">
                <Input
                  type="file"
                  accept="image/*"
                  onChange={(e) =>
                    setSelectedFiles((prev) => ({
                      ...prev,
                      [student.usn]: e.target.files?.[0] ?? null,
                    }))
                  }
                />
                <Button
                  onClick={() => uploadOMR(student.usn)}
                  disabled={
                    !selectedFiles[student.usn] || processing[student.usn]
                  }
                >
                  {processing[student.usn]
                    ? "Processing..."
                    : "Upload & Process"}
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Score Popup */}
      <AlertDialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>OMR Processed</AlertDialogTitle>
            <AlertDialogDescription>
              {dialogStudent
                ? `${dialogStudent.name} (${dialogStudent.usn}) scored ${dialogStudent.score ?? 0}`
                : "Score updated."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setDialogOpen(false)}>
              OK
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
