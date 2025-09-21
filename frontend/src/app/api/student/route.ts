import { type NextRequest, NextResponse } from "next/server";
import { prisma } from "~/lib/prisma";

interface StudentData {
  name: string;
  usn: string;
  score?: number;
  testId: string;
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const testId = searchParams.get("testId");
  if (!testId) {
    return NextResponse.json({ message: "Test ID required" }, { status: 400 });
  }

  const students = await prisma.student.findMany({ where: { testId } });
  return NextResponse.json(students);
}

export async function POST(req: NextRequest) {
  const data = (await req.json()) as StudentData;
  if (!data.testId || !data.name || !data.usn) {
    return NextResponse.json({ message: "Missing required fields" }, { status: 400 });
  }

  const student = await prisma.student.create({ data });
  return NextResponse.json(student);
}
