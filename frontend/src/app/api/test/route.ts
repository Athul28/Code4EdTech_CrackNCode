import { type NextRequest, NextResponse } from "next/server";
import { prisma } from "~/lib/prisma";

interface testData {
  userId: string;
  name: string;
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const userId = searchParams.get("userId");
  if (!userId) {
    return NextResponse.json(
      { message: "User ID not defined" },
      { status: 400 },
    );
  }
  const tests = await prisma.test.findMany({ where: { userId } });
  return NextResponse.json(tests);
}

export async function POST(req: NextRequest) {
  const data = (await req.json()) as testData;
  const test = await prisma.test.create({ data });
  return NextResponse.json(test);
}
