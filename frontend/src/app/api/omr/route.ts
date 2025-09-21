
/* eslint-disable */
import { type NextRequest, NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import path from "path";
import { prisma } from "~/lib/prisma";
import { GoogleGenerativeAI } from "@google/generative-ai";

const geminiApiKey = process.env.GEMINI_API;
if (!geminiApiKey) {
  throw new Error("GEMINI_API environment variable is not set.");
}
// @ts-ignore
const genAI = new GoogleGenerativeAI(geminiApiKey);

// Example answer key (could also store in DB)
const ANSWER_KEY: Record<string, Record<string, string[]>> = {
  python: {
    "1": ["A"],
    "2": ["C"],
    "3": ["C"],
    "4": ["C"],
    "5": ["C"],
    "6": ["A"],
    "7": ["C"],
    "8": ["C"],
    "9": ["B"],
    "10": ["C"],
    "11": ["A"],
    "12": ["A"],
    "13": ["D"],
    "14": ["A"],
    "15": ["B"],
    "16": ["A", "B", "C", "D"],
    "17": ["C"],
    "18": ["D"],
    "19": ["A"],
    "20": ["B"],
  },
  data_analysis: {
    "21": ["A"],
    "22": ["D"],
    "23": ["B"],
    "24": ["A"],
    "25": ["C"],
    "26": ["B"],
    "27": ["A"],
    "28": ["B"],
    "29": ["D"],
    "30": ["C"],
    "31": ["C"],
    "32": ["A"],
    "33": ["B"],
    "34": ["C"],
    "35": ["A"],
    "36": ["B"],
    "37": ["D"],
    "38": ["B"],
    "39": ["A"],
    "40": ["B"],
  },
  mysql: {
    "41": ["C"],
    "42": ["C"],
    "43": ["C"],
    "44": ["B"],
    "45": ["B"],
    "46": ["A"],
    "47": ["C"],
    "48": ["B"],
    "49": ["D"],
    "50": ["A"],
    "51": ["C"],
    "52": ["B"],
    "53": ["C"],
    "54": ["C"],
    "55": ["A"],
    "56": ["B"],
    "57": ["B"],
    "58": ["A"],
    "59": ["A", "B"],
    "60": ["B"],
  },
  power_bi: {
    "61": ["B"],
    "62": ["C"],
    "63": ["A"],
    "64": ["B"],
    "65": ["C"],
    "66": ["B"],
    "67": ["B"],
    "68": ["C"],
    "69": ["C"],
    "70": ["B"],
    "71": ["B"],
    "72": ["B"],
    "73": ["D"],
    "74": ["B"],
    "75": ["A"],
    "76": ["B"],
    "77": ["B"],
    "78": ["B"],
    "79": ["B"],
    "80": ["B"],
  },
  advanced_statistics: {
    "81": ["A"],
    "82": ["B"],
    "83": ["C"],
    "84": ["B"],
    "85": ["C"],
    "86": ["B"],
    "87": ["B"],
    "88": ["B"],
    "89": ["A"],
    "90": ["B"],
    "91": ["C"],
    "92": ["B"],
    "93": ["C"],
    "94": ["B"],
    "95": ["B"],
    "96": ["B"],
    "97": ["C"],
    "98": ["A"],
    "99": ["B"],
    "100": ["C"],
  },
};

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File;
    const usn = formData.get("usn") as string;
    const testId = formData.get("testId") as string;

    if (!file || !usn || !testId) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 },
      );
    }

    // Save uploaded file temporarily
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filePath = path.join("/tmp", file.name);
    await writeFile(filePath, buffer);

    // Call Gemini
    // @ts-ignore
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const prompt = `
You are an OMR evaluator. Your task is to extract responses from a scanned OMR sheet image.  

Rules:
1. The sheet has 100 questions divided into 5 subjects:
   - python → Q1–20
   - data_analysis → Q21–40
   - mysql → Q41–60
   - power_bi → Q61–80
   - advanced_statistics → Q81–100
2. Each question has options A, B, C, D.
3. If one bubble is filled, return it as a single-element array, e.g. "1": ["B"].
4. If multiple bubbles are filled, include all in an array, e.g. "16": ["A","B","C","D"].
5. Do not infer or guess. Only return what is marked.
6. Respond **only in valid JSON**, no explanations or extra text.

Output JSON format:
{
  "responses": {
    "python": { "1": ["A"], "2": ["C"], ... },
    "data_analysis": { "21": ["B"], "22": ["D"], ... },
    "mysql": { "41": ["C"], ... },
    "power_bi": { "61": ["B"], ... },
    "advanced_statistics": { "81": ["A"], ... }
  }
}

`;

    const result = await model.generateContent([
      { inlineData: { mimeType: file.type, data: buffer.toString("base64") } },
      { text: prompt },
    ]);

    let textResponse = result.response.text();

    textResponse = textResponse
      .replace(/```json\s*/g, "")
      .replace(/```/g, "")
      .trim();

    console.log(textResponse);
    let omrData;
    try {
      omrData = JSON.parse(textResponse);
    } catch (err) {
      console.error(
        "Failed to parse JSON:",
        err,
        "Response was:",
        textResponse,
      );
      return NextResponse.json(
        { error: "Invalid JSON from Gemini" },
        { status: 500 },
      );
    }

    // Compare with answer key
    let score = 0;
    for (const subject of Object.keys(ANSWER_KEY)) {
      const answers = ANSWER_KEY[subject];
      // @ts-ignore: Object.keys may get undefined if responses is undefined
      const responses = omrData.responses?.[subject] ?? {};
      // @ts-ignore
      for (const q of Object.keys(answers)) {
        // @ts-ignore
        const correct = answers[q];
        // @ts-ignore
        const given = responses[q] ?? [];
        // @ts-ignore
        if (JSON.stringify(correct.sort()) === JSON.stringify(given.sort())) {
          score++;
        }
      }
    }

    // Update DB (student identified by usn + testId)
    await prisma.student.updateMany({
      where: { usn, testId },
      data: { score },
    });
    // @ts-ignore
    return NextResponse.json({ success: true, score, omrData });
    // @ts-ignore
  } catch (error: any) {
    console.error("OMR Processing Error:", error);
    // @ts-ignore
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
