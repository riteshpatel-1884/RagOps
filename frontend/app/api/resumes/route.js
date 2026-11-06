import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const dir = path.join(process.cwd(), "public", "resume");
    const all = await fs.readdir(dir);
    const files = all.filter((f) => f.toLowerCase().endsWith(".pdf"));
    return NextResponse.json({ files });
  } catch {
    return NextResponse.json({ files: [] });
  }
}
