import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

const [, , inputPath, outputPath] = process.argv;
const salt = process.env.STUDENT_ID_HASH_SALT;

if (!inputPath || !outputPath) {
  console.error(
    "Usage: STUDENT_ID_HASH_SALT=... node scripts/hash-roster.mjs input.csv output.csv",
  );
  process.exit(1);
}

if (!salt) {
  console.error("STUDENT_ID_HASH_SALT is required.");
  process.exit(1);
}

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];
    if (quoted && char === '"' && next === '"') {
      value += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (!quoted && char === ",") {
      values.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  values.push(value);
  return values;
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function normalize(value) {
  return String(value ?? "").trim().replace(/\s+/g, " ").toLowerCase();
}

function hash(text) {
  return createHash("sha256").update(text).digest("hex");
}

const text = await readFile(inputPath, "utf8");
const lines = text.split(/\r?\n/).filter((line) => line.trim());
const header = parseCsvLine(lines[0]).map((column) => column.trim());
const index = Object.fromEntries(header.map((column, i) => [column, i]));

for (const required of ["university_id", "first_name", "last_name"]) {
  if (!(required in index)) {
    throw new Error(`Missing required column: ${required}`);
  }
}

const output = [
  [
    "university_id_hash",
    "identity_hash",
    "first_initial",
    "last_initial",
    "section",
    "active",
  ].join(","),
];

for (const line of lines.slice(1)) {
  const row = parseCsvLine(line);
  const universityId = normalize(row[index.university_id]);
  const firstName = normalize(row[index.first_name]);
  const lastName = normalize(row[index.last_name]);
  const section = row[index.section] ?? "";
  const active = row[index.active] || "true";

  if (!universityId || !firstName || !lastName) {
    throw new Error(`Missing required roster value in row: ${line}`);
  }

  output.push([
    hash(`${salt}:student_id:${universityId}`),
    hash(`${salt}:student_identity:${universityId}:${firstName}:${lastName}`),
    firstName.slice(0, 1).toUpperCase(),
    lastName.slice(0, 1).toUpperCase(),
    section,
    active,
  ].map(csvEscape).join(","));
}

await writeFile(outputPath, `${output.join("\n")}\n`);
