#!/usr/bin/env node

import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const TOOL_DIR = path.dirname(fileURLToPath(import.meta.url));

// Updating the grammar is an explicit maintenance operation. Change these
// constants together after reviewing a new llvm/vscode-mlir revision.
const UPSTREAM_COMMIT = "1ea9567704457cc9dfd508a1a3d8f11beff5a7b3";
const FILES = [
  {
    upstream: "grammar.json",
    local: "grammar.json",
    sha256: "5942b4066a1f9c626547acddc1237b205e3a44bf6f27988ad30e7753dd0e82aa",
  },
  {
    upstream: "LICENSE",
    local: "LICENSE.llvm.txt",
    sha256: "94b301bb652ae351f5941655bf533f1a136300ad1396be7452c8a04ff2c91374",
  },
];

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

for (const file of FILES) {
  const url = `https://raw.githubusercontent.com/llvm/vscode-mlir/${UPSTREAM_COMMIT}/${file.upstream}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: HTTP ${response.status}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  const actual = digest(bytes);
  if (actual !== file.sha256) {
    throw new Error(`${file.upstream} SHA-256 mismatch: expected ${file.sha256}, got ${actual}`);
  }
  if (file.upstream === "grammar.json") {
    const parsed = JSON.parse(bytes.toString("utf8"));
    if (parsed.scopeName !== "source.mlir") {
      throw new Error(`unexpected grammar scope: ${parsed.scopeName}`);
    }
  }
  await writeFile(path.join(TOOL_DIR, file.local), bytes);
  process.stdout.write(`${file.local}  ${actual}\n`);
}
