#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { createRenderer, inferLanguage, renderFile } from "./render.mjs";
import { defaults } from "./theme.mjs";

const RENDERED_SUFFIXES = new Set([
  ".mlir", ".yaml", ".yml", ".py", ".json", ".jsonc", ".jsonl",
  ".sh", ".bash", ".zsh", ".cc", ".cpp", ".cxx", ".h", ".hpp",
  ".s", ".asm", ".ll", ".txt", ".log",
]);

function usageError(message) {
  throw new Error(`${message}\nusage: render-directory.mjs INPUT_DIR -o OUTPUT_DIR [--transparent]`);
}

function parse(argv) {
  let input;
  let output;
  let transparent = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "-o" || arg === "--output") {
      output = argv[index + 1];
      if (output === undefined) usageError(`${arg} requires a value`);
      index += 1;
    } else if (arg === "--transparent") {
      transparent = true;
    } else if (arg.startsWith("-")) {
      usageError(`unknown option: ${arg}`);
    } else if (input === undefined) {
      input = arg;
    } else {
      usageError(`unexpected argument: ${arg}`);
    }
  }
  if (input === undefined || output === undefined) {
    usageError("input and output directories are required");
  }
  return { input: path.resolve(input), output: path.resolve(output), transparent };
}

async function collect(directory, relative = "") {
  const entries = await readdir(path.join(directory, relative), { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const child = path.join(relative, entry.name);
    if (entry.isDirectory()) {
      files.push(...await collect(directory, child));
    } else if (entry.isFile() && RENDERED_SUFFIXES.has(path.extname(entry.name).toLowerCase())) {
      files.push(child);
    }
  }
  return files;
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

const paths = parse(process.argv.slice(2));
const renderer = await createRenderer();
const records = [];
await mkdir(paths.output, { recursive: true });
try {
  for (const relative of await collect(paths.input)) {
    const input = path.join(paths.input, relative);
    const output = path.join(paths.output, `${relative}.svg`);
    const lang = inferLanguage(input);
    const result = await renderFile(renderer, {
      input,
      output,
      lang,
      lines: undefined,
      lineNumbers: false,
      fontSize: defaults.fontSize,
      fontFamily: defaults.fontFamily,
      lineHeight: defaults.lineHeight,
      tabWidth: defaults.tabWidth,
      paddingX: defaults.paddingX,
      paddingY: defaults.paddingY,
      accentDialects: [...defaults.accentDialects],
      transparent: paths.transparent,
      dumpTokens: false,
    });
    records.push({
      source: relative,
      sourceSha256: result.metadata.sourceSha256,
      output: `${relative}.svg`,
      outputSha256: digest(Buffer.from(result.svg, "utf8")),
      language: lang,
      width: result.layout.width,
      height: result.layout.height,
    });
  }
} finally {
  renderer.dispose();
}

const manifest = {
  schemaVersion: 1,
  renderer: "snippet2svg/0.1.0",
  grammarSha256: renderer.grammarSha256,
  sourceRoot: path.basename(paths.input),
  transparent: paths.transparent,
  records,
};
await writeFile(path.join(paths.output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
process.stdout.write(`${records.length} snippets -> ${paths.output}\n`);
