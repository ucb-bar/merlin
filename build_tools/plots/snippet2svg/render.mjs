#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

import { createHighlighter } from "shiki";

import {
  colors,
  defaults,
  paperTheme,
  semanticStyles,
} from "./theme.mjs";

const TOOL_DIR = path.dirname(fileURLToPath(import.meta.url));
const GRAMMAR_PATH = path.join(TOOL_DIR, "grammar.json");
const TOOL_VERSION = "0.1.0";
const SHIKI_VERSION = "4.4.3";
export const EXPECTED_GRAMMAR_SHA256 =
  "5942b4066a1f9c626547acddc1237b205e3a44bf6f27988ad30e7753dd0e82aa";

const BUILTIN_LANGUAGES = Object.freeze([
  "yaml",
  "python",
  "json",
  "jsonc",
  "shellscript",
  "cpp",
  "asm",
  "llvm",
]);

const LANGUAGE_ALIASES = Object.freeze({
  mlir: "mlir",
  yaml: "yaml",
  yml: "yaml",
  python: "python",
  py: "python",
  json: "json",
  jsonc: "jsonc",
  jsonl: "json",
  bash: "shellscript",
  sh: "shellscript",
  shell: "shellscript",
  shellscript: "shellscript",
  zsh: "shellscript",
  cpp: "cpp",
  "c++": "cpp",
  cc: "cpp",
  cxx: "cpp",
  asm: "asm",
  assembly: "asm",
  llvm: "llvm",
  ll: "llvm",
  text: "text",
  txt: "text",
  plaintext: "text",
  plain: "text",
  log: "text",
});

const EXTENSION_LANGUAGES = Object.freeze({
  ".mlir": "mlir",
  ".yaml": "yaml",
  ".yml": "yaml",
  ".py": "python",
  ".json": "json",
  ".jsonc": "jsonc",
  ".jsonl": "json",
  ".sh": "shellscript",
  ".bash": "shellscript",
  ".zsh": "shellscript",
  ".cc": "cpp",
  ".cpp": "cpp",
  ".cxx": "cpp",
  ".h": "cpp",
  ".hpp": "cpp",
  ".s": "asm",
  ".asm": "asm",
  ".ll": "llvm",
  ".txt": "text",
  ".log": "text",
  ".md": "text",
});

const HELP = `Usage:
  snippet2svg INPUT [-o OUTPUT] [options]

Options:
  -o, --output FILE           SVG output (default: INPUT with .svg suffix)
  --lang LANGUAGE             Override language inferred from the suffix
  --lines START:END           Inclusive original source line range
  --segments A:B,C:D          Non-contiguous ranges separated by a visible ellipsis
  --line-numbers              Show original source line numbers
  --font-size N               Font size in SVG units (default: 18)
  --font-family NAME          SVG font-family fallback list
  --line-height N             Line-height multiplier (default: 1.4)
  --tab-width N               Rendering-only tab width (default: 4)
  --padding-x N               Horizontal padding (default: 18)
  --padding-y N               Vertical padding (default: 14)
  --accent-dialect DIALECT    Add a dialect to semantic emphasis; repeatable
  --transparent               Omit the default white background rectangle
  --dump-tokens               Print selected token contents/scopes/styles as JSON
  -h, --help                  Show this help
  --version                   Show the renderer version
`;

function fail(message) {
  const error = new Error(message);
  error.name = "UsageError";
  throw error;
}

function takeValue(argv, index, option) {
  if (index + 1 >= argv.length || argv[index + 1].startsWith("-")) {
    fail(`${option} requires a value`);
  }
  return argv[index + 1];
}

function positiveNumber(value, option, { integer = false } = {}) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0 || (integer && !Number.isInteger(parsed))) {
    fail(`${option} must be a positive ${integer ? "integer" : "number"}`);
  }
  return parsed;
}

export function parseArgs(argv) {
  const options = {
    input: undefined,
    output: undefined,
    lang: undefined,
    lines: undefined,
    segments: undefined,
    lineNumbers: false,
    fontSize: defaults.fontSize,
    fontFamily: defaults.fontFamily,
    fontWeight: defaults.fontWeight,
    lineHeight: defaults.lineHeight,
    tabWidth: defaults.tabWidth,
    paddingX: defaults.paddingX,
    paddingY: defaults.paddingY,
    accentDialects: [...defaults.accentDialects],
    transparent: false,
    dumpTokens: false,
    help: false,
    version: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
    } else if (arg === "--version") {
      options.version = true;
    } else if (arg === "-o" || arg === "--output") {
      options.output = takeValue(argv, i, arg);
      i += 1;
    } else if (arg === "--lang") {
      options.lang = normalizeLanguage(takeValue(argv, i, arg));
      i += 1;
    } else if (arg === "--lines") {
      options.lines = parseLineRange(takeValue(argv, i, arg));
      i += 1;
    } else if (arg === "--segments") {
      options.segments = parseSegments(takeValue(argv, i, arg));
      i += 1;
    } else if (arg === "--line-numbers") {
      options.lineNumbers = true;
    } else if (arg === "--font-size") {
      options.fontSize = positiveNumber(takeValue(argv, i, arg), arg);
      i += 1;
    } else if (arg === "--font-family") {
      options.fontFamily = takeValue(argv, i, arg);
      i += 1;
    } else if (arg === "--line-height") {
      options.lineHeight = positiveNumber(takeValue(argv, i, arg), arg);
      i += 1;
    } else if (arg === "--tab-width") {
      options.tabWidth = positiveNumber(takeValue(argv, i, arg), arg, { integer: true });
      i += 1;
    } else if (arg === "--padding-x") {
      options.paddingX = positiveNumber(takeValue(argv, i, arg), arg);
      i += 1;
    } else if (arg === "--padding-y") {
      options.paddingY = positiveNumber(takeValue(argv, i, arg), arg);
      i += 1;
    } else if (arg === "--accent-dialect") {
      const dialect = takeValue(argv, i, arg);
      if (!/^[A-Za-z_][A-Za-z0-9_$-]*$/.test(dialect)) {
        fail(`${arg} must be an MLIR dialect namespace`);
      }
      if (!options.accentDialects.includes(dialect)) {
        options.accentDialects.push(dialect);
      }
      i += 1;
    } else if (arg === "--transparent") {
      options.transparent = true;
    } else if (arg === "--dump-tokens") {
      options.dumpTokens = true;
    } else if (arg.startsWith("-")) {
      fail(`unknown option: ${arg}`);
    } else if (options.input === undefined) {
      options.input = arg;
    } else {
      fail(`unexpected positional argument: ${arg}`);
    }
  }

  if (!options.help && !options.version && options.input === undefined) {
    fail("missing input file");
  }
  if (options.lines !== undefined && options.segments !== undefined) {
    fail("--lines and --segments are mutually exclusive");
  }
  return options;
}

export function parseLineRange(value) {
  const match = /^(\d+):(\d+)$/.exec(value);
  if (match === null) {
    fail(`invalid line range '${value}'; expected START:END`);
  }
  const start = Number(match[1]);
  const end = Number(match[2]);
  if (start < 1 || end < start) {
    fail(`invalid line range '${value}'; require 1 <= START <= END`);
  }
  return { start, end };
}

export function parseSegments(value) {
  const parts = value.split(",");
  if (parts.length === 0 || parts.some((part) => part === "")) {
    fail(`invalid segments '${value}'; expected A:B,C:D`);
  }
  const ranges = parts.map(parseLineRange);
  for (let index = 1; index < ranges.length; index += 1) {
    if (ranges[index].start <= ranges[index - 1].end) {
      fail("--segments ranges must be ordered and non-overlapping");
    }
  }
  return ranges;
}

export function normalizeLanguage(value) {
  const key = value.toLowerCase();
  const language = LANGUAGE_ALIASES[key];
  if (language === undefined) {
    const supported = [...new Set(Object.values(LANGUAGE_ALIASES))].sort().join(", ");
    fail(`unsupported language '${value}'; choose one of: ${supported}`);
  }
  return language;
}

export function inferLanguage(filename) {
  return EXTENSION_LANGUAGES[path.extname(filename).toLowerCase()] ?? "text";
}

function outputPathFor(input) {
  const extension = path.extname(input);
  return extension === "" ? `${input}.svg` : input.slice(0, -extension.length) + ".svg";
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function normalizeNewlines(source) {
  return source.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
}

function escapeText(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeAttribute(value) {
  return escapeText(String(value))
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function scopeNames(token) {
  const scopes = [];
  for (const explanation of token.explanation ?? []) {
    for (const scope of explanation.scopes ?? []) {
      if (!scopes.includes(scope.scopeName)) {
        scopes.push(scope.scopeName);
      }
    }
  }
  return scopes;
}

function explanationPieces(token) {
  const explanations = token.explanation ?? [];
  if (explanations.length === 0) {
    return [{ ...token, scopes: [] }];
  }
  const pieces = [];
  let offset = token.offset;
  for (const explanation of explanations) {
    pieces.push({
      ...token,
      content: explanation.content,
      offset,
      scopes: (explanation.scopes ?? []).map((scope) => scope.scopeName),
    });
    offset += explanation.content.length;
  }
  if (pieces.map((piece) => piece.content).join("") !== token.content) {
    return [{ ...token, scopes: scopeNames(token) }];
  }
  return pieces;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function semanticMatcher(dialects) {
  const namespaces = dialects.map(escapeRegExp).join("|");
  const customType = namespaces === "" ? "(?!)" : `!(?:${namespaces})\\.[A-Za-z0-9_$.-]+`;
  const dialectOperation = namespaces === "" ? "(?!)" : `(?:${namespaces})\\.[A-Za-z0-9_$.-]+`;
  const ssa = "%[A-Za-z0-9_.$:#-]+";
  const builtinType = "(?:complex|memref|tensor|tuple|vector|index|none|bf16|f16|f32|f64|f80|f128|[us]?i[0-9]+)";
  return new RegExp(`${ssa}|${customType}|${dialectOperation}|\\b${builtinType}\\b`, "g");
}

function hasScope(piece, scope) {
  return piece.scopes.some((name) => name === scope || name.startsWith(`${scope}.`));
}

function isOperationPosition(line, start, end) {
  const prefix = line.slice(0, start);
  const suffix = line.slice(end);
  if (suffix.trimStart().startsWith("=")) {
    return false;
  }
  if (prefix.trim() === "") {
    return true;
  }
  return /^\s*%[A-Za-z0-9_.$:#-]+(?:\s*,\s*%[A-Za-z0-9_.$:#-]+)*\s*=\s*$/.test(prefix);
}

function semanticKind(content, piece, line, start, end) {
  if (content.startsWith("%") && hasScope(piece, "variable.other.mlir")) {
    return "ssa";
  }
  if (content.startsWith("!") && hasScope(piece, "entity.name.type.mlir")) {
    return "accentType";
  }
  if (
    content.includes(".")
      && hasScope(piece, "variable.other.enummember.mlir")
      && isOperationPosition(line, start, end)
  ) {
    return "accentOperation";
  }
  if (hasScope(piece, "entity.name.type.mlir")) {
    return "builtinType";
  }
  return undefined;
}

function splitSemanticPiece(piece, matcher, line, pieceStart) {
  if (piece.type === 1 || piece.type === 2) {
    return [{ ...piece, semanticKind: undefined }];
  }
  const output = [];
  let cursor = 0;
  for (const match of piece.content.matchAll(matcher)) {
    const start = match.index;
    if (start > cursor) {
      output.push({ ...piece, content: piece.content.slice(cursor, start), semanticKind: undefined });
    }
    const content = match[0];
    output.push({
      ...piece,
      content,
      semanticKind: semanticKind(content, piece, line, pieceStart + start, pieceStart + start + content.length),
    });
    cursor = start + content.length;
  }
  if (cursor < piece.content.length) {
    output.push({ ...piece, content: piece.content.slice(cursor), semanticKind: undefined });
  }
  return output.length === 0 ? [{ ...piece, semanticKind: undefined }] : output;
}

export function applySemanticOverrides(tokenLines, language, dialects) {
  if (language !== "mlir") {
    return tokenLines.map((line) => line.map((token) => ({
      ...token,
      scopes: scopeNames(token),
      semanticKind: undefined,
    })));
  }
  const matcher = semanticMatcher(dialects);
  return tokenLines.map((line) => {
    const sourceLine = line.map((token) => token.content).join("");
    let column = 0;
    const output = [];
    for (const token of line) {
      for (const piece of explanationPieces(token)) {
        output.push(...splitSemanticPiece(piece, matcher, sourceLine, column));
        column += piece.content.length;
      }
    }
    return output;
  });
}

function renderedStyle(token) {
  const semantic = token.semanticKind === undefined
    ? undefined
    : semanticStyles[token.semanticKind];
  return {
    color: semantic?.color ?? token.color ?? colors.text,
    fontWeight: semantic?.fontWeight,
    fontStyle: token.fontStyle,
  };
}

function expandTabs(tokens, tabWidth) {
  const output = [];
  let column = 0;
  for (const token of tokens) {
    let rendered = "";
    for (const character of token.content) {
      if (character === "\t") {
        const count = tabWidth - (column % tabWidth);
        rendered += " ".repeat(count);
        column += count;
      } else {
        rendered += character;
        column += 1;
      }
    }
    output.push({ ...token, content: rendered });
  }
  return output;
}

function fontAttributes(style) {
  const attributes = [`fill="${escapeAttribute(style.color)}"`];
  if (style.fontWeight !== undefined) {
    attributes.push(`font-weight="${style.fontWeight}"`);
  } else if ((style.fontStyle ?? 0) & 2) {
    attributes.push('font-weight="700"');
  }
  if ((style.fontStyle ?? 0) & 1) {
    attributes.push('font-style="italic"');
  }
  if ((style.fontStyle ?? 0) & 4) {
    attributes.push('text-decoration="underline"');
  }
  return attributes.join(" ");
}

function renderTspan(token, x) {
  if (token.content === "") {
    return "";
  }
  const kind = token.semanticKind ?? "syntax";
  return `<tspan x="${x}" data-kind="${escapeAttribute(kind)}" ${fontAttributes(renderedStyle(token))}>${escapeText(token.content)}</tspan>`;
}

function displayColumns(tokens) {
  return tokens.reduce((count, token) => count + Array.from(token.content).length, 0);
}

function validateTokenLines(tokenLines, sourceLines) {
  assert.equal(tokenLines.length, sourceLines.length, "tokenizer changed source line count");
  for (let index = 0; index < sourceLines.length; index += 1) {
    const rendered = tokenLines[index].map((token) => token.content).join("");
    assert.equal(rendered, sourceLines[index], `tokenizer changed source line ${index + 1}`);
  }
}

function selectedBounds(lineCount, range) {
  if (range === undefined) {
    return { start: 1, end: lineCount };
  }
  if (range.end > lineCount) {
    fail(`line range ${range.start}:${range.end} exceeds ${lineCount}-line input`);
  }
  return range;
}

function selectedRanges(lineCount, lineRange, segments) {
  if (segments !== undefined) {
    return segments.map((range) => selectedBounds(lineCount, range));
  }
  return [selectedBounds(lineCount, lineRange)];
}

export async function createRenderer() {
  const grammarBytes = await readFile(GRAMMAR_PATH);
  const grammarSha256 = sha256(grammarBytes);
  if (grammarSha256 !== EXPECTED_GRAMMAR_SHA256) {
    throw new Error(
      `MLIR grammar digest mismatch: expected ${EXPECTED_GRAMMAR_SHA256}, got ${grammarSha256}`,
    );
  }
  const rawGrammar = JSON.parse(grammarBytes.toString("utf8"));
  if (rawGrammar.scopeName !== "source.mlir") {
    throw new Error(`unexpected MLIR grammar scope: ${rawGrammar.scopeName}`);
  }
  // The upstream display name is "MLIR". Shiki resolves language IDs by the
  // registration name, so normalize the copied grammar to the lowercase CLI ID.
  const mlirGrammar = { ...rawGrammar, name: "mlir" };
  const highlighter = await createHighlighter({
    themes: [paperTheme],
    langs: [mlirGrammar, ...BUILTIN_LANGUAGES],
  });
  return {
    grammarSha256,
    highlighter,
    dispose() {
      highlighter.dispose();
    },
  };
}

export function renderSource(renderer, source, options) {
  const normalizedSource = normalizeNewlines(source);
  const sourceLines = normalizedSource.split("\n");
  const result = renderer.highlighter.codeToTokens(normalizedSource, {
    lang: options.lang,
    theme: paperTheme.name,
    includeExplanation: "scopeName",
  });
  validateTokenLines(result.tokens, sourceLines);
  const semanticLines = applySemanticOverrides(
    result.tokens,
    options.lang,
    options.accentDialects,
  );
  validateTokenLines(semanticLines, sourceLines);

  const ranges = selectedRanges(sourceLines.length, options.lines, options.segments);
  const selected = [];
  for (let rangeIndex = 0; rangeIndex < ranges.length; rangeIndex += 1) {
    const range = ranges[rangeIndex];
    if (rangeIndex > 0 && range.start > ranges[rangeIndex - 1].end + 1) {
      selected.push({
        originalLine: undefined,
        omission: true,
        tokens: [{
          content: "...",
          color: colors.lineNumber,
          fontStyle: 0,
          scopes: [],
          semanticKind: "omission",
        }],
      });
    }
    selected.push(...semanticLines
      .slice(range.start - 1, range.end)
      .map((tokens, index) => ({
        originalLine: range.start + index,
        omission: false,
        tokens: expandTabs(tokens, options.tabWidth),
      })));
  }

  const lineHeight = options.fontSize * options.lineHeight;
  const characterWidth = options.fontSize * defaults.characterWidthEm;
  const characterAdvance = options.fontSize * defaults.characterAdvanceEm;
  const maxColumns = Math.max(0, ...selected.map((line) => displayColumns(line.tokens)));
  const lastSourceLine = Math.max(...ranges.map((range) => range.end));
  const lineNumberDigits = options.lineNumbers ? String(lastSourceLine).length : 0;
  const gutterWidth = options.lineNumbers
    ? (lineNumberDigits * characterWidth) + (options.fontSize * 1.35)
    : 0;
  const codeX = options.paddingX + gutterWidth;
  const width = Math.ceil(
    options.paddingX * 2
      + gutterWidth
      + maxColumns * characterWidth
      + options.fontSize * defaults.rightSafetyEm,
  );
  const height = Math.ceil(options.paddingY * 2 + selected.length * lineHeight);
  const firstBaseline = options.paddingY + options.fontSize * 0.88;

  const metadata = {
    renderer: `snippet2svg/${TOOL_VERSION}`,
    shiki: SHIKI_VERSION,
    grammarSha256: renderer.grammarSha256,
    sourceSha256: sha256(Buffer.from(source, "utf8")),
    language: options.lang,
    sourceLines: ranges.map((range) => `${range.start}:${range.end}`).join(","),
    tabWidth: options.tabWidth,
    accentDialects: options.accentDialects,
    fontWeight: options.fontWeight,
  };

  const svg = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeAttribute(`${options.lang} source code`)}">`,
    `  <metadata>${escapeText(JSON.stringify(metadata))}</metadata>`,
  ];
  if (!options.transparent) {
    svg.push(`  <rect x="0" y="0" width="${width}" height="${height}" fill="${colors.background}"/>`);
  }
  svg.push(`  <g xml:space="preserve" font-family="${escapeAttribute(options.fontFamily)}" font-size="${options.fontSize}" font-weight="${options.fontWeight}" font-variant-ligatures="none">`);
  for (let index = 0; index < selected.length; index += 1) {
    const line = selected[index];
    const y = firstBaseline + index * lineHeight;
    if (options.lineNumbers && line.originalLine !== undefined) {
      const numberX = codeX - options.fontSize * 0.75;
      svg.push(`    <text x="${numberX}" y="${y}" text-anchor="end" fill="${colors.lineNumber}">${line.originalLine}</text>`);
    }
    let column = 0;
    const spans = line.tokens.map((token) => {
      const tokenX = codeX + column * characterAdvance;
      column += Array.from(token.content).length;
      return renderTspan(token, tokenX);
    }).join("");
    svg.push(`    <text x="${codeX}" y="${y}" fill="${colors.text}">${spans}</text>`);
  }
  svg.push("  </g>", "</svg>", "");

  return {
    svg: svg.join("\n"),
    dump: selected.map((line) => ({
      line: line.originalLine,
      omission: line.omission,
      tokens: line.tokens.map((token) => ({
        content: token.content,
        color: renderedStyle(token).color,
        fontWeight: renderedStyle(token).fontWeight,
        fontStyle: token.fontStyle,
        semanticKind: token.semanticKind,
        scopes: token.scopes,
      })),
    })),
    layout: { width, height, maxColumns, lineHeight, codeX },
    metadata,
  };
}

export async function renderFile(renderer, options) {
  const inputPath = path.resolve(options.input);
  const outputPath = path.resolve(options.output ?? outputPathFor(options.input));
  if (inputPath === outputPath) {
    fail("output path must differ from input path");
  }

  const beforeStat = await stat(inputPath);
  const source = await readFile(inputPath, "utf8");
  const sourceHash = sha256(Buffer.from(source, "utf8"));
  const lang = options.lang ?? inferLanguage(inputPath);
  const rendered = renderSource(renderer, source, { ...options, lang });

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, rendered.svg, "utf8");

  const afterSource = await readFile(inputPath, "utf8");
  const afterStat = await stat(inputPath);
  assert.equal(sha256(Buffer.from(afterSource, "utf8")), sourceHash, "input content changed");
  assert.equal(afterStat.mtimeMs, beforeStat.mtimeMs, "input modification time changed");
  return { ...rendered, inputPath, outputPath };
}

async function main() {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
      process.stdout.write(HELP);
      return;
    }
    if (options.version) {
      process.stdout.write(`${TOOL_VERSION}\n`);
      return;
    }
    const renderer = await createRenderer();
    try {
      const result = await renderFile(renderer, options);
      if (options.dumpTokens) {
        process.stdout.write(`${JSON.stringify(result.dump, null, 2)}\n`);
      } else {
        process.stdout.write(`${result.outputPath}\n`);
      }
    } finally {
      renderer.dispose();
    }
  } catch (error) {
    process.stderr.write(`snippet2svg: ${error.message}\n`);
    process.exitCode = 1;
  }
}

const invokedPath = process.argv[1] === undefined ? undefined : pathToFileURL(path.resolve(process.argv[1])).href;
if (invokedPath === import.meta.url) {
  await main();
}
