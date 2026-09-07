import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test, { after, before } from "node:test";

import {
  EXPECTED_GRAMMAR_SHA256,
  createRenderer,
  inferLanguage,
  parseArgs,
  parseLineRange,
  parseSegments,
  renderFile,
  renderSource,
} from "../render.mjs";
import { colors, defaults } from "../theme.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.join(HERE, "fixture.mlir");

let renderer;
let source;
const fixtureSources = new Map();

function baseOptions(overrides = {}) {
  return {
    lang: "mlir",
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
    ...overrides,
  };
}

function decodeXmlText(value) {
  return value
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&apos;", "'")
    .replaceAll("&amp;", "&");
}

function renderedTextByLine(svg) {
  const lines = [];
  const textPattern = /<text x="[^\"]+" y="[^\"]+" fill="#[0-9A-F]+">(.*?)<\/text>/g;
  for (const match of svg.matchAll(textPattern)) {
    const withoutTags = match[1].replace(/<[^>]+>/g, "");
    lines.push(decodeXmlText(withoutTags));
  }
  return lines;
}

function token(result, content) {
  return result.dump.flatMap((line) => line.tokens).find((item) => item.content === content);
}

function hash(value) {
  return createHash("sha256").update(value).digest("hex");
}

before(async () => {
  renderer = await createRenderer();
  source = await readFile(FIXTURE, "utf8");
  for (const name of ["fixture.yaml", "fixture.py", "fixture.jsonc", "fixture.sh", "fixture.txt"]) {
    fixtureSources.set(name, await readFile(path.join(HERE, name), "utf8"));
  }
});

after(() => renderer.dispose());

test("uses the reviewed LLVM MLIR grammar bytes", () => {
  assert.equal(renderer.grammarSha256, EXPECTED_GRAMMAR_SHA256);
});

test("renders fixture as native SVG", () => {
  const result = renderSource(renderer, source, baseOptions());
  assert.match(result.svg, /<svg /);
  assert.match(result.svg, /<text /);
  assert.match(result.svg, /<tspan /);
  assert.match(result.svg, new RegExp(`<rect[^>]+fill="${colors.background}"`));
});

test("uses medium weight for ordinary paper code", () => {
  const result = renderSource(renderer, source, baseOptions());
  assert.match(result.svg, new RegExp(`<g[^>]+font-weight="${defaults.fontWeight}"`));
  assert.equal(defaults.fontWeight, 500);
});

test("escapes XML-sensitive MLIR syntax without changing displayed text", () => {
  const input = '%0 = "test.op"() {note = "A & B"} : () -> tensor<2xi8>\n';
  const result = renderSource(renderer, input, baseOptions());
  assert.match(result.svg, /&amp;/);
  assert.match(result.svg, /&lt;/);
  assert.match(result.svg, /&gt;/);
  assert.equal(renderedTextByLine(result.svg).join("\n"), input);
});

test("preserves complete source text and indentation", () => {
  const result = renderSource(renderer, source.trimEnd(), baseOptions());
  assert.equal(renderedTextByLine(result.svg).join("\n"), source.trimEnd());
});

test("selects inclusive source lines", () => {
  const result = renderSource(renderer, source, baseOptions({ lines: { start: 6, end: 8 } }));
  assert.deepEqual(renderedTextByLine(result.svg), source.split("\n").slice(5, 8));
});

test("renders ordered source segments with an explicit omission", () => {
  const result = renderSource(renderer, source, baseOptions({
    segments: [{ start: 2, end: 3 }, { start: 10, end: 11 }],
    lineNumbers: true,
  }));
  const lines = renderedTextByLine(result.svg);
  assert.deepEqual(lines, [source.split("\n")[1], source.split("\n")[2], "...",
    source.split("\n")[9], source.split("\n")[10]]);
  assert.match(result.svg, />2<\/text>/);
  assert.match(result.svg, />10<\/text>/);
  assert.equal(result.metadata.sourceLines, "2:3,10:11");
});

test("rejects malformed, overlapping, and mixed selections", () => {
  assert.throws(() => parseSegments("1:2,"), /expected A:B,C:D/);
  assert.throws(() => parseSegments("1:3,3:4"), /ordered and non-overlapping/);
  assert.throws(() => parseArgs([
    "input.mlir", "--lines", "1:2", "--segments", "4:5,8:9",
  ]), /mutually exclusive/);
});

test("line numbers retain original indices", () => {
  const result = renderSource(renderer, source, baseOptions({
    lines: { start: 10, end: 12 },
    lineNumbers: true,
  }));
  assert.match(result.svg, />10<\/text>/);
  assert.match(result.svg, />11<\/text>/);
  assert.match(result.svg, />12<\/text>/);
  assert.doesNotMatch(result.svg, />1<\/text>/);
});

test("default omits line-number gutter", () => {
  const result = renderSource(renderer, source, baseOptions());
  assert.doesNotMatch(result.svg, /text-anchor="end"/);
});

test("rejects malformed and out-of-bounds ranges", () => {
  assert.throws(() => parseLineRange("6-18"), /expected START:END/);
  assert.throws(() => parseLineRange("0:2"), /1 <= START <= END/);
  assert.throws(
    () => renderSource(renderer, source, baseOptions({ lines: { start: 1, end: 999 } })),
    /exceeds/,
  );
});

test("recognizes and emphasizes Merlin operations", () => {
  const result = renderSource(renderer, source, baseOptions());
  const item = token(result, "merlin_iface.matmul");
  assert.ok(item);
  assert.equal(item.semanticKind, "accentOperation");
  assert.equal(item.color, colors.navy);
});

test("does not confuse a Merlin attribute key with an operation", () => {
  const result = renderSource(renderer, source, baseOptions());
  const item = token(result, "merlin_iface.target");
  assert.ok(item);
  assert.notEqual(item.semanticKind, "accentOperation");
});

test("recognizes SSA values", () => {
  const result = renderSource(renderer, source, baseOptions());
  const item = token(result, "%acc0");
  assert.ok(item);
  assert.equal(item.semanticKind, "ssa");
  assert.equal(item.color, colors.indigo);
});

test("recognizes Merlin and builtin types", () => {
  const result = renderSource(renderer, source, baseOptions());
  const custom = token(result, "!merlin_iface.acc");
  const tensor = token(result, "tensor");
  const i32 = token(result, "i32");
  assert.equal(custom?.semanticKind, "accentType");
  assert.equal(custom?.color, colors.sage);
  assert.equal(tensor?.semanticKind, "builtinType");
  assert.equal(tensor?.color, colors.slate);
  assert.equal(i32?.semanticKind, "builtinType");
  assert.equal(i32?.color, colors.slate);
});

test("strings and numbers use the paper palette", () => {
  const result = renderSource(renderer, source, baseOptions());
  assert.ok(result.dump.flatMap((line) => line.tokens).some(
    (item) => item.content.includes("gemmini") && item.color === colors.taupe,
  ));
  assert.ok(result.dump.flatMap((line) => line.tokens).some(
    (item) => item.content === "32" && item.color === colors.burgundy,
  ));
});

test("transparent mode omits only the background", () => {
  const result = renderSource(renderer, source, baseOptions({ transparent: true }));
  assert.doesNotMatch(result.svg, /<rect /);
  assert.match(result.svg, /<text /);
});

test("output is deterministic", () => {
  const one = renderSource(renderer, source, baseOptions()).svg;
  const two = renderSource(renderer, source, baseOptions()).svg;
  assert.equal(one, two);
});

test("generated SVG has no browser-only or raster content", () => {
  const svg = renderSource(renderer, source, baseOptions()).svg;
  for (const forbidden of ["foreignObject", "data:image", "base64", "<image", "<script", "<filter"]) {
    assert.equal(svg.includes(forbidden), false, forbidden);
  }
});

test("renderFile leaves input bytes and mtime unchanged", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "snippet2svg-"));
  const input = path.join(directory, "input.mlir");
  const output = path.join(directory, "output.svg");
  await writeFile(input, source, "utf8");
  const beforeBytes = await readFile(input);
  const beforeStat = await stat(input);
  await renderFile(renderer, { ...baseOptions(), input, output });
  const afterBytes = await readFile(input);
  const afterStat = await stat(input);
  assert.equal(hash(afterBytes), hash(beforeBytes));
  assert.equal(afterStat.mtimeMs, beforeStat.mtimeMs);
});

test("tabs expand deterministically for display", () => {
  const result = renderSource(renderer, "\t%0 = test.op\n", baseOptions({ tabWidth: 4 }));
  assert.equal(renderedTextByLine(result.svg)[0].startsWith("    %0"), true);
});

test("layout includes a conservative longest-line allowance", () => {
  const input = "x".repeat(120);
  const result = renderSource(renderer, input, baseOptions({ lang: "text" }));
  const minimum = 120 * defaults.fontSize * defaults.characterWidthEm;
  assert.ok(result.layout.width > minimum);
});

test("infers all paper snippet languages", () => {
  assert.equal(inferLanguage("capsule.mlir"), "mlir");
  assert.equal(inferLanguage("capsule.yaml"), "yaml");
  assert.equal(inferLanguage("derive.py"), "python");
  assert.equal(inferLanguage("facts.jsonc"), "jsonc");
  assert.equal(inferLanguage("run.sh"), "shellscript");
  assert.equal(inferLanguage("kernel.ll"), "llvm");
  assert.equal(inferLanguage("verdict.txt"), "text");
});

test("general snippet languages render through Shiki", () => {
  for (const [name, lang] of [
    ["fixture.yaml", "yaml"],
    ["fixture.py", "python"],
    ["fixture.jsonc", "jsonc"],
    ["fixture.sh", "shellscript"],
    ["fixture.txt", "text"],
  ]) {
    const text = fixtureSources.get(name);
    assert.ok(text, `missing fixture: ${name}`);
    const result = renderSource(renderer, text, baseOptions({ lang }));
    assert.match(result.svg, /<svg /);
    assert.equal(
      renderedTextByLine(result.svg).join("\n"),
      text.replaceAll("\r\n", "\n").replaceAll("\r", "\n"),
    );
  }
});

test("CLI parser accepts publication controls", () => {
  const options = parseArgs([
    "input.mlir",
    "--lines", "6:18",
    "--line-numbers",
    "--font-size", "20",
    "--font-family", "IBM Plex Mono",
    "--tab-width", "2",
    "--accent-dialect", "gemmini",
    "-o", "output.svg",
  ]);
  assert.deepEqual(options.lines, { start: 6, end: 18 });
  assert.equal(options.lineNumbers, true);
  assert.equal(options.fontSize, 20);
  assert.equal(options.fontFamily, "IBM Plex Mono");
  assert.equal(options.tabWidth, 2);
  assert.deepEqual(options.accentDialects, ["merlin_iface", "gemmini"]);
  assert.equal(options.output, "output.svg");
});
