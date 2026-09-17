export const colors = Object.freeze({
  background: "#FFFFFF",
  text: "#202124",
  indigo: "#292968",
  navy: "#003F6B",
  slate: "#4F596A",
  burgundy: "#8A3042",
  sage: "#3F673F",
  taupe: "#6B4D3E",
  lineNumber: "#686D75",
});

export const defaults = Object.freeze({
  accentDialects: ["merlin_iface"],
  fontFamily: "'DejaVu Sans Mono', 'Liberation Mono', Consolas, monospace",
  // Paper snippets use a medium face throughout. This is intentionally below
  // bold: it survives reduction and print without flattening the hierarchy
  // between ordinary code (500) and emphasized operations/types (700).
  fontWeight: 500,
  fontSize: 22,
  lineHeight: 1.35,
  paddingX: 18,
  paddingY: 14,
  tabWidth: 4,
  // Monospace fonts are usually close to 0.6em wide. The larger estimate and
  // one-em safety margin prevent clipping when PowerPoint substitutes a font.
  characterAdvanceEm: 0.6,
  characterWidthEm: 0.66,
  rightSafetyEm: 1,
});

export const semanticStyles = Object.freeze({
  accentOperation: Object.freeze({ color: colors.navy, fontWeight: 700 }),
  accentType: Object.freeze({ color: colors.sage, fontWeight: 700 }),
  ssa: Object.freeze({ color: colors.indigo, fontWeight: 500 }),
  builtinType: Object.freeze({ color: colors.slate }),
  omission: Object.freeze({ color: colors.lineNumber, fontWeight: 700 }),
});

// This is a deliberately restrained light TextMate theme. The vendored LLVM
// grammar supplies syntax classes. Merlin-specific emphasis is applied after
// tokenization by render.mjs and is configured through accentDialects above.
export const paperTheme = Object.freeze({
  name: "merlin-paper",
  type: "light",
  fg: colors.text,
  bg: colors.background,
  colors: {
    "editor.foreground": colors.text,
    "editor.background": colors.background,
  },
  settings: [
    {
      settings: {
        foreground: colors.text,
        background: colors.background,
      },
    },
    {
      scope: ["comment", "comment.line.double-slash.mlir"],
      settings: { foreground: colors.sage, fontStyle: "italic" },
    },
    {
      scope: ["string", "string.quoted.double.mlir"],
      settings: { foreground: colors.taupe },
    },
    {
      scope: ["constant.numeric", "constant.numeric.mlir"],
      settings: { foreground: colors.burgundy },
    },
    {
      scope: ["constant.language", "constant.language.mlir"],
      settings: { foreground: colors.burgundy },
    },
    {
      scope: ["variable", "variable.other.mlir", "variable.mlir"],
      settings: { foreground: colors.slate },
    },
    {
      scope: ["variable.other.enummember.mlir"],
      settings: { foreground: colors.slate },
    },
    {
      scope: ["entity.name.type", "entity.name.type.mlir", "storage.type"],
      settings: { foreground: colors.slate },
    },
    {
      scope: ["entity.name.function", "support.function", "support.function.builtin"],
      settings: { foreground: colors.navy },
    },
    {
      scope: ["keyword", "keyword.other.mlir", "keyword.control.mlir", "storage.modifier"],
      settings: { foreground: colors.indigo },
    },
    {
      scope: [
        "punctuation",
        "punctuation.other.mlir",
        "punctuation.bracket.angle.begin.mlir",
        "punctuation.bracket.angle.end.mlir",
        "punctuation.definition.string.begin.mlir",
        "punctuation.definition.string.end.mlir",
      ],
      settings: { foreground: colors.text },
    },
  ],
});
