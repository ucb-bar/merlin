# Vendored MLIR TextMate grammar

`grammar.json` is copied without semantic changes from the official
[`llvm/vscode-mlir`](https://github.com/llvm/vscode-mlir) repository.

- Upstream commit: `1ea9567704457cc9dfd508a1a3d8f11beff5a7b3`
- Upstream path: `grammar.json`
- TextMate scope: `source.mlir`
- SHA-256: `5942b4066a1f9c626547acddc1237b205e3a44bf6f27988ad30e7753dd0e82aa`
- License: Apache-2.0 WITH LLVM-exception
- License copy: `LICENSE.llvm.txt`

The vscode-mlir integration workflow copies this grammar from
`mlir/utils/textmate/mlir.json` in llvm-project. The same grammar is present in
this repository's LLVM checkout at revision
`a47bddccec30255619bb8c37fa59700e661d4e66`.

Rendering never contacts the network. To refresh the vendored files after
reviewing and updating the pinned commit and hashes in `update-grammar.mjs`, run:

```sh
npm run update:grammar
```

The update command refuses files whose digest differs from the reviewed hash.
