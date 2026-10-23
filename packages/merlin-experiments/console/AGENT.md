# AGENT.md — packages/merlin-experiments/console

## Purpose

The Bubble Tea v2 frontend of the experiment command console (`merlin-console`). It spawns the
read-only data process `python -m merlin_experiments.tracking.console.serve` and renders its
`merlin.console.snapshot.v1` JSON lines: every panel in one view, drill-down inspection, zoom, filter,
sort, time windows and experiment switching.

## Layout

- `main.go` — flags, the data-process bridge, `--render-once WxH` (print one frame, no TTY) and
  `--replay FILE` (render recorded JSON lines).
- `internal/proto` — typed mirror of the section contract in
  `src/merlin_experiments/tracking/console/sources/__init__.py`; delta snapshots merge per section.
- `internal/ui` — the model (keys, mouse, layout, inspector), the panels and the widgets (exact-size
  boxes, bars, sparklines, braille plots, Gantt).

## Build and test

```sh
merlin-experiment console --build          # go build -trimpath -o <out/build>/console/merlin-console .
go vet ./... && go test ./...              # from this directory (Go >= 1.26)
merlin-console --python .venv/bin/python --operator-private --render-once 210x56
```

## Invariants

- Render only what the data process sent; never read run records here.
- Every frame is exactly the terminal size; record text passes `clean()` (no control characters).
- Missing numbers render `—`, never `0`. Dependencies are pinned in `go.mod`/`go.sum`
  (charm.land/bubbletea/v2 v2.1.0, charm.land/lipgloss/v2 v2.0.6, charmbracelet/x/ansi v0.11.9).
