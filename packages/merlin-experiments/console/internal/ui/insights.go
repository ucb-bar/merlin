package ui

import (
	"fmt"
	"image/color"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/ucb-bar/merlin/packages/merlin-experiments/console/internal/proto"
)

// The insight page (v): ten panels drawn by one renderer from the generic panel shape of
// sources/insight_common.py. Summary lines and table rows are selectable; Enter inspects the row's
// key through its source; s cycles the table orders a panel offers (its sorts).

var insightZone = map[string]color.Color{
	"performance": lipgloss.Color("#22d3ee"), "cases": lipgloss.Color("#2dd4bf"), "tools": lipgloss.Color("#f472b6"),
	"grader": lipgloss.Color("#4ade80"), "integrity": lipgloss.Color("#f87171"),
	"attempts": lipgloss.Color("#c084fc"), "certification": lipgloss.Color("#a3e635"),
	"focus": lipgloss.Color("#fbbf24"), "cost": lipgloss.Color("#93c5fd"),
	"library": lipgloss.Color("#fb923c"),
}

func init() {
	for id, c := range insightZone {
		zone[id] = c
	}
}

func isInsight(id string) bool {
	_, ok := insightZone[id]
	return ok
}

func levelColor(level string) color.Color {
	switch level {
	case "ok":
		return cGood
	case "info":
		return cInfo
	case "warn":
		return cWarn
	case "bad":
		return cBad
	}
	return cText
}

func levelGlyph(level string) string {
	switch level {
	case "ok":
		return paint(cGood, "✓")
	case "info":
		return paint(cInfo, "·")
	case "warn":
		return paint(cWarn, "▲")
	case "bad":
		return paint(cBad, "✖")
	}
	return " "
}

// insightRow is the selectable row of one summary line or table row: its source inspects a key; a
// row without one shows its own text.
func insightRow(id, key string, local ...string) Row {
	if key != "" {
		return Row{Kind: id, Key: key, Local: local}
	}
	return Row{Kind: "local", Key: id, Local: local}
}

// tableWidths fits the columns into w cells: the first column takes what the others leave; trailing
// columns are dropped (never squeezed below their header) when the panel is too narrow.
func tableWidths(t *proto.InsightTable, w int) []int {
	n := len(t.Columns)
	widths := make([]int, n)
	for i, c := range t.Columns {
		widths[i] = ansi.StringWidth(c)
	}
	for _, r := range t.Rows {
		for i, c := range r.Cells {
			if i < n {
				widths[i] = max(widths[i], min(ansi.StringWidth(c), 16))
			}
		}
	}
	first := 10
	if n > 0 {
		first = min(max(10, widths[0]), 36)
	}
	used, keep := first, 1
	for i := 1; i < n; i++ {
		if used+1+widths[i] > w {
			break
		}
		used += 1 + widths[i]
		keep++
	}
	out := widths[:keep]
	if keep > 0 {
		out[0] = max(6, w-(used-first))
	}
	return out
}

func tableLine(cells []string, widths []int, c color.Color, header bool) string {
	parts := make([]string, 0, len(widths))
	for i, wd := range widths {
		cell := ""
		if i < len(cells) {
			cell = clean(cells[i])
		}
		if i == 0 {
			cell = fit(ansi.Truncate(cell, wd, "…"), wd)
		} else {
			cell = strings.Repeat(" ", max(0, wd-ansi.StringWidth(cell))) + ansi.Truncate(cell, wd, "…")
		}
		if header {
			parts = append(parts, dim(cell))
		} else {
			parts = append(parts, paint(c, cell))
		}
	}
	return strings.Join(parts, " ")
}

func (m *Model) pInsight(st *proto.State, id string, w, h int, focused bool) (string, string, []string) {
	in, have := st.Insights[id]
	title := strings.ToUpper(id)
	if in.Title != "" {
		title = clean(in.Title)
	}
	lines, rows := []string{}, []Row{}
	if !have {
		lines = append(lines, dim("waiting for the data process…"))
		rows = append(rows, Row{Kind: "local", Key: id})
		return title, "", m.list(id, nil, rows, lines, h, focused)
	}
	if in.Error != "" {
		lines = append(lines, paint(cBad, "reader error: "+clean(in.Error)))
		rows = append(rows, Row{Kind: "local", Key: id, Local: []string{clean(in.Error)}})
	}
	warns, hint := 0, ""
	labelW := min(28, max(14, w*45/100))
	for _, l := range in.Summary {
		if !m.matches(l.Label + " " + l.Value) {
			continue
		}
		if l.Level == "warn" || l.Level == "bad" {
			warns++
		}
		value := paint(levelColor(l.Level), clean(l.Value))
		lines = append(lines, levelGlyph(l.Level)+" "+fit(dim(clean(l.Label)), labelW)+" "+value)
		rows = append(rows, insightRow(id, l.Key, clean(l.Label)+": "+clean(l.Value)))
	}
	for _, s := range in.Series {
		if len(s.V) < 2 || w < 30 {
			continue
		}
		vals := make([]*float64, len(s.V))
		for i := range s.V {
			vals[i] = &s.V[i]
		}
		last := s.V[len(s.V)-1]
		label := fit(dim(clean(s.Label)), labelW)
		tail := " " + paint(cText, fmt.Sprintf("%.4g", last))
		spark := sparkline(vals, max(4, w-labelW-4-ansi.StringWidth(tail)), zone[id])
		lines = append(lines, "  "+label+" "+spark+tail)
		rows = append(rows, insightRow(id, "", clean(s.Label)))
	}
	if t := in.Table; t != nil && len(t.Rows) > 0 {
		widths := tableWidths(t, w-3) // the list cursor, the level glyph and a space precede a row
		lines = append(lines, "  "+tableLine(t.Columns, widths, cText, true))
		rows = append(rows, Row{Kind: "local", Key: id, Local: in.Notes})
		ordered, label := insightOrder(t.Rows, in.Sorts, m.sortMode[id])
		if label != "" {
			hint = "s: " + label
		}
		for _, r := range ordered {
			if !m.matches(strings.Join(r.Cells, " ")) {
				continue
			}
			lines = append(lines, levelGlyph(r.Level)+" "+tableLine(r.Cells, widths, levelColor(r.Level), false))
			rows = append(rows, insightRow(id, r.Key, strings.Join(r.Cells, " · ")))
		}
	}
	// Notes only when the panel has room left (zoomed, or a short panel).
	if len(lines)+len(in.Notes) <= h-2 {
		for _, n := range in.Notes {
			lines = append(lines, dim("  · "+clean(n)))
			rows = append(rows, Row{Kind: "local", Key: id, Local: []string{clean(n)}})
		}
	}
	if warns > 0 {
		hint = strings.TrimPrefix(hint+fmt.Sprintf(" · %d to look at", warns), " · ")
	}
	return title, hint, m.list(id, nil, rows, lines, h, focused)
}

// insightOrder applies the panel's mode-th sort (cycled by s) to its table rows; rows an order omits
// keep their place after the ordered ones. Without sorts the rows stay as sent.
func insightOrder(rows []proto.InsightRow, sorts []proto.InsightSort, mode int) ([]proto.InsightRow, string) {
	if len(sorts) == 0 {
		return rows, ""
	}
	s := sorts[mode%len(sorts)]
	out := make([]proto.InsightRow, 0, len(rows))
	used := make([]bool, len(rows))
	for _, i := range s.Order {
		if i >= 0 && i < len(rows) && !used[i] {
			used[i] = true
			out = append(out, rows[i])
		}
	}
	for i, r := range rows {
		if !used[i] {
			out = append(out, r)
		}
	}
	return out, clean(s.Label)
}

// OpenPage selects the page shown first ("insights" for the insight page, anything else the main page).
func (m *Model) OpenPage(page string) {
	if page == "insights" && m.page == 0 {
		m.togglePage()
		m.status = ""
	}
}

// insightGrid lays out the insight page: four rows at full width, five when narrow; CASES, the one
// wide table, has the second row to itself either way.
func (m *Model) insightGrid(place func(id string, x, y, pw, ph int), x0, w, h int) {
	mw := w - x0
	cols := func(y, rh int, ids []string, fracs []int) {
		x := x0
		for i, id := range ids {
			cw := mw * fracs[i] / 100
			if i == len(ids)-1 {
				cw = w - x
			}
			place(id, x, y, cw, rh)
			x += cw
		}
	}
	if mw >= 150 {
		r1 := max(9, h*25/100)
		r2 := max(9, h*35/100)
		r3 := max(6, h*21/100)
		r4 := max(5, h-r1-r2-r3)
		cols(0, r1, []string{"performance", "tools", "grader"}, []int{36, 30, 34})
		cols(r1, r2, []string{"cases"}, []int{100})
		cols(r1+r2, r3, []string{"certification", "integrity", "cost"}, []int{40, 32, 28})
		cols(r1+r2+r3, r4, []string{"library", "attempts", "focus"}, []int{40, 30, 30})
		return
	}
	rh := max(5, h/5)
	pairs := [][]string{{"performance", "tools"}, {"cases"}, {"grader", "certification"}, {"integrity", "cost"},
		{"library", "attempts", "focus"}}
	for i, ids := range pairs {
		y := i * rh
		height := rh
		if i == len(pairs)-1 {
			height = max(3, h-y)
		}
		fracs := []int{50, 50}
		if len(ids) == 3 {
			fracs = []int{34, 33, 33}
		}
		if len(ids) == 1 {
			fracs = []int{100}
		}
		cols(y, height, ids, fracs)
	}
}
