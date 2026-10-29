package ui

import (
	"fmt"
	"image/color"
	"math"
	"strings"
	"time"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// Palette: one hue per zone so each panel reads as its own region (dark-terminal tuned).
var (
	cText    = lipgloss.Color("#d8dee9")
	cDim     = lipgloss.Color("#6b7280")
	cFaint   = lipgloss.Color("#3b4252")
	cBorder  = lipgloss.Color("#4c566a")
	cFocus   = lipgloss.Color("#f5f5f5")
	cGood    = lipgloss.Color("#a3e635")
	cWarn    = lipgloss.Color("#facc15")
	cBad     = lipgloss.Color("#f87171")
	cInfo    = lipgloss.Color("#60a5fa")
	cAgent   = lipgloss.Color("#f472b6")
	cL2      = lipgloss.Color("#38bdf8")
	cL3      = lipgloss.Color("#4ade80")
	cBuild   = lipgloss.Color("#fb923c")
	cVerify  = lipgloss.Color("#a78bfa")
	cCalib   = lipgloss.Color("#fde047")
	cHost    = lipgloss.Color("#7dd3fc")
	cQueue   = lipgloss.Color("#94a3b8")
	cCache   = lipgloss.Color("#818cf8")
	cOutput  = lipgloss.Color("#f472b6")
	cReason  = lipgloss.Color("#fbbf24")
	cFresh   = lipgloss.Color("#22d3ee")
	cHeading = lipgloss.Color("#e5e7eb")
)

// Zone colours per panel id.
var zone = map[string]color.Color{
	"experiments": lipgloss.Color("#93c5fd"), "progress": lipgloss.Color("#4ade80"),
	"stages": lipgloss.Color("#c084fc"), "alerts": lipgloss.Color("#f87171"),
	"gantt": lipgloss.Color("#f0abfc"), "agents": lipgloss.Color("#f472b6"),
	"tokens": lipgloss.Color("#fbbf24"), "cpu": lipgloss.Color("#38bdf8"),
	"perf": lipgloss.Color("#22d3ee"), "capsules": lipgloss.Color("#a3e635"),
	"passes": lipgloss.Color("#fb923c"), "events": lipgloss.Color("#cbd5e1"),
	"inspect": lipgloss.Color("#f5f5f5"),
}

// Kind colours for resources, lanes and agents.
var kindColor = map[string]color.Color{
	"agent": cAgent, "L2": cL2, "L3": cL3, "build": cBuild, "verify": cVerify, "calibration": cCalib,
	"host": cHost, "unknown": cDim, "idle": cFaint, "setup": cBuild, "queue": cQueue, "turn": cAgent,
	"tool": cBuild, "wait": cQueue,
}

var kindGlyph = map[string]string{
	"agent": "A", "L2": "2", "L3": "3", "build": "B", "verify": "V", "calibration": "C", "host": "H",
	"unknown": "?", "idle": "·",
}

// tailFit keeps the end of an identifier (where run versions differ) when it must be shortened.
func tailFit(s string, w int) string {
	if ansi.StringWidth(s) <= w {
		return fit(s, w)
	}
	return fit(ansi.TruncateLeft(s, ansi.StringWidth(s)-w+1, "…"), w)
}

func fg(c color.Color) lipgloss.Style { return lipgloss.NewStyle().Foreground(c) }

// paint, bold and dim colour one line of text. Rendering a style per cell dominated the frame cost,
// so each (colour, weight) is rendered once and its escape prefix and suffix are reused.
func paint(c color.Color, s string) string { return styled(c, false, s) }

func bold(c color.Color, s string) string { return styled(c, true, s) }

func dim(s string) string { return styled(cDim, false, s) }

type styleKey struct {
	c    color.Color
	bold bool
}

var styleCache = map[styleKey][2]string{}

const styleProbe = "\uE000"

func styled(c color.Color, b bool, s string) string {
	if s == "" {
		return ""
	}
	st := fg(c)
	if b {
		st = st.Bold(true)
	}
	if strings.ContainsRune(s, '\n') {
		return st.Render(s)
	}
	k := styleKey{c, b}
	parts, ok := styleCache[k]
	if !ok {
		r := st.Render(styleProbe)
		i := strings.Index(r, styleProbe)
		if i < 0 {
			return st.Render(s)
		}
		parts = [2]string{r[:i], r[i+len(styleProbe):]}
		styleCache[k] = parts
	}
	return parts[0] + s + parts[1]
}

// clean strips control characters from record text so it cannot drive the terminal.
func clean(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r == '\t':
			b.WriteRune(' ')
		case r < 0x20 || r == 0x7f || (r >= 0x80 && r < 0xa0):
			b.WriteRune(' ')
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// fit truncates (ANSI-aware) and pads a styled line to exactly w cells.
func fit(s string, w int) string {
	if w <= 0 {
		return ""
	}
	if ansi.StringWidth(s) > w {
		s = ansi.Truncate(s, w, "…")
	}
	if pad := w - ansi.StringWidth(s); pad > 0 {
		s += strings.Repeat(" ", pad)
	}
	return s
}

// box draws a rounded panel of exactly w x h cells. The title carries the zone colour; a focused
// panel gets a bright border. Body lines are fitted, extra lines dropped, missing lines blank.
func box(id, title, hint string, body []string, w, h int, focused bool) string {
	if w < 4 || h < 2 {
		return strings.Repeat(fit("", w)+"\n", max(0, h-1)) + fit("", w)
	}
	bc := cBorder
	if focused {
		bc = cFocus
	}
	b := fg(bc)
	zc := zone[id]
	if zc == nil {
		zc = cHeading
	}
	label := " " + bold(zc, title) + " "
	if hint != "" {
		label += dim(hint) + " "
	}
	inner := w - 2
	label = ansi.Truncate(label, max(0, inner-1), "…")
	top := b.Render("╭─") + label + b.Render(strings.Repeat("─", max(0, inner-1-ansi.StringWidth(label)))+"╮")
	lines := []string{top}
	for i := 0; i < h-2; i++ {
		line := ""
		if i < len(body) {
			line = body[i]
		}
		lines = append(lines, b.Render("│")+fit(line, inner)+b.Render("│"))
	}
	lines = append(lines, b.Render("╰"+strings.Repeat("─", inner)+"╯"))
	return strings.Join(lines, "\n")
}

// --- numbers ---------------------------------------------------------------------------------

func f(p *float64) (float64, bool) {
	if p == nil || math.IsNaN(*p) || math.IsInf(*p, 0) {
		return 0, false
	}
	return *p, true
}

// human renders a count compactly: 1.2k, 3.4M, 5.6G; nil is an em dash, never 0.
func human(p *float64) string {
	v, ok := f(p)
	if !ok {
		return "—"
	}
	return humanF(v)
}

func humanF(v float64) string {
	a := math.Abs(v)
	switch {
	case a >= 1e9:
		return fmt.Sprintf("%.2fG", v/1e9)
	case a >= 1e6:
		return fmt.Sprintf("%.2fM", v/1e6)
	case a >= 1e4:
		return fmt.Sprintf("%.1fk", v/1e3)
	case a == math.Trunc(a) || a >= 100:
		return fmt.Sprintf("%.0f", v)
	default:
		return fmt.Sprintf("%.2f", v)
	}
}

func pct(p *float64, digits int) string {
	v, ok := f(p)
	if !ok {
		return "—"
	}
	return fmt.Sprintf("%.*f%%", digits, v)
}

func intOr(p *float64, dash string) string {
	v, ok := f(p)
	if !ok {
		return dash
	}
	return fmt.Sprintf("%.0f", v)
}

func dur(seconds float64) string {
	if seconds < 0 || math.IsNaN(seconds) {
		return "—"
	}
	switch {
	case seconds >= 86400:
		return fmt.Sprintf("%dd%02dh", int(seconds/86400), int(math.Mod(seconds, 86400)/3600))
	case seconds >= 3600:
		return fmt.Sprintf("%dh%02dm", int(seconds/3600), int(math.Mod(seconds, 3600)/60))
	case seconds >= 60:
		return fmt.Sprintf("%dm%02ds", int(seconds/60), int(math.Mod(seconds, 60)))
	default:
		return fmt.Sprintf("%.1fs", seconds)
	}
}

func durP(p *float64) string {
	v, ok := f(p)
	if !ok {
		return "—"
	}
	return dur(v)
}

func ago(now float64, p *float64) string {
	v, ok := f(p)
	if !ok || now <= 0 {
		return "—"
	}
	return dur(now-v) + " ago"
}

func clock(p *float64) string {
	v, ok := f(p)
	if !ok {
		return "--:--:--"
	}
	return time.Unix(int64(v), 0).UTC().Format("15:04:05")
}

// --- bars and sparklines -----------------------------------------------------------------------

var eighths = []string{"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"}

// bar is a fractional horizontal bar of exactly w cells; frac outside [0,1] is clamped.
func bar(frac float64, w int, c color.Color) string {
	if w <= 0 {
		return ""
	}
	if math.IsNaN(frac) || frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	cells := frac * float64(w)
	full := int(cells)
	part := int((cells - float64(full)) * 8)
	s := strings.Repeat("█", full)
	if full < w && part > 0 {
		s += eighths[part]
	}
	used := ansi.StringWidth(s)
	return paint(c, s) + paint(cFaint, strings.Repeat("·", max(0, w-used)))
}

// stackBar draws consecutive segments (values out of total) with their colours.
func stackBar(values []float64, colors []color.Color, total float64, w int) string {
	if w <= 0 {
		return ""
	}
	if total <= 0 {
		return paint(cFaint, strings.Repeat("·", w))
	}
	var b strings.Builder
	used := 0
	for i, v := range values {
		n := int(math.Round(v / total * float64(w)))
		if v > 0 && n == 0 {
			n = 1
		}
		n = min(n, w-used)
		if n <= 0 {
			continue
		}
		b.WriteString(paint(colors[i], strings.Repeat("█", n)))
		used += n
	}
	b.WriteString(paint(cFaint, strings.Repeat("·", max(0, w-used))))
	return b.String()
}

var sparks = []rune("▁▂▃▄▅▆▇█")

// sparkline resamples values to w cells (nil gaps stay blank).
func sparkline(values []*float64, w int, c color.Color) string {
	if w <= 0 {
		return ""
	}
	pts := resample(values, w)
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, p := range pts {
		if v, ok := f(p); ok {
			lo, hi = math.Min(lo, v), math.Max(hi, v)
		}
	}
	var b strings.Builder
	for _, p := range pts {
		v, ok := f(p)
		if !ok {
			b.WriteRune(' ')
			continue
		}
		idx := 0
		if hi > lo {
			idx = int((v - lo) / (hi - lo) * float64(len(sparks)-1))
		}
		b.WriteRune(sparks[idx])
	}
	return paint(c, b.String())
}

func resample(values []*float64, w int) []*float64 {
	if len(values) == 0 {
		return make([]*float64, w)
	}
	out := make([]*float64, w)
	if len(values) <= w {
		copy(out[w-len(values):], values)
		return out
	}
	for i := 0; i < w; i++ {
		out[i] = values[int(float64(i)*float64(len(values)-1)/float64(max(1, w-1)))]
	}
	return out
}

// --- braille plotting ----------------------------------------------------------------------------

// Plot is a braille canvas: each cell holds a 2x4 dot matrix; the last series to set a dot colours it.
type Plot struct {
	W, H   int
	dots   [][]uint8
	colors [][]color.Color
}

func NewPlot(w, h int) *Plot {
	p := &Plot{W: max(1, w), H: max(1, h)}
	p.dots = make([][]uint8, p.H)
	p.colors = make([][]color.Color, p.H)
	for i := range p.dots {
		p.dots[i] = make([]uint8, p.W)
		p.colors[i] = make([]color.Color, p.W)
	}
	return p
}

var brailleBits = [4][2]uint8{{0x01, 0x08}, {0x02, 0x10}, {0x04, 0x20}, {0x40, 0x80}}

// Set lights dot (x, y) in dot coordinates: x in [0, 2W), y in [0, 4H) with y=0 at the top.
func (p *Plot) Set(x, y int, c color.Color) {
	if x < 0 || y < 0 || x >= 2*p.W || y >= 4*p.H {
		return
	}
	cx, cy := x/2, y/4
	p.dots[cy][cx] |= brailleBits[y%4][x%2]
	p.colors[cy][cx] = c
}

// Line draws a straight line between two dots.
func (p *Plot) Line(x0, y0, x1, y1 int, c color.Color) {
	dx, dy := abs(x1-x0), -abs(y1-y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	e := dx + dy
	for {
		p.Set(x0, y0, c)
		if x0 == x1 && y0 == y1 {
			return
		}
		e2 := 2 * e
		if e2 >= dy {
			e += dy
			x0 += sx
		}
		if e2 <= dx {
			e += dx
			y0 += sy
		}
	}
}

// Series plots (xs, ys) scaled into [xlo,xhi] x [ylo,yhi], connecting consecutive points.
func (p *Plot) Series(xs, ys []float64, xlo, xhi, ylo, yhi float64, c color.Color) {
	if xhi <= xlo {
		xhi = xlo + 1
	}
	if yhi <= ylo {
		yhi = ylo + 1
	}
	px := func(x float64) int { return int((x - xlo) / (xhi - xlo) * float64(2*p.W-1)) }
	py := func(y float64) int { return 4*p.H - 1 - int((y-ylo)/(yhi-ylo)*float64(4*p.H-1)) }
	prevOK := false
	var lx, ly int
	for i := range xs {
		if i >= len(ys) || math.IsNaN(ys[i]) || math.IsNaN(xs[i]) {
			prevOK = false
			continue
		}
		x, y := px(xs[i]), py(ys[i])
		if prevOK {
			p.Line(lx, ly, x, y, c)
		} else {
			p.Set(x, y, c)
		}
		lx, ly, prevOK = x, y, true
	}
}

func (p *Plot) Lines() []string {
	out := make([]string, p.H)
	for y := 0; y < p.H; y++ {
		var b strings.Builder
		for x := 0; x < p.W; x++ {
			if p.dots[y][x] == 0 {
				b.WriteRune(' ')
				continue
			}
			b.WriteString(paint(p.colors[y][x], string(rune(0x2800+int(p.dots[y][x])))))
		}
		out[y] = b.String()
	}
	return out
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// floats turns a nullable series into float64 with NaN gaps.
func floats(values []*float64) []float64 {
	out := make([]float64, len(values))
	for i, p := range values {
		if v, ok := f(p); ok {
			out[i] = v
		} else {
			out[i] = math.NaN()
		}
	}
	return out
}

func span(values ...[]float64) (float64, float64, bool) {
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, vs := range values {
		for _, v := range vs {
			if !math.IsNaN(v) {
				lo, hi = math.Min(lo, v), math.Max(hi, v)
			}
		}
	}
	return lo, hi, !math.IsInf(lo, 0)
}

// --- gantt ---------------------------------------------------------------------------------------

// GanttRow is one lane: segments [start, end) in epoch seconds with a colour each.
type GanttRow struct {
	Label string
	Color color.Color
	Segs  []GanttSeg
}

type GanttSeg struct {
	Start, End float64
	Color      color.Color
	Glyph      rune
}

// gantt renders lanes over [t0, t1] into w cells after a label column of lw cells; cursor >= 0
// highlights that column.
func gantt(rows []GanttRow, t0, t1 float64, lw, w, cursor int) []string {
	out := []string{}
	if t1 <= t0 {
		t1 = t0 + 1
	}
	rev := lipgloss.NewStyle().Reverse(true)
	for _, row := range rows {
		cells := make([]string, w)
		raw := make([]string, w)
		for _, s := range row.Segs {
			if s.End < t0 || s.Start > t1 {
				continue
			}
			a := int((math.Max(s.Start, t0) - t0) / (t1 - t0) * float64(w))
			b := int((math.Min(s.End, t1) - t0) / (t1 - t0) * float64(w))
			a, b = max(0, min(w-1, a)), max(0, min(w-1, b))
			g := s.Glyph
			if g == 0 {
				g = '█'
			}
			for i := a; i <= b; i++ {
				cells[i] = paint(s.Color, string(g))
				raw[i] = string(g)
			}
		}
		var b strings.Builder
		for i := range cells {
			switch {
			case i == cursor && raw[i] != "":
				b.WriteString(rev.Render(raw[i]))
			case i == cursor:
				b.WriteString(rev.Render("·"))
			case raw[i] != "":
				b.WriteString(cells[i])
			default:
				b.WriteString(paint(cFaint, "·"))
			}
		}
		out = append(out, fit(paint(row.Color, clean(row.Label)), lw)+" "+b.String())
	}
	return out
}

// axis is a time axis line for [t0, t1] over w cells (labels at both ends and the middle).
func axis(t0, t1 float64, lw, w int) string {
	left := time.Unix(int64(t0), 0).UTC().Format("15:04")
	mid := time.Unix(int64((t0+t1)/2), 0).UTC().Format("15:04")
	right := time.Unix(int64(t1), 0).UTC().Format("15:04")
	line := []rune(strings.Repeat("─", w))
	put := func(at int, s string) {
		for i, r := range s {
			if at+i >= 0 && at+i < len(line) {
				line[at+i] = r
			}
		}
	}
	put(0, left)
	put(w/2-len(mid)/2, mid)
	put(w-len(right), right)
	return strings.Repeat(" ", lw+1) + dim(string(line))
}

// stateGlyph and colour for experiment states.
func stateStyle(state string) (string, color.Color) {
	switch state {
	case "LIVE":
		return "◉", cGood
	case "STALLED":
		return "◐", cWarn
	case "ACTIVE":
		return "◎", cInfo
	case "INTERRUPTED":
		return "✖", cBad
	case "FINISHED":
		return "✔", cL2
	case "ENDED":
		return "◌", cDim
	case "STORED":
		return "▣", cVerify
	}
	return "?", cDim
}

func sevStyle(sev string) (string, color.Color) {
	switch sev {
	case "error":
		return "✖", cBad
	case "warn", "warning":
		return "▲", cWarn
	}
	return "●", cInfo
}

func tierGlyph(status string) string {
	switch status {
	case "pass", "passed":
		return paint(cGood, "●")
	case "fail", "failed", "mismatch":
		return paint(cBad, "✖")
	case "", "unavailable", "not_run", "skipped":
		return paint(cFaint, "·")
	case "running":
		return paint(cInfo, "◐")
	}
	return paint(cWarn, "○")
}
