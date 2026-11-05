package ui

import (
	"fmt"
	"image/color"
	"math"
	"sort"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/ucb-bar/merlin/packages/merlin-experiments/console/internal/proto"
)

// --- what the run is doing now ---------------------------------------------------------------------
//
// The NOW line answers "what is happening" in one sentence per side: the author (its phase from the
// lifecycle records), the grader (what the run's cores are busy with), and the latest grade.

// Activity colours by category id (sources/activity.py CATEGORIES); anything else is "other".
var actColors = map[string]color.Color{
	"author": cAgent, "startup": lipgloss.Color("#eab308"), "validate": cVerify, "build": cBuild,
	"test": cCache, "L2": cL2, "L3": cL3, "calibration": lipgloss.Color("#2dd4bf"),
	"harness": cQueue, "other": lipgloss.Color("#a8a29e"), "idle": cFaint,
}

// Plain words for activity ids in legends.
var actWords = map[string]string{
	"author": "authoring", "startup": "author startup", "validate": "checks", "build": "build",
	"test": "test/verify", "L2": "L2 sim", "L3": "L3 sim", "calibration": "calibration",
	"harness": "harness", "other": "not this run", "idle": "idle",
}

func actColor(id string) color.Color {
	if c, ok := actColors[id]; ok {
		return c
	}
	return actColors["other"]
}

// codeIDs maps the section's one-character codes to category ids.
func codeIDs(a proto.Activity) map[byte]string {
	out := map[byte]string{}
	for _, c := range a.Categories {
		if len(c.Code) == 1 {
			out[c.Code[0]] = c.ID
		}
	}
	return out
}

func (m *Model) nowLine(st *proto.State, w int) string {
	if st.Experiment.ID == "" {
		return ""
	}
	l := st.Lifecycle
	now := st.GeneratedAt
	author := ""
	switch l.Phase {
	case "authoring":
		author = bold(cGood, "authoring") + dim(" for ") + paint(cText, ago2(now, l.PhaseSince))
	case "startup":
		author = bold(actColor("startup"), "starting a turn") + dim(" · ") + paint(cText, clean(l.PhaseDetail))
	case "ended":
		author = bold(cWarn, "stopped") + dim(" "+ago(now, l.PhaseSince)) + dim(" · ") + paint(cText, clip(clean(l.PhaseDetail), min(70, w/3)))
	default:
		if st.Agents.Totals.Working != nil && *st.Agents.Totals.Working > 0 {
			author = bold(cGood, fmt.Sprintf("%.0f agents working", *st.Agents.Totals.Working))
		} else {
			author = dim("no lifecycle records")
		}
	}
	if !m.live() {
		// A stored run: say how it ended rather than what its author is doing.
		end := clean(st.Experiment.StateBasis)
		if c := st.Progress.Completion; c != nil && c.Reason != "" {
			end = "completion: " + clean(c.Reason)
		}
		author = bold(cVerify, strings.ToLower(st.Experiment.State)) + dim(" · ") + paint(cText, clip(end, min(80, w/3)))
		if l.Phase != "" && l.Phase != "unknown" {
			author += dim(" · author ") + paint(cText, l.Phase)
		}
	}
	cores := ""
	if a := st.Activity; len(a.Now) > 0 {
		count := map[string]int{}
		ids := codeIDs(a)
		for _, c := range a.Now {
			if len(c.Code) == 1 {
				if id := ids[c.Code[0]]; id != "" && id != "idle" {
					count[id]++
				}
			}
		}
		parts := []string{}
		for _, cat := range a.Categories {
			if n := count[cat.ID]; n > 0 {
				parts = append(parts, paint(actColor(cat.ID), fmt.Sprintf("%s %d", actWords[cat.ID], n)))
			}
		}
		if len(parts) == 0 {
			cores = dim("all idle")
		} else {
			cores = strings.Join(parts, dim(" · "))
		}
		cores = dim(fmt.Sprintf("cores (%d) ", len(a.Now))) + cores
	}
	grade := gradeNow(st.Progress, now)
	parts := []string{bold(cHeading, " NOW ") + dim("author ") + author}
	if cores != "" {
		parts = append(parts, cores)
	}
	if grade != "" {
		parts = append(parts, grade)
	}
	return fit(strings.Join(parts, dim("  │  ")), w)
}

func ago2(now float64, p *float64) string {
	v, ok := f(p)
	if !ok || now <= 0 {
		return "—"
	}
	return dur(now - v)
}

// --- CORES: a Gantt of what every core of the run works on -----------------------------------------
//
// One row per core (or two cores per row, upper and lower half cell, when the panel is short), one
// cell per time slice; the colour is the slice's dominant activity and the height its utilization.
// The history starts when the console began watching the run. Enter on a row inspects that core.

type pairKey struct{ fg, bg color.Color }

var pairCache = map[pairKey][2]string{}

// pairCell paints s with a foreground and a background colour (cached like styled).
func pairCell(fgc, bgc color.Color, s string) string {
	k := pairKey{fgc, bgc}
	parts, ok := pairCache[k]
	if !ok {
		r := lipgloss.NewStyle().Foreground(fgc).Background(bgc).Render(styleProbe)
		i := strings.Index(r, styleProbe)
		if i < 0 {
			return s
		}
		parts = [2]string{r[:i], r[i+len(styleProbe):]}
		pairCache[k] = parts
	}
	return parts[0] + s + parts[1]
}

// holdBuckets is how many unobserved buckets a core's last observation is carried across. The
// sampler slows down under its CPU budget, so a gap of a few buckets is a slower sample, not an
// absence; a longer gap still renders empty.
const holdBuckets = 5

// coreCells resamples one core's buckets to w cells over [t0, t1]: the dominant code and the mean
// utilization digit per cell ('\x00' for no observation).
func coreCells(codes, util string, h0, dt, t0, t1 float64, w int) ([]byte, []int) {
	out, us := make([]byte, w), make([]int, w)
	if dt <= 0 || t1 <= t0 {
		return out, us
	}
	for i := 0; i < w; i++ {
		a := t0 + float64(i)/float64(w)*(t1-t0)
		b := t0 + float64(i+1)/float64(w)*(t1-t0)
		i0 := max(0, int(math.Floor((a-h0)/dt)))
		i1 := min(len(codes), int(math.Ceil((b-h0)/dt)))
		count := map[byte]int{}
		usum, un := 0, 0
		for j := i0; j < i1; j++ {
			c := codes[j]
			if c == ' ' {
				continue
			}
			weight := 1
			if j < len(util) && util[j] >= '0' && util[j] <= '9' {
				d := int(util[j] - '0')
				usum, un = usum+d, un+1
				weight += d
			}
			if c == '.' {
				weight = 1
			}
			count[c] += weight
		}
		if len(count) == 0 {
			// Nothing observed in this cell: carry the latest observation from the few buckets before it.
			for j := min(i0, len(codes)) - 1; j >= 0 && j >= i0-holdBuckets; j-- {
				if codes[j] != ' ' {
					count[codes[j]] = 1
					if j < len(util) && util[j] >= '0' && util[j] <= '9' {
						usum, un = int(util[j]-'0'), 1
					}
					break
				}
			}
		}
		best, bw := byte(0), 0
		for c, n := range count {
			if n > bw || (n == bw && c < best) {
				best, bw = c, n
			}
		}
		// A slice that was busy at all shows its busy activity rather than idle.
		if best == '.' {
			for c, n := range count {
				if c != '.' && n > 2 {
					best = c
				}
			}
		}
		out[i] = best
		if un > 0 {
			us[i] = usum / un
		}
	}
	return out, us
}

func (m *Model) pCores(st *proto.State, w, h int, focused bool) (string, string, []string) {
	a := st.Activity
	if a.History == nil || len(a.History.Cores) == 0 {
		if !m.live() {
			return m.pRecordedCPU(st, w, h)
		}
		title, hint, body := m.pCPU(st, w, h, focused)
		if len(a.Notes) > 0 {
			body = append([]string{dim(clean(a.Notes[0]))}, body...)
		}
		return title, hint, body
	}
	hist := a.History
	ids := codeIDs(a)
	t1 := st.GeneratedAt
	if !m.live() {
		t1 = hist.T0 + float64(hist.N)*hist.DT
	}
	t0 := hist.Since
	if win := m.win().seconds; win > 0 && t1-win > t0 {
		t0 = t1 - win
	}
	if t1-t0 < 60 {
		t0 = t1 - 60 // A young history fills the right edge first.
	}
	head := []string{}
	mc := st.Machine
	line := dim("run cpus ") + paint(cText, ranges(a.CPUs))
	if v, ok := f(mc.BusyCPUs); ok {
		line += dim(" · machine ") + paint(cText, fmt.Sprintf("%.1f/%.0f", v, float64(len(mc.Cores)))) + dim(" busy")
	}
	if c := st.Console; c.CPUPct != nil {
		line += dim(" · console ") + paint(cText, pct(c.CPUPct, 1)) + dim(" on ") + paint(cText, ranges(c.CPUs))
	}
	if io := st.Resources.Process.IO; io != nil {
		if v, ok := f(io.ReadBps); ok {
			line += dim(" · run reads ") + paint(cText, humanF(v)+"B/s")
		}
	}
	line += dim(" · since ") + paint(cText, clock(&hist.Since))
	nowW := 0
	if w >= 70 {
		nowW = min(30, w/4)
	}
	room := h - 2 - len(head) - 2 // borders, header, axis, legend
	paired := len(a.CPUs) > room && !m.zoom
	lw := 4
	if paired {
		lw = 6
	}
	cw := max(10, w-lw-nowW-3) // list() prefixes one cell
	now := map[int]string{}
	short := map[int]string{} // for two cores per row: busy % and a short label each
	for _, c := range a.Now {
		if len(c.Top) > 0 {
			t := c.Top[0]
			now[c.CPU] = paint(actColor(t.Activity), fmt.Sprintf("%3.0f%% ", t.Pct)) + dim(clean(t.Label))
			short[c.CPU] = paint(actColor(t.Activity), fmt.Sprintf("%3.0f%%", t.Pct)) + dim(" "+fit(clean(t.Label), max(4, nowW/2-6)))
		} else {
			short[c.CPU] = dim("  idle")
		}
	}
	type coreRow struct {
		cpu   int
		codes []byte
		util  []int
	}
	cores := []coreRow{}
	for _, c := range hist.Cores {
		codes, util := coreCells(c.Codes, c.Util, hist.T0, hist.DT, t0, t1, cw)
		cores = append(cores, coreRow{c.CPU, codes, util})
	}
	sort.Slice(cores, func(i, j int) bool { return cores[i].cpu < cores[j].cpu })
	rows, lines := []Row{}, []string{}
	idleCell := paint(cFaint, "·")
	if !paired {
		for _, c := range cores {
			var b strings.Builder
			for i := range c.codes {
				id := ids[c.codes[i]]
				switch {
				case c.codes[i] == 0:
					b.WriteByte(' ')
				case id == "idle" || id == "":
					b.WriteString(idleCell)
				default:
					b.WriteString(paint(actColor(id), string(sparks[min(7, max(1, c.util[i]*8/10))])))
				}
			}
			lines = append(lines, dim(fmt.Sprintf("%3d ", c.cpu))+b.String()+"  "+fit(now[c.cpu], nowW))
			rows = append(rows, Row{Kind: "activity", Key: fmt.Sprint(c.cpu)})
		}
	} else {
		// Two cores per row: the upper half cell is the first core, the lower half the second.
		for i := 0; i < len(cores); i += 2 {
			top := cores[i]
			var bot *coreRow
			if i+1 < len(cores) {
				bot = &cores[i+1]
			}
			var b strings.Builder
			for x := range top.codes {
				fc := colorOf(ids, top.codes[x])
				bc := cFaint
				if bot != nil {
					bc = colorOf(ids, bot.codes[x])
				}
				b.WriteString(pairCell(fc, bc, "▀"))
			}
			label := fmt.Sprintf("%5d ", top.cpu)
			right := now[top.cpu]
			if bot != nil {
				label = fmt.Sprintf("%2d/%-2d ", top.cpu, bot.cpu)
				right = fit(short[top.cpu], nowW/2) + fit(short[bot.cpu], nowW-nowW/2)
			}
			lines = append(lines, dim(label)+b.String()+"  "+fit(right, nowW))
			rows = append(rows, Row{Kind: "activity", Key: fmt.Sprint(top.cpu)})
		}
	}
	// Share of busy core time per activity over the window, the legend and the answer in one line.
	share := map[string]float64{}
	total := 0.0
	for _, c := range cores {
		for i, code := range c.codes {
			if code == 0 {
				continue
			}
			id := ids[code]
			weight := float64(c.util[i]) + 0.5
			if id == "idle" {
				weight = 0
			}
			share[id] += weight
			total += weight
		}
	}
	legend := []string{}
	for _, cat := range a.Categories {
		if cat.ID == "idle" {
			continue
		}
		if v := share[cat.ID]; total > 0 && v > 0 {
			legend = append(legend, paint(actColor(cat.ID), "█")+paint(cText, fmt.Sprintf("%s %.0f%%", actWords[cat.ID], 100*v/total)))
		}
	}
	if len(legend) == 0 {
		legend = append(legend, dim("no busy time observed yet"))
	}
	body := m.list("cpu", head, rows, lines, h-2, focused)
	axisLine := axis(t0, t1, lw-1, cw)
	body = append(body, axisLine, strings.Join(legend, " ")+dim("  │  ")+line)
	hint := "height = busy · ↵ core detail"
	if paired {
		hint = "2 cores per row: upper half, lower half · z one per row · ↵ core detail"
	}
	return "CORES · what each CPU of the run works on", hint, body
}

func colorOf(ids map[byte]string, code byte) color.Color {
	if code == 0 {
		return cFaint
	}
	id := ids[code]
	if id == "" || id == "idle" {
		return lipgloss.Color("#1f2937")
	}
	return actColor(id)
}

// --- the author lane of the TIMELINE ----------------------------------------------------------------

// authorRow is one lane with every author turn: its startup (▒), the agent working (█) and the end
// (✕ when the host interrupted it).
func authorRow(l proto.Lifecycle, t1 float64) GanttRow {
	segs := []GanttSeg{}
	for _, t := range l.Turns {
		if s, ok := f(t.StartupStart); ok {
			e, ok := f(t.StartupEnd)
			if !ok {
				e = t1
			}
			segs = append(segs, GanttSeg{Start: s, End: e, Color: actColor("startup"), Glyph: '▒'})
		}
		if s, ok := f(t.AgentStart); ok {
			e, ok := f(t.AgentEnd)
			if !ok {
				e = t1
			}
			segs = append(segs, GanttSeg{Start: s, End: e, Color: cAgent, Glyph: '█'})
		}
		if e, ok := f(t.AgentEnd); ok && t.EndReason != "" && t.EndReason != "next turn started" {
			segs = append(segs, GanttSeg{Start: e, End: e, Color: cBad, Glyph: '✕'})
		}
	}
	return GanttRow{Label: "author turns", Color: cAgent, Segs: segs}
}

// pRecordedCPU is CORES for a run that is not running: per-core activity is only observed live, so
// it shows what the run itself recorded (its process tree's CPU and memory, sampled every few seconds).
func (m *Model) pRecordedCPU(st *proto.State, w, h int) (string, string, []string) {
	r := st.Resources
	out := []string{dim("Per-core activity is observed only while a console watches a live run. Below: the run's own")}
	out = append(out, dim("process-tree samples (rounds/*.resource_samples.jsonl)."))
	ts := floats(r.Series["t"])
	cpu := floats(r.Series["cpu_pct"])
	if len(ts) < 2 {
		return "CORES · recorded use", "not running", append(out, dim("no resource samples recorded"))
	}
	lo, hi, _ := span(ts)
	_, peak, _ := span(cpu)
	plotH := max(3, h-2-len(out)-3)
	pl := NewPlot(w-10, plotH)
	pl.Series(ts, cpu, lo, hi, 0, math.Max(peak, 1), cL2)
	for i, l := range pl.Lines() {
		lab := "          "
		if i == 0 {
			lab = fit(dim(fmt.Sprintf("%.1f cores", peak/100)), 10)
		}
		out = append(out, lab+l)
	}
	out = append(out, axis(lo, hi, 9, w-10))
	rss := floats(r.Series["rss_bytes"])
	_, rssPeak, _ := span(rss)
	out = append(out, dim("process tree CPU over the run · peak ")+paint(cText, fmt.Sprintf("%.1f cores", peak/100))+
		dim(" · memory peak ")+paint(cText, humanF(rssPeak)+"B"))
	return "CORES · recorded use", "the run's process tree (not per core)", out
}

// clip shortens s to at most n cells (with …) and never pads it.
func clip(s string, n int) string {
	if ansi.StringWidth(s) <= n {
		return s
	}
	return fit(s, n)
}

// gradeNow is the grade the NOW line names: the newest one grading, else the newest that graded
// something, else the copy last published to the agent.
func gradeNow(p proto.Progress, now float64) string {
	var pick *proto.Grade
	for i := range p.Grades {
		if p.Grades[i].State == "grading" {
			pick = &p.Grades[i]
			break
		}
	}
	for i := range p.Grades {
		if n, ok := f(p.Grades[i].NCompleted); pick == nil && ok && n > 0 {
			pick = &p.Grades[i]
		}
	}
	if pick == nil {
		if pub := p.Published; pub != nil {
			if v, ok := f(pub.NSelected); ok {
				g := dim("grade ") + paint(cGood, intOr(pub.NPassed, "—")) + dim(fmt.Sprintf("/%.0f pass", v))
				if t, ok := f(pub.UpdatedAt); ok && now > 0 {
					g += dim(" (" + dur(now-t) + " ago)")
				}
				return g
			}
		}
		return ""
	}
	word := "last grade "
	if pick.State == "grading" {
		word = "grading "
	}
	g := dim(word) + paint(cText, intOr(pick.NCompleted, "—")) + dim("/"+intOr(pick.NSelected, "—")+" ") +
		paint(cGood, intOr(pick.NPassed, "0")+"✔")
	if n, ok := f(pick.NL2); ok && n > 0 {
		g += " " + paint(cInfo, fmt.Sprintf("%.0f L2✔", n))
	}
	g += " " + paint(cBad, intOr(pick.NFailed, "0")+"✖")
	if t, ok := f(pick.UpdatedAt); ok && now > 0 {
		g += dim(" · last result " + dur(max(0, now-t)) + " ago")
	}
	return g
}
