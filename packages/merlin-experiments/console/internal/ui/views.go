package ui

import (
	"fmt"
	"image/color"
	"math"
	"sort"
	"strings"
	"time"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/ucb-bar/merlin/packages/merlin-experiments/console/internal/proto"
)

// --- timeline ----------------------------------------------------------------------------------
//
// The TIMELINE panel is the experiment's wall clock: one lane per agent (turns ▆, tool calls █,
// awaits ░), one lane per job kind (build, verify, L2, L3, queue), two concurrency lanes (busy agents
// and running simulations), round/stop markers on the axis, and the lead agent's wall-time
// attribution on top. With the panel focused, , and . (< and > for 10x) move a time cursor; the last
// line reports what was happening at that instant.

func (m *Model) pGantt(st *proto.State, w, h int, focused bool) (string, string, []string) {
	t0, t1 := m.window2(st)
	if win := st.Timeline.Window; win != nil && windows[m.window].seconds == 0 {
		t0, t1 = win.Start, win.End
	}
	lw := min(18, w/5)
	cw := max(10, w-lw-2) // list() prefixes one cell
	rows := []GanttRow{}
	sel := []Row{}
	agents := append([]proto.Agent{}, st.Agents.Agents...)
	sort.SliceStable(agents, func(i, j int) bool { return agents[i].Role == "lead" && agents[j].Role != "lead" })
	for _, a := range agents {
		segs := []GanttSeg{}
		for _, sp := range a.Spans {
			s, e, k, ok := spanTimes(sp)
			if !ok {
				continue
			}
			if math.IsNaN(e) {
				e = t1
			}
			c := kindColor[k]
			if c == nil {
				c = cAgent
			}
			g := '▆'
			if k == "tool" {
				g = '█'
			} else if k == "wait" {
				g = '░'
			}
			segs = append(segs, GanttSeg{Start: s, End: e, Color: c, Glyph: g})
		}
		rows = append(rows, GanttRow{Label: agentName(a), Color: cAgent, Segs: segs})
		sel = append(sel, Row{Kind: "agents", Key: a.ID})
	}
	lanes := map[string][]GanttSeg{}
	for _, sp := range st.Jobs.Spans {
		s, ok := f(sp.Start)
		if !ok {
			continue
		}
		e, ok := f(sp.End)
		if !ok {
			e = t1
		}
		c := kindColor[sp.Lane]
		if c == nil {
			c = cDim
		}
		g := '█'
		if sp.Status == "failed" || sp.Status == "fail" {
			c, g = cBad, '▓'
		}
		lanes[sp.Lane] = append(lanes[sp.Lane], GanttSeg{Start: s, End: e, Color: c, Glyph: g})
	}
	for _, lane := range []string{"setup", "build", "verify", "calibration", "L2", "L3", "queue"} {
		if segs, ok := lanes[lane]; ok {
			rows = append(rows, GanttRow{Label: "jobs " + lane, Color: kindColor[lane], Segs: segs})
			sel = append(sel, Row{Kind: "jobs", Key: lane})
		}
	}
	cursor := -1
	if focused && m.tcursor >= 0 {
		cursor = min(cw-1, m.tcursor)
	}
	lines := gantt(rows, t0, t1, lw, cw, cursor)
	if c := st.Timeline.Concurrency; c != nil && len(c.Agents) > 0 {
		lines = append(lines, fit(paint(cAgent, "∥ busy agents"), lw)+" "+density(c.Agents, c.T0, c.DT, t0, t1, cw, cAgent, cursor))
		sel = append(sel, Row{Kind: "local", Key: "timeline", Local: localDetail("timeline", st.Timeline)})
		lines = append(lines, fit(paint(cL3, "∥ simulations"), lw)+" "+density(c.Sim, c.T0, c.DT, t0, t1, cw, cL3, cursor))
		sel = append(sel, Row{Kind: "local", Key: "timeline", Local: localDetail("timeline", st.Timeline)})
	}
	header := []string{m.attribution(st, w), markerAxis(st.Timeline, t0, t1, lw, cw)}
	if len(rows) == 0 {
		lines = []string{dim("no agent or job spans recorded in this window")}
	}
	body := m.list("gantt", header, sel, lines, h-1, focused)
	if focused {
		body = append(body, m.cursorReport(st, t0, t1, cw))
	} else {
		body = append(body, dim(" , . move the time cursor when focused · Enter inspects a lane"))
	}
	legend := paint(cAgent, "▆") + dim("turn ") + paint(cBuild, "█") + dim("tool/build ") + paint(cQueue, "░") +
		dim("await ") + paint(cL2, "█") + dim("L2 ") + paint(cL3, "█") + dim("L3 ") + paint(cBad, "▓") + dim("failed ") +
		dim("· window "+windows[m.window].label)
	return "TIMELINE", legend, body
}

// attribution is the one-line answer to "where did the lead's wall time go".
func (m *Model) attribution(st *proto.State, w int) string {
	tl := st.Timeline
	if tl.Lead == nil || tl.Window == nil || tl.Window.Seconds <= 0 {
		if tl.Recorded != nil {
			return dim("lead ") + paint(cAgent, fmt.Sprintf("think %.0f%%", pctOf(tl.Recorded.ThinkS, tl.Recorded.ThinkS+tl.Recorded.ToolWaitS))) + dim(" (recorded)")
		}
		return dim("wall-time attribution not recorded")
	}
	l, total := tl.Lead, tl.Window.Seconds
	parts := []float64{l.SetupS, l.ThinkS, l.ToolS, l.WaitS, l.IdleS}
	colors := []color.Color{cCalib, cAgent, cBuild, cQueue, cFaint}
	bw := min(28, max(8, w/6))
	s := dim("lead ") + stackBar(parts, colors, total, bw) + " "
	if l.SetupS > 0 {
		s += paint(cCalib, fmt.Sprintf("setup %.0f%% ", pctOf(l.SetupS, total)))
	}
	s += paint(cAgent, fmt.Sprintf("think %.0f%%", pctOf(l.ThinkS, total))) + " " +
		paint(cBuild, fmt.Sprintf("tool %.0f%%", pctOf(l.ToolS, total))) + " " +
		paint(cQueue, fmt.Sprintf("await %.0f%%", pctOf(l.WaitS, total))) + " " +
		dim(fmt.Sprintf("idle %.0f%%", pctOf(l.IdleS, total)))
	if r := tl.Recorded; r != nil {
		s += dim(" · recorded ") + paint(cAgent, fmt.Sprintf("think %.0f%%", pctOf(r.ThinkS, r.ThinkS+r.ToolWaitS)))
	}
	p := tl.Parallelism
	if v, ok := f(p.AgentsAvg); ok {
		s += dim(" · agents ∥ ") + paint(cText, fmt.Sprintf("%.1f", v)) + dim(" avg/") + paint(cText, intOr(p.AgentsPeak, "—")) + dim(" peak")
	}
	if v, ok := f(tl.Overlap.LeadBlockedOnSimS); ok {
		s += dim(" · blocked on sim ") + paint(cText, dur(v))
	}
	if v := tl.Verdict; v != nil {
		s += dim(" → ") + bold(cWarn, strings.ReplaceAll(v.Bound, "_", " ")+" bound")
	}
	return s
}

func pctOf(v, total float64) float64 {
	if total <= 0 {
		return 0
	}
	return 100 * v / total
}

// density draws a per-bucket count series over [t0, t1] as shaded cells.
func density(vals []float64, s0, dt, t0, t1 float64, w int, c color.Color, cursor int) string {
	if dt <= 0 || t1 <= t0 {
		return strings.Repeat(" ", w)
	}
	cells := make([]float64, w)
	hits := make([]int, w)
	peak := 0.0
	for i, v := range vals {
		t := s0 + (float64(i)+0.5)*dt
		if t < t0 || t > t1 {
			continue
		}
		x := min(w-1, int((t-t0)/(t1-t0)*float64(w)))
		cells[x] += v
		hits[x]++
	}
	for i := range cells {
		if hits[i] > 0 {
			cells[i] /= float64(hits[i])
			peak = math.Max(peak, cells[i])
		}
	}
	var b strings.Builder
	for i, v := range cells {
		g := paint(cFaint, "·")
		if hits[i] > 0 && v > 0.05 {
			g = paint(c, string(sparks[min(len(sparks)-1, int(v/math.Max(peak, 1)*float64(len(sparks)-1)+0.5))]))
		}
		if i == cursor {
			g = lipgloss.NewStyle().Reverse(true).Render(ansi.Strip(g))
		}
		b.WriteString(g)
	}
	return b.String()
}

// markerAxis is the time axis with round and stop markers (▼) placed on it; a marker label is
// written after it where the axis has room (labels never overwrite the clock labels).
func markerAxis(tl proto.Timeline, t0, t1 float64, lw, w int) string {
	base := []rune(ansi.Strip(axis(t0, t1, lw, w)))[lw+1:]
	colors := map[int]color.Color{}
	for _, mk := range tl.Markers {
		if mk.T < t0 || mk.T > t1 || t1 <= t0 {
			continue
		}
		x := min(w-1, int((mk.T-t0)/(t1-t0)*float64(w)))
		// Never cover a clock label: move to the nearest free axis cell.
		for d := 0; d < w; d++ {
			if x-d >= 0 && base[x-d] == '─' {
				x -= d
				break
			}
			if x+d < w && base[x+d] == '─' {
				x += d
				break
			}
		}
		if base[x] != '─' && base[x] != '▼' {
			continue
		}
		base[x] = '▼'
		colors[x] = cWarn
		if mk.Kind == "stop" || mk.Kind == "finish" {
			colors[x] = cBad
		}
		label := []rune(" " + clean(mk.Label) + " ")
		free := x+len(label) < w
		for i := 1; free && i <= len(label); i++ {
			if base[x+i] != '─' {
				free = false
			}
		}
		if free {
			copy(base[x+1:], label)
			for i := range label {
				colors[x+1+i] = colors[x]
			}
		}
	}
	var b strings.Builder
	for i, r := range base {
		if c, ok := colors[i]; ok {
			b.WriteString(paint(c, string(r)))
		} else {
			b.WriteString(dim(string(r)))
		}
	}
	return strings.Repeat(" ", lw+1) + b.String()
}

// cursorReport says what every lane was doing at the time cursor.
func (m *Model) cursorReport(st *proto.State, t0, t1 float64, w int) string {
	if m.tcursor < 0 {
		return dim(" , . move the time cursor · < > jump 10 · Enter inspects a lane")
	}
	col := min(w-1, m.tcursor)
	at := t0 + (float64(col)+0.5)/float64(w)*(t1-t0)
	parts := []string{}
	for _, a := range st.Agents.Agents {
		for _, sp := range a.Spans {
			s, e, k, ok := spanTimes(sp)
			if ok && (math.IsNaN(e) || e >= at) && s <= at && k != "turn" {
				parts = append(parts, paint(cAgent, agentName(a))+dim(" "+k))
				break
			}
		}
	}
	running := map[string]int{}
	for _, sp := range st.Jobs.Spans {
		s, _ := f(sp.Start)
		e, ok := f(sp.End)
		if !ok {
			e = t1
		}
		if s <= at && e >= at {
			running[sp.Lane]++
		}
	}
	for _, lane := range sortedKeys(running) {
		parts = append(parts, paint(kindColor[lane], fmt.Sprintf("%d %s", running[lane], lane)))
	}
	event := ""
	best := math.Inf(1)
	for _, ev := range st.Events.Events {
		if t, ok := f(ev.T); ok && math.Abs(t-at) < best && math.Abs(t-at) < (t1-t0)/float64(w)*3 {
			best, event = math.Abs(t-at), clean(ev.Text)
		}
	}
	s := bold(cFocus, " ▏"+time.Unix(int64(at), 0).UTC().Format("01-02 15:04:05")) + " "
	if len(parts) == 0 {
		parts = append(parts, dim("no tool, await or job open"))
	}
	s += strings.Join(parts, dim(" · "))
	if event != "" {
		s += dim("  ⟶ ") + paint(cText, event)
	}
	return s
}

// --- cores --------------------------------------------------------------------------------------
//
// CORES is every CPU of the machine: a cell per core shaded by utilization and coloured by who holds
// it (this experiment, another run, a reservation with nobody on it, the general pool), the average
// per holder over time, the console's own cost, and per-core history rows when there is room (zoom).

var holderColors = []color.Color{cAgent, cVerify, cCalib, cBuild}

func (m *Model) holderColor(core proto.MachineCore, st *proto.State) color.Color {
	if core.Mine {
		return cL2
	}
	for _, hd := range core.Holders {
		if hd == "general" {
			continue
		}
		for i, a := range st.Machine.Allocations {
			if a.Name == hd && a.Kind == "run" {
				return holderColors[i%len(holderColors)]
			}
			if a.Name == hd && a.Kind == "reservation" {
				return cWarn
			}
		}
	}
	return cDim
}

func (m *Model) pCPU(st *proto.State, w, h int, focused bool) (string, string, []string) {
	mc := st.Machine
	r := st.Resources
	out := []string{}
	n := len(mc.Cores)
	kinds := map[int]proto.Core{}
	for _, c := range r.Cores {
		kinds[c.CPU] = c
	}
	if n == 0 {
		out = append(out, dim("machine CPUs not sampled yet"))
	} else {
		cell := max(1, min(3, w/n))
		var b strings.Builder
		ruler := []rune(strings.Repeat(" ", n*cell))
		step := 1 // label every step-th core, as densely as the labels fit (no machine size assumed)
		for step*cell < 4 {
			step *= 2
		}
		for i, c := range mc.Cores {
			if i%step == 0 {
				copy(ruler[i*cell:], []rune(fmt.Sprint(c.CPU)))
			}
			col := m.holderColor(c, st)
			g := paint(cFaint, strings.Repeat("?", cell))
			if u, ok := f(c.UtilPct); ok {
				g = paint(col, strings.Repeat(string(sparks[min(7, int(u/12.5))]), cell))
				if k, ok := kinds[c.CPU]; ok && c.Mine && k.Kind != "" && k.Kind != "idle" && u >= 5 && cell >= 2 {
					g = paint(col, strings.Repeat(string(sparks[min(7, int(u/12.5))]), cell-1)) + paint(kindColor[k.Kind], kindGlyph[k.Kind])
				}
			}
			b.WriteString(g)
		}
		out = append(out, b.String(), dim(string(ruler)))
		groups := map[string][]proto.MachineCore{}
		for _, c := range mc.Cores {
			key := "general"
			if c.Mine {
				key = "this experiment"
			} else {
				for _, hd := range c.Holders {
					if hd != "general" {
						key = hd
						break
					}
				}
			}
			groups[key] = append(groups[key], c)
		}
		names := sortedKeys(groups)
		sort.SliceStable(names, func(i, j int) bool { return names[i] == "this experiment" && names[j] != "this experiment" })
		for _, name := range names {
			cs := groups[name]
			sum, k := 0.0, 0
			for _, c := range cs {
				if u, ok := f(c.UtilPct); ok {
					sum, k = sum+u, k+1
				}
			}
			avg := "—"
			if k > 0 {
				avg = fmt.Sprintf("%3.0f%%", sum/float64(k))
			}
			col := m.holderColor(cs[0], st)
			hist := m.groupHist[name]
			line := paint(col, "■ ") + fit(paint(cText, tailFit(clean(name), 22)), 23) + dim(fmt.Sprintf("%2d cpu ", len(cs))) +
				paint(cText, avg) + " " + sparkline(hist, max(4, w-38), col)
			out = append(out, line)
		}
	}
	if c := st.Console; c.Policy != "" || c.CPUPct != nil {
		s := dim("console ") + paint(cText, pct(c.CPUPct, 1)) + dim(" of a core · on ") + paint(cText, ranges(c.CPUs))
		if c.Shared {
			s += paint(cWarn, " shares the experiment's CPUs (SCHED_IDLE)")
		} else if len(c.Avoided) > 0 {
			s += dim(" · avoids ") + paint(cText, ranges(c.Avoided))
		}
		if c.Throttled {
			s += paint(cWarn, " · throttled to "+durP(c.IntervalS))
		}
		out = append(out, s)
	}
	if s := r.Series["cpu_pct"]; len(s) > 1 && h-2-len(out) >= 2 {
		out = append(out, dim("run tree cpu ")+sparkline(s, max(4, w-24), cHost)+" "+paint(cText, pct(s[len(s)-1], 0)))
	}
	// Per-core history for the experiment's cores (all cores when zoomed), as much as fits.
	room := h - 2 - len(out)
	show := []proto.MachineCore{}
	for _, c := range mc.Cores {
		if c.Mine || (m.zoom && m.focus == "cpu") {
			show = append(show, c)
		}
	}
	if room >= 2 && len(show) > 0 {
		if len(show) > room {
			show = show[:room]
		}
		for _, c := range show {
			out = append(out, fmt.Sprintf("%3d ", c.CPU)+m.machineTimeline(c.CPU, max(8, w-16), m.holderColor(c, st))+" "+m.machineNow(c.CPU))
		}
	}
	legend := paint(cL2, "■") + dim("this ") + paint(cAgent, "■") + dim("other run ") + paint(cWarn, "■") + dim("reserved ") +
		paint(cDim, "■") + dim("general")
	m.rows["cpu"] = []Row{{Kind: "local", Local: append(localDetail("machine", mc), localDetail("resources", r)...)}}
	return "CORES", legend, out
}

func ranges(cpus []int) string {
	if len(cpus) == 0 {
		return "—"
	}
	sorted := append([]int{}, cpus...)
	sort.Ints(sorted)
	parts := []string{}
	start, prev := sorted[0], sorted[0]
	flush := func() {
		if start == prev {
			parts = append(parts, fmt.Sprint(start))
		} else {
			parts = append(parts, fmt.Sprintf("%d-%d", start, prev))
		}
	}
	for _, c := range sorted[1:] {
		if c == prev+1 {
			prev = c
			continue
		}
		flush()
		start, prev = c, c
	}
	flush()
	return strings.Join(parts, ",")
}

// recordMachine keeps a per-core and per-holder utilization history from the machine section.
func (m *Model) recordMachine() {
	mc := m.State.Machine
	at := m.State.GeneratedAt
	if len(mc.Cores) == 0 || at == m.machineAt {
		return
	}
	m.machineAt = at
	groups := map[string][]float64{}
	for _, c := range mc.Cores {
		u, ok := f(c.UtilPct)
		hist := append(m.machHist[c.CPU], coreSample{at: at, util: u, ok: ok})
		if len(hist) > 1800 {
			hist = hist[len(hist)-1800:]
		}
		m.machHist[c.CPU] = hist
		key := "general"
		if c.Mine {
			key = "this experiment"
		} else {
			for _, hd := range c.Holders {
				if hd != "general" {
					key = hd
					break
				}
			}
		}
		if ok {
			groups[key] = append(groups[key], u)
		}
	}
	for key, us := range groups {
		sum := 0.0
		for _, u := range us {
			sum += u
		}
		v := sum / float64(len(us))
		hist := append(m.groupHist[key], &v)
		if len(hist) > 600 {
			hist = hist[len(hist)-600:]
		}
		m.groupHist[key] = hist
	}
}

func (m *Model) machineTimeline(cpu, w int, c color.Color) string {
	hist := m.machHist[cpu]
	if len(hist) == 0 {
		return strings.Repeat(" ", w)
	}
	start := max(0, len(hist)-w)
	var b strings.Builder
	b.WriteString(strings.Repeat(" ", w-(len(hist)-start)))
	for _, s := range hist[start:] {
		if !s.ok {
			b.WriteString(paint(cFaint, "?"))
			continue
		}
		if s.util < 2 {
			b.WriteString(paint(cFaint, "·"))
			continue
		}
		b.WriteString(paint(c, string(sparks[min(7, int(s.util/12.5))])))
	}
	return b.String()
}

func (m *Model) machineNow(cpu int) string {
	hist := m.machHist[cpu]
	if len(hist) == 0 || !hist[len(hist)-1].ok {
		return dim("  —")
	}
	return paint(cText, fmt.Sprintf("%3.0f%%", hist[len(hist)-1].util))
}

// --- experiment browser and live jump ------------------------------------------------------------

var browseFilters = []struct {
	label  string
	states []string
}{
	{"all", nil},
	{"live", []string{"LIVE", "STALLED", "ACTIVE"}},
	{"finished", []string{"FINISHED"}},
	{"interrupted", []string{"INTERRUPTED", "ENDED"}},
	{"stored", []string{"STORED"}},
}

func (m *Model) liveExperiments() []proto.Experiment {
	out := []proto.Experiment{}
	for _, e := range m.Experiments.Items {
		if e.State == "LIVE" || e.State == "STALLED" || e.State == "ACTIVE" {
			out = append(out, e)
		}
	}
	return out
}

// jumpLive opens the next running experiment (cycling when several run).
func (m *Model) jumpLive() {
	live := m.liveExperiments()
	if len(live) == 0 {
		m.flash("no experiment is running")
		return
	}
	next := live[0]
	for i, e := range live {
		if e.ID == m.State.Experiment.ID {
			next = live[(i+1)%len(live)]
		}
	}
	if next.ID == m.State.Experiment.ID {
		m.flash("already on the only running experiment")
		return
	}
	m.send(map[string]any{"cmd": "select", "id": next.ID})
	m.flash("opening running " + next.RunID)
}

func (m *Model) browseItems() []proto.Experiment {
	out := []proto.Experiment{}
	want := browseFilters[m.browseState].states
	q := strings.ToLower(m.browseQuery)
	for _, e := range m.Experiments.Items {
		if want != nil && !contains(want, e.State) {
			continue
		}
		text := strings.ToLower(e.RunID + " " + e.TaskName + " " + e.State + " " + e.StateBasis + " " + e.Target)
		if q != "" && !strings.Contains(text, q) {
			continue
		}
		out = append(out, e)
	}
	return out
}

func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

func (m *Model) browseKey(s, text string) {
	items := m.browseItems()
	switch s {
	case "esc":
		if m.browseQuery != "" {
			m.browseQuery = ""
		} else {
			m.browse = false
		}
	case "enter":
		if m.browseCursor < len(items) {
			e := items[m.browseCursor]
			m.send(map[string]any{"cmd": "select", "id": e.ID})
			m.flash("opening " + e.RunID)
			m.browse = false
		}
	case "down", "ctrl+n":
		m.browseCursor++
	case "up", "ctrl+p":
		m.browseCursor--
	case "pgdown":
		m.browseCursor += 10
	case "pgup":
		m.browseCursor -= 10
	case "tab":
		m.browseState = (m.browseState + 1) % len(browseFilters)
		m.browseCursor = 0
	case "shift+tab":
		m.browseState = (m.browseState + len(browseFilters) - 1) % len(browseFilters)
		m.browseCursor = 0
	case "ctrl+l":
		m.browse = false
		m.jumpLive()
	case "backspace":
		if m.browseQuery != "" {
			m.browseQuery = m.browseQuery[:len(m.browseQuery)-1]
			m.browseCursor = 0
		}
	default:
		if text != "" {
			m.browseQuery += text
			m.browseCursor = 0
		}
	}
	m.browseCursor = max(0, min(len(m.browseItems())-1, m.browseCursor))
}

// overlayBrowse is the full experiment history: every run on the machine, filterable by state and
// text, newest activity first within state, with its task, start, duration and the deciding record.
func (m *Model) overlayBrowse(lines []string, w, h int) []string {
	bw, bh := w-4, h-3
	inner := bw - 3
	items := m.browseItems()
	tabs := []string{}
	for i, bf := range browseFilters {
		n := 0
		for _, e := range m.Experiments.Items {
			if bf.states == nil || contains(bf.states, e.State) {
				n++
			}
		}
		label := fmt.Sprintf(" %s %d ", bf.label, n)
		if i == m.browseState {
			tabs = append(tabs, lipgloss.NewStyle().Reverse(true).Bold(true).Render(label))
		} else {
			tabs = append(tabs, dim(label))
		}
	}
	body := []string{strings.Join(tabs, " ") + dim("   type to search · Tab state · ↑/↓ · Enter open · Ctrl-L running · Esc close")}
	body = append(body, bold(cWarn, " search: ")+paint(cText, m.browseQuery+"▏"))
	runW := max(24, inner*34/100)
	taskW := max(16, inner*20/100)
	head := fmt.Sprintf("  %-12s %s %s %-11s %9s %9s  %s", "state", fit("run", runW), fit("task", taskW), "started", "duration", "last rec", "decided by")
	body = append(body, dim(fit(head, inner)))
	room := bh - 2 - len(body)
	off := 0
	if m.browseCursor >= room {
		off = m.browseCursor - room + 1
	}
	now := m.State.GeneratedAt
	if now == 0 {
		now = float64(time.Now().Unix())
	}
	for i := off; i < len(items) && i < off+room; i++ {
		e := items[i]
		g, c := stateStyle(e.State)
		started, length := "—", "—"
		if v, ok := f(e.StartedAt); ok {
			started = time.Unix(int64(v), 0).UTC().Format("01-02 15:04")
			end := now
			if lr, ok := f(e.LastRecordAt); ok && e.State != "LIVE" && e.State != "STALLED" && e.State != "ACTIVE" {
				end = lr
			}
			length = dur(end - v)
		}
		mark := "  "
		if e.ID == m.State.Experiment.ID {
			mark = bold(cFocus, "● ")
		}
		row := mark + paint(c, fit(g+" "+e.State, 12)) + " " + paint(cText, fit(tailFit(clean(e.RunID), runW), runW)) + " " +
			dim(fit(tailFit(clean(e.TaskName), taskW), taskW)) + " " + paint(cText, fmt.Sprintf("%-11s", started)) +
			paint(cText, fmt.Sprintf(" %9s", length)) + dim(fmt.Sprintf(" %9s", ago(now, e.LastRecordAt))) + "  " + dim(clean(e.StateBasis))
		if i == m.browseCursor {
			row = bold(cFocus, "▶") + ansi.Cut(row, 1, ansi.StringWidth(row))
		}
		body = append(body, row)
	}
	if len(items) == 0 {
		body = append(body, dim("  no experiment matches"))
	}
	block := strings.Split(box("experiments", "EXPERIMENTS", fmt.Sprintf("%d of %d", len(items), len(m.Experiments.Items)), body, bw, bh, true), "\n")
	x, y := 2, 2
	for i, l := range block {
		if y+i < len(lines) {
			lines[y+i] = splice(lines[y+i], x, l, bw)
		}
	}
	return lines
}
