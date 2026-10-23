// Package ui is the Bubble Tea command console: one view of every panel, drill-down inspection,
// zoom, filter and experiment switching. It renders what the data process sent and nothing else.
package ui

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/ucb-bar/merlin/packages/merlin-experiments/console/internal/proto"
)

// Messages delivered by the collector bridge.
type (
	// LineMsg carries one decoded data-process message.
	LineMsg struct{ Msg any }
	// CollectorExitMsg reports that the data process ended.
	CollectorExitMsg struct {
		Err    string
		Stderr []string
	}
	tickMsg time.Time
)

// Sender writes one command to the data process.
type Sender func(cmd map[string]any)

type rect struct{ x, y, w, h int }

func (r rect) contains(x, y int) bool { return x >= r.x && x < r.x+r.w && y >= r.y && y < r.y+r.h }

// Panel order for focus cycling.
var panelOrder = []string{"experiments", "progress", "stages", "alerts", "gantt", "agents", "tokens", "cpu",
	"perf", "capsules", "passes", "events"}

var windows = []struct {
	label   string
	seconds float64
}{{"15m", 900}, {"1h", 3600}, {"6h", 21600}, {"all", 0}}

// Row is a selectable line in a panel: Enter inspects it.
type Row struct {
	Kind, Key string
	Local     []string // shown when the source has no deeper record
}

// Model is the console state.
type Model struct {
	W, H        int
	State       *proto.State
	Experiments proto.Experiments
	Hello       proto.Hello
	send        Sender

	focus        string
	zoom         bool
	sidebar      int // -1 auto, 0 hidden, 1 shown
	cursor       map[string]int
	scroll       map[string]int
	rows         map[string][]Row
	filter       string
	filtering    bool
	sortMode     map[string]int
	window       int
	paused       bool
	frozen       *proto.State
	help         bool
	inspect      *proto.Inspect
	inspectLoc   []string
	inspectOpen  bool
	inspectOff   int
	nextReq      int
	pendingReq   int
	status       string
	statusAt     time.Time
	frame        int
	lastMsg      time.Time
	collector    string
	stderr       []string
	rects        map[string]rect
	coreHist     map[int][]coreSample
	coreOrder    []int
	machHist     map[int][]coreSample
	groupHist    map[string][]*float64
	machineAt    float64
	tcursor      int
	browse       bool
	browseQuery  string
	browseState  int
	browseCursor int
	now          func() time.Time
}

type coreSample struct {
	at   float64
	util float64
	ok   bool
	kind string
}

// New builds the console model; send may be nil (replay and render-once modes).
func New(send Sender) *Model {
	if send == nil {
		send = func(map[string]any) {}
	}
	return &Model{W: 200, H: 50, State: proto.NewState(), send: send, focus: "progress", sidebar: -1,
		cursor: map[string]int{}, scroll: map[string]int{}, rows: map[string][]Row{}, sortMode: map[string]int{},
		rects: map[string]rect{}, coreHist: map[int][]coreSample{}, window: 3, now: time.Now,
		machHist: map[int][]coreSample{}, groupHist: map[string][]*float64{}, tcursor: -1}
}

func (m *Model) Init() tea.Cmd { return tick(m.live()) }

// tick drives the clock and the live pulse only; data arrives as messages. One redraw a second keeps
// the console's own CPU use negligible next to the experiment it watches.
func tick(bool) tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m *Model) live() bool {
	s := m.State.Experiment.State
	return s == "LIVE" || s == "ACTIVE" || s == "STALLED"
}

func (m *Model) shown() *proto.State {
	if m.paused && m.frozen != nil {
		return m.frozen
	}
	return m.State
}

func (m *Model) flash(s string) {
	m.status = s
	m.statusAt = m.now()
}

// Apply handles one data-process message (exported for render-once and tests).
func (m *Model) Apply(msg any) {
	m.lastMsg = m.now()
	switch v := msg.(type) {
	case *proto.Hello:
		m.Hello = *v
	case *proto.Experiments:
		m.Experiments = *v
	case *proto.Snapshot:
		if v.Experiment.ID != m.State.Experiment.ID {
			m.coreHist = map[int][]coreSample{}
			m.cursor, m.scroll = map[string]int{}, map[string]int{}
			m.inspectOpen = false
			m.groupHist = map[string][]*float64{}
			m.tcursor = -1
		}
		m.State.Merge(v)
		m.recordCores()
		m.recordMachine()
	case *proto.Inspect:
		if v.RequestID == m.pendingReq {
			m.inspect = v
		}
	case *proto.Error:
		m.flash("data process: " + clean(v.Message))
	}
}

func (m *Model) recordCores() {
	r := m.State.Resources
	if r.Mode != "live" {
		return
	}
	now := m.State.GeneratedAt
	order := []int{}
	for _, c := range r.Cores {
		u, ok := f(c.UtilPct)
		hist := append(m.coreHist[c.CPU], coreSample{at: now, util: u, ok: ok, kind: c.Kind})
		if len(hist) > 3600 {
			hist = hist[len(hist)-3600:]
		}
		m.coreHist[c.CPU] = hist
		order = append(order, c.CPU)
	}
	m.coreOrder = order
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch v := msg.(type) {
	case tea.WindowSizeMsg:
		m.W, m.H = v.Width, v.Height
	case LineMsg:
		m.Apply(v.Msg)
	case CollectorExitMsg:
		m.collector = "data process exited"
		if v.Err != "" {
			m.collector += ": " + clean(v.Err)
		}
		m.stderr = v.Stderr
	case tickMsg:
		m.frame++
		return m, tick(m.live())
	case tea.KeyPressMsg:
		return m.key(v)
	case tea.MouseClickMsg:
		m.click(v.X, v.Y)
	case tea.MouseWheelMsg:
		for id, r := range m.rects {
			if r.contains(v.X, v.Y) {
				if v.Button == tea.MouseWheelUp {
					m.move(id, -3)
				} else if v.Button == tea.MouseWheelDown {
					m.move(id, 3)
				}
			}
		}
	}
	return m, nil
}

func (m *Model) panels() []string {
	out := []string{}
	for _, id := range panelOrder {
		if id == "experiments" && !m.sidebarShown() {
			continue
		}
		out = append(out, id)
	}
	return out
}

func (m *Model) sidebarShown() bool {
	if m.sidebar == -1 {
		return m.W >= 170
	}
	return m.sidebar == 1
}

func (m *Model) cycle(step int) {
	ps := m.panels()
	idx := 0
	for i, id := range ps {
		if id == m.focus {
			idx = i
		}
	}
	m.focus = ps[(idx+step+len(ps))%len(ps)]
}

func (m *Model) move(id string, step int) {
	n := len(m.rows[id])
	if n == 0 {
		m.scroll[id] = max(0, m.scroll[id]+step)
		return
	}
	m.cursor[id] = max(0, min(n-1, m.cursor[id]+step))
}

func (m *Model) key(k tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	s := k.String()
	if m.filtering {
		switch s {
		case "enter", "esc":
			m.filtering = false
		case "backspace":
			if m.filter != "" {
				m.filter = m.filter[:len(m.filter)-1]
			}
		default:
			if k.Text != "" {
				m.filter += k.Text
			}
		}
		m.cursor[m.focus] = 0
		return m, nil
	}
	if m.browse {
		if s == "ctrl+c" {
			m.send(map[string]any{"cmd": "quit"})
			return m, tea.Quit
		}
		m.browseKey(s, k.Text)
		return m, nil
	}
	if m.focus == "gantt" {
		step := map[string]int{",": -1, ".": 1, "<": -10, ">": 10}[s]
		if step != 0 {
			if m.tcursor < 0 {
				m.tcursor = m.W / 3
			}
			m.tcursor = max(0, m.tcursor+step)
			return m, nil
		}
	}
	if m.inspectOpen {
		switch s {
		case "esc", "q", "enter", "i":
			m.inspectOpen = false
			return m, nil
		case "j", "down":
			m.inspectOff++
			return m, nil
		case "k", "up":
			m.inspectOff = max(0, m.inspectOff-1)
			return m, nil
		case "pgdown", "space":
			m.inspectOff += 10
			return m, nil
		case "pgup":
			m.inspectOff = max(0, m.inspectOff-10)
			return m, nil
		case "y":
			return m, m.copyPath()
		}
	}
	switch s {
	case "q", "ctrl+c":
		m.send(map[string]any{"cmd": "quit"})
		return m, tea.Quit
	case "tab", "right", "l":
		m.cycle(1)
	case "shift+tab", "left", "h":
		m.cycle(-1)
	case "j", "down":
		m.move(m.focus, 1)
	case "k", "up":
		m.move(m.focus, -1)
	case "pgdown", "ctrl+d":
		m.move(m.focus, 10)
	case "pgup", "ctrl+u":
		m.move(m.focus, -10)
	case "g", "home":
		m.cursor[m.focus], m.scroll[m.focus] = 0, 0
	case "G", "end":
		m.cursor[m.focus] = max(0, len(m.rows[m.focus])-1)
	case "z":
		m.zoom = !m.zoom
	case "e":
		if m.sidebarShown() {
			m.sidebar = 0
			if m.focus == "experiments" {
				m.focus = "progress"
			}
		} else {
			m.sidebar, m.focus = 1, "experiments"
		}
	case "L":
		m.jumpLive()
	case "b", "o":
		m.browse, m.browseCursor, m.browseQuery = true, 0, ""
		for i, e := range m.browseItems() {
			if e.ID == m.State.Experiment.ID {
				m.browseCursor = i
			}
		}
	case "/":
		m.filtering, m.filter = true, ""
	case "esc":
		m.filter, m.zoom, m.help, m.tcursor = "", false, false, -1
	case "s":
		m.sortMode[m.focus]++
	case "t":
		m.window = (m.window + 1) % len(windows)
		m.flash("time window " + windows[m.window].label)
	case "p", "space":
		m.paused = !m.paused
		if m.paused {
			cp := *m.State
			m.frozen = &cp
			m.flash("display paused; collection continues")
		} else {
			m.frozen = nil
		}
	case "r":
		m.send(map[string]any{"cmd": "refresh"})
		m.flash("refresh requested")
	case "?":
		m.help = !m.help
	case "y":
		return m, m.copyPath()
	case "enter", "i":
		m.activate()
	case "1", "2", "3", "4", "5", "6", "7", "8", "9":
		ps := m.panels()
		if n := int(s[0] - '1'); n < len(ps) {
			m.focus = ps[n]
		}
	}
	return m, nil
}

func (m *Model) selected(id string) (Row, bool) {
	rows := m.rows[id]
	c := m.cursor[id]
	if c < 0 || c >= len(rows) {
		return Row{}, false
	}
	return rows[c], true
}

// activate: Enter on the experiments list switches experiment; elsewhere it opens the inspector.
func (m *Model) activate() {
	row, ok := m.selected(m.focus)
	if !ok {
		return
	}
	if row.Kind == "experiment" {
		m.send(map[string]any{"cmd": "select", "id": row.Key})
		m.flash("opening " + row.Key)
		return
	}
	m.inspectOpen, m.inspectOff = true, 0
	m.inspectLoc = row.Local
	m.inspect = nil
	if row.Kind == "local" {
		m.inspect = &proto.Inspect{Title: m.focus + " · " + firstNonEmpty(row.Key, "detail")}
	}
	if row.Kind != "" && row.Kind != "local" {
		m.nextReq++
		m.pendingReq = m.nextReq
		m.send(map[string]any{"cmd": "inspect", "request_id": m.nextReq, "kind": row.Kind, "key": row.Key})
	}
}

func (m *Model) copyPath() tea.Cmd {
	var path string
	if m.inspect != nil && len(m.inspect.Paths) > 0 {
		path = m.inspect.Paths[0]
	} else if m.State.Experiment.RunDir != "" {
		path = m.State.Experiment.RunDir
	}
	if path == "" {
		return nil
	}
	m.flash("copied " + path)
	return tea.SetClipboard(path)
}

func (m *Model) click(x, y int) {
	for id, r := range m.rects {
		if !r.contains(x, y) {
			continue
		}
		if id == "inspect" {
			return
		}
		m.focus = id
		row := y - r.y - 1 - m.headerRows(id)
		if row >= 0 && len(m.rows[id]) > 0 {
			m.cursor[id] = max(0, min(len(m.rows[id])-1, m.scroll[id]+row))
		}
		return
	}
}

// headerRows is how many body lines precede a panel's selectable rows (set while rendering).
func (m *Model) headerRows(id string) int { return headerRows[id] }

var headerRows = map[string]int{}

// View renders the whole console.
func (m *Model) View() tea.View {
	v := tea.NewView(m.Render())
	v.AltScreen = true
	v.MouseMode = tea.MouseModeCellMotion
	v.WindowTitle = "merlin console · " + m.State.Experiment.RunID
	return v
}

// Render is the frame as a string of exactly H lines (used by View and render-once).
func (m *Model) Render() string {
	w, h := max(60, m.W), max(16, m.H)
	st := m.shown()
	m.rects = map[string]rect{}
	lines := m.header(st, w)
	body := h - len(lines) - 1
	grid := m.grid(st, w, body, len(lines))
	lines = append(lines, grid...)
	lines = append(lines, m.footer(w))
	if m.inspectOpen {
		lines = m.overlayInspect(lines, w, h)
	}
	if m.browse {
		lines = m.overlayBrowse(lines, w, h)
	}
	if m.help {
		lines = m.overlayHelp(lines, w, h)
	}
	for i := range lines {
		lines[i] = fit(lines[i], w)
	}
	return strings.Join(lines[:min(len(lines), h)], "\n")
}

func (m *Model) grid(st *proto.State, w, h, top int) []string {
	canvas := make([]string, h)
	for i := range canvas {
		canvas[i] = strings.Repeat(" ", w)
	}
	place := func(id string, x, y, pw, ph int) {
		if pw < 4 || ph < 3 {
			return
		}
		m.rects[id] = rect{x, y + top, pw, ph}
		block := strings.Split(m.panel(st, id, pw, ph), "\n")
		for i, line := range block {
			if y+i < len(canvas) {
				canvas[y+i] = splice(canvas[y+i], x, line, pw)
			}
		}
	}
	x0 := 0
	if m.sidebarShown() && !m.zoom {
		sw := 30
		if w >= 220 {
			sw = 36
		}
		place("experiments", 0, 0, sw, h)
		x0 = sw
	}
	mw := w - x0
	if m.zoom {
		place(m.focus, x0, 0, mw, h)
		return canvas
	}
	r1 := max(7, h*20/100)
	r2 := max(9, h*29/100)
	r3 := max(7, h*23/100)
	r4 := max(6, h-r1-r2-r3)
	if r1+r2+r3+r4 > h {
		r4 = max(0, h-r1-r2-r3)
	}
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
	cols(0, r1, []string{"progress", "stages", "alerts"}, []int{36, 38, 26})
	cols(r1, r2, []string{"gantt", "agents"}, []int{62, 38})
	cols(r1+r2, r3, []string{"tokens", "cpu", "perf"}, []int{33, 27, 40})
	cols(r1+r2+r3, r4, []string{"capsules", "passes", "events"}, []int{50, 22, 28})
	return canvas
}

// splice replaces cells [x, x+w) of an ANSI line with block.
func splice(line string, x int, block string, w int) string {
	left := ansi.Cut(line, 0, x)
	right := ansi.Cut(line, x+w, ansi.StringWidth(line))
	return fit(left, x) + fit(block, w) + right
}

func (m *Model) header(st *proto.State, w int) []string {
	e := st.Experiment
	glyph, c := stateStyle(e.State)
	if m.live() && !m.paused {
		pulse := []string{"◉", "◎", "○", "◎"}[m.frame%4]
		glyph = pulse
	}
	mode := "PUBLIC"
	if m.Hello.OperatorPrivate || st.Operator {
		mode = "OPERATOR"
	}
	run := e.RunID
	if run == "" {
		run = "waiting for the data process…"
	}
	now := st.GeneratedAt
	elapsed := "—"
	if v, ok := f(e.StartedAt); ok && now > 0 {
		end := now
		if lr, ok := f(e.LastRecordAt); ok && !m.live() {
			end = lr
		}
		elapsed = dur(end - v)
	}
	top := bold(c, " "+glyph+" "+e.State) + "  " + bold(cHeading, clean(run)) +
		dim("  task ") + paint(cText, clean(e.TaskName)) + dim("  elapsed ") + paint(cText, elapsed) +
		dim("  last record ") + paint(cText, ago(now, e.LastRecordAt))
	if !m.live() && e.State != "" {
		top += "  " + bold(cVerify, "REPLAY · records until "+clock(e.LastRecordAt))
	}
	right := bold(cWarn, mode) + dim("  "+time.Now().UTC().Format("15:04:05")+" UTC ")
	top = fit(top, w-ansi.StringWidth(right)) + right
	basis := dim(" basis: ") + paint(cText, clean(e.StateBasis))
	if e.RunDir != "" {
		basis += dim("   run ") + dim(clean(e.RunDir))
	}
	// Right side: a running experiment one key away, and what the console itself costs.
	side := ""
	if live := m.liveExperiments(); len(live) > 0 && !m.live() {
		side += bold(cGood, fmt.Sprintf("◉ %d running: ", len(live))) + paint(cText, strings.TrimRight(tailFit(clean(live[0].RunID), 32), " ")) + bold(cWarn, " [L]") + "  "
	} else if len(live) > 1 {
		side += bold(cGood, fmt.Sprintf("◉ %d running", len(live))) + bold(cWarn, " [L] next") + "  "
	}
	if c := st.Console; c.CPUPct != nil || c.Policy != "" {
		side += dim("console ") + paint(cText, pct(c.CPUPct, 1)) + dim(" cpu on ") + paint(cText, ranges(c.CPUs)) + " "
	}
	if side != "" {
		basis = fit(basis, w-ansi.StringWidth(side)) + side
	}
	return []string{top, basis}
}

func (m *Model) footer(w int) string {
	keys := " ⇥ focus  ↵ inspect  z zoom  b browse runs  L running  / filter  s sort  t window  , . time cursor  p pause  y copy  ? help  q quit"
	status := ""
	if m.filtering {
		status = bold(cWarn, " filter: "+m.filter+"▏")
	} else if m.filter != "" {
		status = paint(cWarn, " filter: "+m.filter)
	}
	if m.status != "" && m.now().Sub(m.statusAt) < 4*time.Second {
		status += paint(cInfo, "  "+m.status)
	}
	if m.collector != "" {
		tail := ""
		if len(m.stderr) > 0 {
			tail = ": " + m.stderr[len(m.stderr)-1]
		}
		status += paint(cBad, "  "+m.collector+clean(tail))
	}
	if m.paused {
		status += bold(cWarn, "  PAUSED")
	}
	win := dim("  window ") + paint(cText, windows[m.window].label)
	return fit(dim(keys)+win+status, w)
}

func (m *Model) overlayInspect(lines []string, w, h int) []string {
	dw := max(50, w*48/100)
	x := w - dw
	top, bh := 2, h-3
	body := []string{}
	title := "inspect"
	if m.inspect != nil {
		title = clean(m.inspect.Title)
		if m.inspect.Error != "" {
			body = append(body, paint(cBad, clean(m.inspect.Error)))
		}
		if m.inspect.Notice != "" {
			body = append(body, dim(clean(m.inspect.Notice)))
		}
		for _, sec := range m.inspect.Sections {
			body = append(body, bold(cHeading, clean(sec.Title)))
			for _, l := range sec.Lines {
				body = append(body, "  "+paint(cText, clean(l)))
			}
			body = append(body, "")
		}
		if len(m.inspect.Paths) > 0 {
			body = append(body, bold(cHeading, "records"))
			for _, p := range m.inspect.Paths {
				body = append(body, "  "+dim(clean(p)))
			}
		}
	} else if m.pendingReq != 0 && m.inspectLoc == nil {
		body = append(body, dim("reading records…"))
	}
	if len(m.inspectLoc) > 0 && (m.inspect == nil || len(m.inspect.Sections) == 0) {
		body = append(body, m.inspectLoc...)
	}
	off := min(m.inspectOff, max(0, len(body)-1))
	body = body[off:]
	block := strings.Split(box("inspect", title, "j/k scroll · y copy path · esc close", body, dw, bh, true), "\n")
	m.rects["inspect"] = rect{x, top, dw, bh}
	for i, l := range block {
		if top+i < len(lines) {
			lines[top+i] = splice(lines[top+i], x, l, dw)
		}
	}
	return lines
}

func (m *Model) overlayHelp(lines []string, w, h int) []string {
	help := []string{
		bold(cHeading, "Navigation"),
		"  Tab / Shift-Tab, ←/→, 1-9   focus a panel      mouse click   focus + select",
		"  j/k ↑/↓  PgUp/PgDn  g/G      move in a panel    wheel         scroll",
		"  Enter / i                    inspect the selected row (capsule, agent, pass, job, stage, event)",
		"  z                            zoom the focused panel (Esc to return)",
		"  e                            show/hide the experiments list; Enter there opens an experiment",
		"  b / o                        browse every experiment (type to search, Tab: all/live/finished/...)",
		"  L                            jump to the running experiment (again: the next running one)",
		"  , . (< >)                    TIMELINE focused: move the time cursor 1 (10) cells; Esc clears it",
		bold(cHeading, "Views"),
		"  /  filter rows in the focused panel     s  cycle sort     t  time window (15m, 1h, 6h, all)",
		"  p / Space  pause the display (collection continues)     r  force a refresh",
		"  y  copy the inspected record path (or the run directory) to the clipboard",
		bold(cHeading, "Reading the console"),
		"  ◉ LIVE  ◐ STALLED  ◎ ACTIVE (no managed owner)  ✖ INTERRUPTED  ✔ FINISHED  ◌ ENDED  ▣ STORED",
		"  Every state names the record that decided it. Missing values are shown as —, never as 0.",
		"  Cycles keep their source: spike model (estimate), gSIM (RTL simulation), L2 (functional model).",
		"  CPU activity is sampled telemetry, not evidence. Diagnostics compare recorded numbers only.",
		"  CORES: every CPU, coloured by holder (this run, other runs, idle reservation, general pool).",
		"  The console runs on CPUs no experiment holds, at nice 10 (SCHED_IDLE when it must share).",
	}
	bw, bh := min(w-4, 110), min(h-4, len(help)+2)
	block := strings.Split(box("inspect", "help", "? or esc to close", help, bw, bh, true), "\n")
	x, y := (w-bw)/2, (h-bh)/2
	for i, l := range block {
		if y+i < len(lines) {
			lines[y+i] = splice(lines[y+i], x, l, bw)
		}
	}
	return lines
}

// matches applies the text filter to a row label (case-insensitive substring).
func (m *Model) matches(text string) bool {
	if m.filter == "" {
		return true
	}
	return strings.Contains(strings.ToLower(text), strings.ToLower(m.filter))
}

// visible returns the slice of body lines for a scrolled, cursor-highlighted list.
func (m *Model) list(id string, header []string, rows []Row, lines []string, h int, focused bool) []string {
	m.rows[id] = rows
	headerRows[id] = len(header)
	room := max(1, h-2-len(header))
	c := m.cursor[id]
	if c >= len(rows) {
		c = max(0, len(rows)-1)
		m.cursor[id] = c
	}
	off := m.scroll[id]
	if c < off {
		off = c
	}
	if c >= off+room {
		off = c - room + 1
	}
	off = max(0, min(off, max(0, len(lines)-room)))
	m.scroll[id] = off
	out := append([]string{}, header...)
	for i := off; i < len(lines) && i < off+room; i++ {
		line := lines[i]
		if focused && i == c && len(rows) > 0 {
			line = bold(cFocus, "▶") + line
		} else {
			line = " " + line
		}
		out = append(out, line)
	}
	if len(lines) > room && len(out) > 0 {
		// The last visible row names the scroll position.
		more := dim(fmt.Sprintf(" ↕ %d-%d of %d", off+1, min(len(lines), off+room), len(lines)))
		out[len(out)-1] = more
	}
	return out
}

// localDetail renders any value as indented JSON lines for the inspector.
func localDetail(title string, v any) []string {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return []string{title}
	}
	out := []string{bold(cHeading, title)}
	for _, l := range strings.Split(string(raw), "\n") {
		out = append(out, "  "+paint(cText, clean(l)))
	}
	return out
}

func sortedKeys[V any](mp map[string]V) []string {
	keys := make([]string, 0, len(mp))
	for k := range mp {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
