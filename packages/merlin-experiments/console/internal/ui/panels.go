package ui

import (
	"fmt"
	"image/color"
	"math"
	"sort"
	"strings"

	"github.com/ucb-bar/merlin/packages/merlin-experiments/console/internal/proto"
)

// panel renders one panel of exactly w x h.
func (m *Model) panel(st *proto.State, id string, w, h int) string {
	focused := m.focus == id
	inner := w - 2
	var title, hint string
	var body []string
	switch id {
	case "experiments":
		title, hint, body = m.pExperiments(inner, h, focused)
	case "progress":
		title, hint, body = m.pProgress(st, inner, h, focused)
	case "stages":
		title, hint, body = m.pStages(st, inner, h, focused)
	case "alerts":
		title, hint, body = m.pAlerts(st, inner, h, focused)
	case "gantt":
		title, hint, body = m.pGantt(st, inner, h, focused)
	case "agents":
		title, hint, body = m.pAgents(st, inner, h, focused)
	case "tokens":
		title, hint, body = m.pTokens(st, inner, h, focused)
	case "cpu":
		title, hint, body = m.pCPU(st, inner, h, focused)
	case "perf":
		title, hint, body = m.pPerf(st, inner, h, focused)
	case "capsules":
		title, hint, body = m.pCapsules(st, inner, h, focused)
	case "passes":
		title, hint, body = m.pPasses(st, inner, h, focused)
	case "events":
		title, hint, body = m.pEvents(st, inner, h, focused)
	}
	if e := sectionError(st, id); e != "" {
		body = append([]string{paint(cBad, "reader error: "+clean(e))}, body...)
	}
	return box(id, title, hint, body, w, h, focused)
}

func sectionError(st *proto.State, id string) string {
	src := map[string]string{"gantt": "timeline", "cpu": "machine", "alerts": ""}[id]
	if src == "" {
		src = id
	}
	if e := st.DecodeErr[src]; e != "" {
		return "decode: " + e
	}
	switch src {
	case "resources":
		return st.Resources.Error
	case "agents":
		return st.Agents.Error
	case "tokens":
		return st.Tokens.Error
	case "jobs":
		return st.Jobs.Error
	case "events":
		return st.Events.Error
	case "capsules":
		return st.Capsules.Error
	case "progress":
		return st.Progress.Error
	case "perf":
		return st.Perf.Error
	case "passes":
		return st.Passes.Error
	case "stages":
		return st.Stages.Error
	case "timeline":
		return firstNonEmpty(st.Timeline.Error, st.Jobs.Error)
	case "machine":
		return firstNonEmpty(st.Machine.Error, st.Resources.Error)
	}
	return ""
}

func notes(ns []string, w int) []string {
	out := []string{}
	for _, n := range ns {
		out = append(out, dim("· "+clean(n)))
	}
	return out
}

// --- experiments ---------------------------------------------------------------------------------

func (m *Model) pExperiments(w, h int, focused bool) (string, string, []string) {
	items := m.Experiments.Items
	rows, lines := []Row{}, []string{}
	counts := map[string]int{}
	now := m.State.GeneratedAt
	lastState := ""
	for _, e := range items {
		counts[e.State]++
		if !m.matches(e.RunID + " " + e.TaskName + " " + e.State) {
			continue
		}
		if e.State != lastState {
			g, c := stateStyle(e.State)
			lines = append(lines, bold(c, g+" "+e.State))
			rows = append(rows, Row{Kind: "header"})
			lastState = e.State
		}
		g, c := stateStyle(e.State)
		mark := " "
		if e.ID == m.State.Experiment.ID {
			mark = bold(cFocus, "●")
		}
		age := ""
		if v, ok := f(e.LastRecordAt); ok && now > 0 {
			age = dur(now - v)
		}
		lines = append(lines, mark+paint(c, g)+" "+paint(cText, tailFit(clean(e.RunID), w-12))+dim(fmt.Sprintf("%8s", age)))
		lines = append(lines, "    "+dim(clean(e.TaskName)))
		rows = append(rows, Row{Kind: "experiment", Key: e.ID}, Row{Kind: "experiment", Key: e.ID})
	}
	header := []string{dim(fmt.Sprintf("%d found · %d live", len(items), counts["LIVE"]+counts["ACTIVE"]+counts["STALLED"]))}
	return "EXPERIMENTS", "↵ open", m.list("experiments", header, rows, lines, h, focused)
}

// --- progress -----------------------------------------------------------------------------------

func (m *Model) pProgress(st *proto.State, w, h int, focused bool) (string, string, []string) {
	p := st.Progress
	out := []string{}
	cur := ""
	if p.SnapshotCurrent != nil && !*p.SnapshotCurrent {
		cur = paint(cWarn, " (retained snapshot)")
	}
	out = append(out, bold(cHeading, "latest grade ")+paint(cText, intOr(p.NCompleted, "—")+"/"+intOr(p.NSelected, "—"))+
		dim(" checked")+bold(cHeading, "  passed ")+paint(cGood, intOr(p.NPassed, "—"))+dim("  state ")+paint(cText, clean(p.State))+cur)
	out = append(out, dim("numerical executions ")+paint(cText, intOr(p.Numeric.Completed, "—"))+
		dim("  pass ")+paint(cGood, intOr(p.Numeric.Passed, "—"))+dim("  fail ")+paint(cBad, intOr(p.Numeric.Failed, "—")))
	labelW := 12
	barW := max(6, w-labelW-26)
	for _, c := range p.Cohorts {
		total, ok := f(c.Total)
		if !ok {
			continue
		}
		all, _ := f(c.All)
		l2, _ := f(c.L2)
		l3, _ := f(c.L3)
		rest := math.Max(0, l2-all)
		segs := stackBar([]float64{all, rest}, []color.Color{cGood, cL2}, total, barW)
		name := clean(c.Name)
		if name == "total" {
			name = "TOTAL"
		}
		out = append(out, fit(paint(cText, name), labelW)+segs+
			fmt.Sprintf(" %s %s %s", paint(cGood, intOr(c.All, "—")), paint(cL2, "L2 "+intOr(c.L2, "—")), paint(cL3, "L3 "+intOr(c.L3, "—")))+
			dim(fmt.Sprintf("/%.0f", total)))
		_ = l3
	}
	if len(p.Cohorts) == 0 {
		out = append(out, dim("cohort coverage not recorded"))
	}
	if s := p.Series["n_passed"]; len(s) > 1 {
		last := human(s[len(s)-1])
		out = append(out, dim("passing over time ")+sparkline(s, max(4, w-26), cGood)+" "+paint(cText, last))
	}
	if c := p.Completion; c != nil {
		out = append(out, bold(cL2, "completed ")+paint(cText, clean(c.Reason))+dim(" after ")+paint(cText, durP(c.ElapsedS)))
	} else if p.Plateau != nil {
		if r, ok := p.Plateau["reason"].(string); ok {
			out = append(out, dim("plateau: "+clean(r)))
		}
	}
	if r := p.LatestRequest; r != nil {
		out = append(out, dim("latest check ")+paint(cText, clean(r.Source))+dim(" queue ")+paint(cText, durP(r.QueueWaitS))+
			dim(" wall ")+paint(cText, durP(r.WallS))+dim(" stage ")+paint(cText, clean(r.Stage)))
	}
	out = append(out, notes(p.Notes, w)...)
	legend := paint(cGood, "■") + dim(" all tiers ") + paint(cL2, "■") + dim(" L2 only")
	m.rows["progress"] = []Row{{Kind: "local", Local: localDetail("progress", p)}}
	return "PROGRESS", legend, out
}

// --- stages -------------------------------------------------------------------------------------

func stageStyle(state string) (string, color.Color) {
	switch state {
	case "done":
		return "✔", cGood
	case "active":
		return "◐", cInfo
	case "blocked":
		return "✖", cBad
	case "pending":
		return "○", cDim
	}
	return "?", cFaint
}

func (m *Model) pStages(st *proto.State, w, h int, focused bool) (string, string, []string) {
	stages := st.Stages.Stages
	chips := []string{}
	for _, s := range stages {
		g, c := stageStyle(s.State)
		name := clean(s.ID)
		chips = append(chips, paint(c, g+" "+name))
	}
	header := []string{}
	if len(chips) > 0 {
		pipeline := strings.Join(chips, dim(" → "))
		header = append(header, pipeline)
	}
	header = append(header, dim(fmt.Sprintf("%-18s %-7s %9s %11s %8s %7s", "stage", "state", "actions", "caps pass", "wall", "tokens")))
	rows, lines := []Row{}, []string{}
	for _, s := range stages {
		g, c := stageStyle(s.State)
		actions := "—"
		if t, ok := f(s.Actions.Total); ok && t > 0 {
			actions = fmt.Sprintf("%s/%s", intOr(s.Actions.Done, "0"), intOr(s.Actions.Total, "0"))
			if fl, ok := f(s.Actions.Failed); ok && fl > 0 {
				actions += paint(cBad, fmt.Sprintf(" ✖%.0f", fl))
			}
		}
		caps := "—"
		if r, ok := f(s.Capsules.Required); ok && r > 0 {
			caps = fmt.Sprintf("%s/%s", intOr(s.Capsules.Passing, "?"), intOr(s.Capsules.Required, "?"))
		}
		basis := ""
		if s.Basis == "ledger" {
			basis = paint(cGood, " L")
		}
		lines = append(lines, fit(paint(c, g+" "+clean(firstNonEmpty(s.Title, s.ID))), 19)+
			fit(paint(c, clean(s.State)), 8)+fmt.Sprintf("%9s %11s %8s %7s", actions, caps, durP(s.WallS), human(s.Tokens))+basis)
		rows = append(rows, Row{Kind: "local", Key: s.ID, Local: localDetail("stage "+s.ID, s)})
	}
	if len(stages) == 0 {
		lines = append(lines, dim("stage records not available"))
	}
	lines = append(lines, notes(st.Stages.Notes, w)...)
	return "STAGES", "derived unless L(edger)", m.list("stages", header, rows, lines, h, focused)
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if x != "" {
			return x
		}
	}
	return ""
}

// --- alerts / bottlenecks ----------------------------------------------------------------------

func (m *Model) pAlerts(st *proto.State, w, h int, focused bool) (string, string, []string) {
	rows, lines := []Row{}, []string{}
	for _, d := range st.Diagnostics {
		if !m.matches(d.Title + " " + d.Detail + " " + d.Code) {
			continue
		}
		g, c := sevStyle(d.Severity)
		times := ""
		if d.Occurrences > 1 {
			times = dim(fmt.Sprintf(" ×%d", d.Occurrences))
		}
		lines = append(lines, paint(c, g)+" "+paint(cText, clean(d.Title))+times)
		rows = append(rows, Row{Kind: "local", Key: d.Code, Local: localDetail(d.Code, d)})
		if d.Detail != "" {
			lines = append(lines, "   "+dim(clean(d.Detail)))
			rows = append(rows, Row{Kind: "local", Key: d.Code, Local: localDetail(d.Code, d)})
		}
	}
	if len(lines) == 0 {
		lines = append(lines, paint(cGood, "no findings from recorded numbers"))
	}
	header := []string{}
	if v := st.Timeline.Verdict; v != nil {
		header = append(header, bold(cWarn, "▶ "+strings.ToUpper(strings.ReplaceAll(v.Bound, "_", " "))+" BOUND ")+
			paint(cText, fmt.Sprintf("%.0f%%", v.SharePct))+dim(" · "+clean(v.Explanation)))
	}
	hint := fmt.Sprintf("%d findings", len(st.Diagnostics))
	return "BOTTLENECKS", hint, m.list("alerts", header, rows, lines, h, focused)
}

// --- gantt ----------------------------------------------------------------------------------------

func spanTimes(sp []any) (float64, float64, string, bool) {
	if len(sp) < 2 {
		return 0, 0, "", false
	}
	a, ok1 := sp[0].(float64)
	b, ok2 := sp[1].(float64)
	if !ok1 {
		return 0, 0, "", false
	}
	if !ok2 {
		b = math.NaN()
	}
	k := "turn"
	if len(sp) > 2 {
		if s, ok := sp[2].(string); ok {
			k = s
		}
	}
	return a, b, k, true
}

func (m *Model) window2(st *proto.State) (float64, float64) {
	t1 := st.GeneratedAt
	if !m.live() {
		if v, ok := f(st.Experiment.LastRecordAt); ok {
			t1 = v
		}
	}
	if win := windows[m.window].seconds; win > 0 {
		return t1 - win, t1
	}
	t0 := t1
	if v, ok := f(st.Experiment.StartedAt); ok {
		t0 = v
	}
	for _, a := range st.Agents.Agents {
		for _, sp := range a.Spans {
			if s, _, _, ok := spanTimes(sp); ok {
				t0 = math.Min(t0, s)
			}
		}
	}
	if t1-t0 < 60 {
		t0 = t1 - 60
	}
	return t0, t1
}

// --- agents ---------------------------------------------------------------------------------------

// agentName prefers the nickname; a lead without one is "lead" rather than its thread id.
func agentName(a proto.Agent) string {
	if a.Role == "lead" && (a.Name == "" || a.Name == a.ID) {
		return "lead"
	}
	return firstNonEmpty(a.Name, a.ID)
}

func agentStyle(state string) (string, color.Color) {
	switch state {
	case "working":
		return "◐", cGood
	case "completed", "finished":
		return "✔", cL2
	case "aborted":
		return "✖", cBad
	case "stopped":
		return "■", cWarn
	}
	return "○", cDim
}

func (m *Model) pAgents(st *proto.State, w, h int, focused bool) (string, string, []string) {
	a := st.Agents
	byParent := map[string][]proto.Agent{}
	ids := map[string]bool{}
	for _, ag := range a.Agents {
		ids[ag.ID] = true
	}
	roots := []proto.Agent{}
	for _, ag := range a.Agents {
		if ag.ParentID != "" && ids[ag.ParentID] && ag.ParentID != ag.ID {
			byParent[ag.ParentID] = append(byParent[ag.ParentID], ag)
		} else {
			roots = append(roots, ag)
		}
	}
	rows, lines := []Row{}, []string{}
	var walk func(ag proto.Agent, prefix string, last bool, depth int)
	walk = func(ag proto.Agent, prefix string, last bool, depth int) {
		if depth > 12 {
			return
		}
		branch := "├─"
		if last {
			branch = "└─"
		}
		if depth == 0 {
			branch = ""
		}
		g, c := agentStyle(ag.State)
		name := agentName(ag)
		path := ""
		if ag.Path != "" {
			path = dim(" " + clean(ag.Path))
		}
		tok := human(ag.Tokens.Total)
		head := dim(prefix+branch) + paint(c, g) + " " + bold(cAgent, clean(name)) + path
		right := dim(fmt.Sprintf(" %6s tok %7s", tok, durP(ag.ActiveS)))
		if fc, ok := f(ag.FailedCommands); ok && fc > 0 {
			right += paint(cBad, fmt.Sprintf(" ✖%.0f", fc))
		}
		if m.matches(name + " " + ag.Path + " " + ag.State) {
			lines = append(lines, fit(head, max(10, w-24))+right)
			rows = append(rows, Row{Kind: "agents", Key: ag.ID})
			if ag.Current != "" && h > 10 {
				lines = append(lines, dim(prefix+"   ↳ ")+paint(cText, clean(ag.Current)))
				rows = append(rows, Row{Kind: "agents", Key: ag.ID})
			}
		}
		kids := byParent[ag.ID]
		next := prefix
		if depth > 0 {
			if last {
				next += "   "
			} else {
				next += "│  "
			}
		}
		for i, kid := range kids {
			walk(kid, next, i == len(kids)-1, depth+1)
		}
	}
	for _, r := range roots {
		walk(r, "", true, 0)
	}
	if len(a.Agents) == 0 {
		lines = append(lines, dim("no agent records"))
	}
	t := a.Totals
	hint := fmt.Sprintf("%s agents · %s working · helpers ≤%s", intOr(t.Agents, "—"), intOr(t.Working, "—"), intOr(t.HelperCeiling, "?"))
	return "AGENT TREE", hint, m.list("agents", nil, rows, lines, h, focused)
}

// --- tokens ----------------------------------------------------------------------------------------

func (m *Model) pTokens(st *proto.State, w, h int, focused bool) (string, string, []string) {
	t := st.Tokens
	out := []string{}
	tot := t.Total
	out = append(out, paint(cFresh, "fresh ")+paint(cText, human(tot.FreshInput))+paint(cCache, "  cache ")+paint(cText, human(tot.CacheRead))+
		paint(cOutput, "  out ")+paint(cText, human(tot.Output))+paint(cReason, "  reason ")+paint(cText, human(tot.Reasoning))+
		dim("  total ")+bold(cHeading, human(tot.Total)))
	line2 := dim("cache hit ") + paint(cText, pctRatio(t.CacheHitRatio)) + dim("  rate/min in ") + paint(cText, human(t.RatePerMin.Input)) +
		dim(" out ") + paint(cText, human(t.RatePerMin.Output))
	out = append(out, line2)
	if b := t.Budget; b != nil {
		used, _ := f(b.UsedPct)
		out = append(out, dim("budget ")+bar(used/100, max(6, w-30), budgetColor(used))+" "+paint(cText, pct(b.UsedPct, 1))+dim(" of ")+paint(cText, human(b.Total)))
	}
	if r := t.RateLimit; r != nil {
		used, _ := f(r.UsedPercent)
		out = append(out, dim("rate limit ")+bar(used/100, max(6, w-30), budgetColor(used))+" "+paint(cText, pct(r.UsedPercent, 0)))
	}
	chartH := h - 2 - len(out) - 1
	ts := floats(t.Series["t"])
	if chartH >= 3 && len(ts) > 1 {
		half := w / 2
		left, right := NewPlot(half-1, chartH), NewPlot(w-half-1, chartH)
		lo, hi, _ := span(ts)
		in := [][]float64{floats(t.Series["fresh_input"]), floats(t.Series["cache_read"])}
		ob := [][]float64{floats(t.Series["output"]), floats(t.Series["reasoning"])}
		_, ihi, _ := span(in...)
		_, ohi, _ := span(ob...)
		left.Series(ts, in[1], lo, hi, 0, ihi, cCache)
		left.Series(ts, in[0], lo, hi, 0, ihi, cFresh)
		right.Series(ts, ob[0], lo, hi, 0, ohi, cOutput)
		right.Series(ts, ob[1], lo, hi, 0, ohi, cReason)
		ll, rl := left.Lines(), right.Lines()
		for i := range ll {
			out = append(out, ll[i]+dim("│")+rl[i])
		}
		out = append(out, fit(dim(" input (cumulative) ↑"+humanF(ihi)), half)+dim(" output ↑"+humanF(ohi)))
	} else if len(ts) <= 1 {
		out = append(out, dim("token series not recorded yet"))
	}
	rows := []Row{{Kind: "local", Local: localDetail("tokens", t)}}
	m.rows["tokens"] = rows
	return "TOKENS", "cumulative", out
}

func pctRatio(p *float64) string {
	v, ok := f(p)
	if !ok {
		return "—"
	}
	return fmt.Sprintf("%.1f%%", v*100)
}

func budgetColor(used float64) color.Color {
	switch {
	case used >= 90:
		return cBad
	case used >= 70:
		return cWarn
	}
	return cGood
}

// --- cpu / resources -----------------------------------------------------------------------------

var shades = []string{"·", "░", "▒", "▓", "█"}

func heatCell(c proto.Core) string {
	u, ok := f(c.UtilPct)
	if !ok {
		return paint(cFaint, "?")
	}
	i := min(4, int(u/20))
	col := kindColor[c.Kind]
	if col == nil || c.Kind == "idle" {
		col = cFaint
	}
	return paint(col, shades[i])
}

func (m *Model) coreTimeline(cpu, w int) string {
	hist := m.coreHist[cpu]
	if len(hist) == 0 {
		return strings.Repeat(" ", w)
	}
	end := hist[len(hist)-1].at
	span := 120.0
	cells := make([]string, w)
	for i := range cells {
		cells[i] = " "
	}
	for _, s := range hist {
		if s.at < end-span {
			continue
		}
		i := min(w-1, max(0, int((s.at-(end-span))/span*float64(w))))
		if !s.ok {
			cells[i] = paint(cFaint, "?")
			continue
		}
		k := s.kind
		if s.util < 2 {
			k = "idle"
		}
		g := kindGlyph[k]
		if g == "" {
			g = "?"
		}
		cells[i] = paint(kindColor[k], g)
	}
	return strings.Join(cells, "")
}

func (m *Model) coreNow(cpu int) string {
	hist := m.coreHist[cpu]
	if len(hist) == 0 || !hist[len(hist)-1].ok {
		return dim("  —")
	}
	return paint(cText, fmt.Sprintf("%3.0f%%", hist[len(hist)-1].util))
}

// --- performance -----------------------------------------------------------------------------------

var perfSorts = []string{"efficiency", "cycles", "divergence", "name"}

func (m *Model) pPerf(st *proto.State, w, h int, focused bool) (string, string, []string) {
	p := st.Perf
	rows := append([]proto.PerfRow{}, p.Rows...)
	mode := perfSorts[m.sortMode["perf"]%len(perfSorts)]
	val := func(r proto.PerfRow) float64 {
		switch mode {
		case "efficiency":
			if v, ok := f(r.EfficiencyPct); ok {
				return v
			}
			return math.Inf(1)
		case "cycles":
			if v, ok := f(r.GsimCycles); ok {
				return -v
			}
			if v, ok := f(r.SpikeModelCycles); ok {
				return -v
			}
			return math.Inf(1)
		case "divergence":
			if v, ok := f(r.DivergencePct); ok {
				return -math.Abs(v)
			}
			return math.Inf(1)
		}
		return 0
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if mode == "name" {
			return rows[i].Capsule < rows[j].Capsule
		}
		return val(rows[i]) < val(rows[j])
	})
	out := []string{}
	s := p.Summary
	out = append(out, dim("modeled ")+paint(cText, intOr(s.Modeled, "—"))+dim(" provisional ")+paint(cText, intOr(s.Provisional, "—"))+
		dim(" unknown ")+paint(cWarn, intOr(s.Unknown, "—"))+dim("  eff best ")+paint(cGood, pct(s.BestEff, 2))+dim(" worst ")+
		paint(cBad, pct(s.WorstEff, 2))+dim(" geo ")+paint(cText, pct(s.GeoEff, 2)))
	// Roofline scatter: log10 arithmetic intensity vs log10 achieved MACs/cycle.
	plotH := 0
	if h >= 16 {
		plotH = max(3, (h-6)/3)
	}
	if plotH > 0 {
		xs, ys := []float64{}, []float64{}
		for _, r := range rows {
			x, ok1 := f(r.Intensity)
			y, ok2 := f(r.AchievedMacsPerCycle)
			if ok1 && ok2 && x > 0 && y > 0 {
				xs, ys = append(xs, math.Log10(x)), append(ys, math.Log10(y))
			}
		}
		if len(xs) > 0 {
			pl := NewPlot(w-8, plotH)
			xlo, xhi, _ := span(xs)
			ylo, yhi, _ := span(ys)
			if pk, ok := f(p.PeakMacsPerCycle); ok && pk > 0 {
				yhi = math.Max(yhi, math.Log10(pk))
				py := math.Log10(pk)
				pl.Series([]float64{xlo - 0.5, xhi + 0.5}, []float64{py, py}, xlo-0.5, xhi+0.5, ylo-0.5, yhi+0.2, cDim)
			}
			for i := range xs {
				pl.Series([]float64{xs[i]}, []float64{ys[i]}, xlo-0.5, xhi+0.5, ylo-0.5, yhi+0.2, cFresh)
			}
			for i, l := range pl.Lines() {
				lab := "      "
				if i == 0 {
					lab = dim("peak ")
				}
				out = append(out, fit(lab, 6)+l)
			}
			out = append(out, dim("      MAC/byte → (log) · achieved MAC/cycle ↑ (log) · dashed = structural peak"))
		}
	}
	// Fixed numeric columns (49 cells); capsule name and busiest resource share the rest.
	rest := max(16, w-50)
	nameW := max(12, rest*60/100)
	busyW := max(4, rest-nameW)
	header := append(out, dim(fit("capsule", nameW)+fmt.Sprintf(" %9s %9s %8s %7s %5s ", "model", "gSIM", "floor", "eff", "Δ%")+fit("busiest", busyW)))
	sel, lines := []Row{}, []string{}
	for _, r := range rows {
		if !m.matches(r.Capsule + " " + r.Cohort) {
			continue
		}
		div := "  —"
		if v, ok := f(r.DivergencePct); ok && r.Comparable {
			div = fmt.Sprintf("%+4.0f", v)
		}
		top := ""
		if r.TopResource != nil {
			top = fmt.Sprintf("%s %s", shortResource(clean(r.TopResource.Name)), pct(r.TopResource.ActivityPct, 0))
		}
		effC := cText
		if v, ok := f(r.EfficiencyPct); ok {
			effC = effColor(v)
		}
		lines = append(lines, fit(paint(cText, clean(r.Capsule)), nameW)+fmt.Sprintf(" %9s %9s %8s ", human(r.SpikeModelCycles), human(r.GsimCycles), human(r.FloorCycles))+
			paint(effC, fmt.Sprintf("%7s", pct(r.EfficiencyPct, 2)))+fmt.Sprintf(" %5s ", div)+dim(fit(top, busyW)))
		sel = append(sel, Row{Kind: "capsules", Key: r.Capsule, Local: localDetail("performance "+r.Capsule, r)})
	}
	if len(rows) == 0 {
		lines = append(lines, dim("no performance records bound to a source"))
	}
	if gs := st.Golden.Comparisons; len(gs) > 0 {
		for _, g := range gs {
			txt := "vs golden " + clean(g.Subject) + ": "
			if g.Comparable {
				txt += fmt.Sprintf("%s / %s = %s", human(g.CandidateCycles), human(g.ReferenceCycles), ratio(g.Ratio))
			} else {
				txt += "not comparable (" + clean(g.Reason) + ")"
			}
			lines = append(lines, paint(cCalib, "★ ")+dim(txt))
			sel = append(sel, Row{Kind: "local", Local: localDetail("golden", g)})
		}
	}
	lines = append(lines, notes(p.Notes, w)...)
	return "PERFORMANCE", "sort " + mode + " · model=spike estimate, gSIM=RTL sim", m.list("perf", header, sel, lines, h, focused)
}

// shortResource keeps the last dotted component of a modeled resource name ("accelerator.storage.accumulator" -> "accumulator").
func shortResource(name string) string {
	if i := strings.LastIndex(name, "."); i >= 0 && i < len(name)-1 {
		return name[i+1:]
	}
	return name
}

func ratio(p *float64) string {
	v, ok := f(p)
	if !ok {
		return "—"
	}
	return fmt.Sprintf("%.2f×", v)
}

func effColor(v float64) color.Color {
	switch {
	case v >= 50:
		return cGood
	case v >= 10:
		return cWarn
	}
	return cBad
}

// --- capsules --------------------------------------------------------------------------------------

var capsuleSorts = []string{"status", "name", "cohort", "op", "cycles"}

func (m *Model) pCapsules(st *proto.State, w, h int, focused bool) (string, string, []string) {
	caps := append([]proto.Capsule{}, st.Capsules.Capsules...)
	mode := capsuleSorts[m.sortMode["capsules"]%len(capsuleSorts)]
	rank := map[string]int{"fail": 0, "failed": 0, "incomplete": 1, "running": 2, "pass": 3, "passed": 3}
	cyc := func(c proto.Capsule) float64 {
		if v, ok := f(c.Cycles["L3"]); ok {
			return v
		}
		if v, ok := f(c.Cycles["L2"]); ok {
			return v
		}
		return -1
	}
	sort.SliceStable(caps, func(i, j int) bool {
		a, b := caps[i], caps[j]
		switch mode {
		case "status":
			ra, ok := rank[a.Status]
			if !ok {
				ra = 1
			}
			rb, ok := rank[b.Status]
			if !ok {
				rb = 1
			}
			if ra != rb {
				return ra < rb
			}
		case "cohort":
			if a.Cohort != b.Cohort {
				return a.Cohort < b.Cohort
			}
		case "op":
			if a.Op != b.Op {
				return a.Op < b.Op
			}
		case "cycles":
			return cyc(a) > cyc(b)
		}
		return a.Name < b.Name
	})
	// Fixed columns: status, cohort, op, tiers, cycles; the rest share what is left.
	rest := max(20, w-36)
	nameW, shapeW, originW, failW := rest*32/100, rest*21/100, rest*22/100, 0
	if rest < 40 {
		nameW, shapeW, originW = rest*60/100, 0, 0
	}
	failW = max(6, rest-nameW-shapeW-originW-3)
	col := func(s string, w int) string {
		if w <= 0 {
			return ""
		}
		return fit(s, w) + " "
	}
	header := []string{dim("  " + col("capsule", nameW) + col("coh", 5) + col("op", 9) + col("shapes", shapeW) + col("origin", originW) +
		col("L0123", 5) + fmt.Sprintf("%8s ", "cycles") + "failure")}
	sel, lines := []Row{}, []string{}
	for _, c := range caps {
		origin := originText(c.Origin)
		shapes := shortShapes(c.Shapes)
		if !m.matches(strings.Join([]string{c.Name, c.Cohort, c.Op, shapes, origin, c.Status, c.Family, c.Failure["category"]}, " ")) {
			continue
		}
		tiers := ""
		for _, t := range []string{"L0", "L1", "L2", "L3"} {
			tiers += tierGlyph(c.Tiers[t])
		}
		cy := "—"
		if v := cyc(c); v >= 0 {
			cy = humanF(v)
		}
		coh := c.Cohort
		if len(coh) > 5 {
			coh = coh[:5]
		}
		lines = append(lines, tierGlyph(c.Status)+" "+col(paint(cText, clean(c.Name)), nameW)+col(dim(clean(coh)), 5)+col(paint(cL2, clean(c.Op)), 9)+
			col(paint(cText, clean(shapes)), shapeW)+col(dim(clean(origin)), originW)+tiers+"  "+fmt.Sprintf("%8s ", cy)+
			fit(paint(cBad, clean(c.Failure["category"])), failW))
		sel = append(sel, Row{Kind: "capsules", Key: c.Name})
	}
	if len(st.Capsules.Capsules) == 0 {
		lines = append(lines, dim("capsule rows hidden or not recorded"))
		for _, k := range sortedKeys(st.Capsules.Counts["cohort"]) {
			lines = append(lines, dim(fmt.Sprintf("  %s %.0f", k, st.Capsules.Counts["cohort"][k])))
		}
	}
	lines = append(lines, notes(st.Capsules.Notes, w)...)
	hint := fmt.Sprintf("%d · sort %s", len(st.Capsules.Capsules), mode)
	return "CAPSULES", hint, m.list("capsules", header, sel, lines, h, focused)
}

// shortShapes drops operand names ("x:1x56x56x64 i8" -> "1x56x56x64 i8") for the table; the inspector keeps them.
func shortShapes(shapes []string) string {
	out := []string{}
	for _, s := range shapes {
		if _, after, ok := strings.Cut(s, ":"); ok {
			s = after
		}
		out = append(out, s)
	}
	return strings.Join(out, ",")
}

func originText(o map[string]any) string {
	parts := []string{}
	for _, k := range []string{"model", "program", "aten_op"} {
		if v, ok := o[k].(string); ok && v != "" {
			parts = append(parts, v)
		}
	}
	if len(parts) == 0 {
		if v, ok := o["reference"].(string); ok {
			return v
		}
	}
	return strings.Join(parts, "/")
}

// --- passes ------------------------------------------------------------------------------------------

func (m *Model) pPasses(st *proto.State, w, h int, focused bool) (string, string, []string) {
	p := st.Passes
	c := p.Counts
	head := []string{dim("library ") + paint(cText, intOr(c["library_modules"], "—")) + dim(" catalog ") + paint(cText, intOr(c["catalog_mechanisms"], "—")) +
		dim(" declared ") + paint(cText, intOr(c["declared"], "—")) + dim(" used ") + paint(cText, intOr(c["used"], "—")) +
		dim(" profiled ") + paint(cText, intOr(c["profiled"], "—"))}
	if len(p.Pipeline) > 0 {
		pipe := []string{}
		for _, n := range p.Pipeline {
			pipe = append(pipe, clean(n))
		}
		head = append(head, paint(cBuild, "▶ ")+dim(strings.Join(pipe, " → ")))
	}
	head = append(head, dim(fmt.Sprintf("  %-24s %-7s %2s %4s %4s %6s %7s", "pass", "kind", "DU", "req", "ex", "churn", "prof")))
	passes := append([]proto.Pass{}, p.Passes...)
	sort.SliceStable(passes, func(i, j int) bool {
		ci, _ := f(passes[i].Churn)
		cj, _ := f(passes[j].Churn)
		if passes[i].Used != passes[j].Used {
			return passes[i].Used
		}
		return ci > cj
	})
	sel, lines := []Row{}, []string{}
	for _, ps := range passes {
		if !m.matches(ps.Name + " " + ps.Kind + " " + ps.Boundary + " " + ps.Status) {
			continue
		}
		du := yes(ps.Declared, "D") + yes(ps.Used, "U")
		g := paint(cFaint, "·")
		if ps.Used {
			g = paint(cGood, "●")
		} else if ps.Declared {
			g = paint(cWarn, "○")
		}
		lines = append(lines, g+" "+fit(paint(cText, clean(ps.Name)), 24)+" "+fit(dim(clean(ps.Kind)), 7)+" "+du+
			fmt.Sprintf(" %4s %4s %6s %7s", intOr(ps.RequiredBy, "·"), intOr(ps.ExercisedBy, "·"), intOr(ps.Churn, "·"), durP(ps.ProfileWallS)))
		sel = append(sel, Row{Kind: "passes", Key: ps.Name})
	}
	if len(passes) == 0 {
		lines = append(lines, dim("no pass records"))
	}
	lines = append(lines, notes(p.Notes, w)...)
	return "PASSES", "D declared U used · req/ex capsules", m.list("passes", head, sel, lines, h, focused)
}

func yes(b bool, s string) string {
	if b {
		return paint(cGood, s)
	}
	return paint(cFaint, "·")
}

// --- events -------------------------------------------------------------------------------------------

func (m *Model) pEvents(st *proto.State, w, h int, focused bool) (string, string, []string) {
	sel, lines := []Row{}, []string{}
	for _, e := range st.Events.Events {
		if !m.matches(e.Kind + " " + e.Text + " " + e.Agent) {
			continue
		}
		g, c := sevStyle(e.Severity)
		if e.Severity == "info" || e.Severity == "" {
			g, c = "·", kindColor[e.Kind]
			if c == nil {
				c = cDim
			}
		}
		who := ""
		if e.Agent != "" {
			who = paint(cAgent, clean(e.Agent)) + " "
		}
		lines = append(lines, dim(clock(e.T))+" "+paint(c, g)+" "+fit(dim(clean(e.Kind)), 9)+" "+who+paint(cText, clean(e.Text)))
		sel = append(sel, Row{Kind: "local", Key: e.Kind, Local: localDetail("event", e)})
	}
	if len(lines) == 0 {
		lines = append(lines, dim("no events recorded"))
	}
	return "EVENTS", fmt.Sprintf("%d newest first", len(st.Events.Events)), m.list("events", nil, sel, lines, h, focused)
}
