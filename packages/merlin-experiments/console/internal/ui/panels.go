package ui

import (
	"fmt"
	"image/color"
	"math"
	"sort"
	"strings"

	"charm.land/lipgloss/v2"
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
	case "agentview":
		title, hint, body = m.pAgentView(st, inner, h, focused)
	case "gantt":
		title, hint, body = m.pGantt(st, inner, h, focused)
	case "agents":
		title, hint, body = m.pAgents(st, inner, h, focused)
	case "tokens":
		title, hint, body = m.pTokens(st, inner, h, focused)
	case "cpu":
		title, hint, body = m.pCores(st, inner, h, focused)
	case "perf":
		title, hint, body = m.pPerf(st, inner, h, focused)
	case "capsules":
		title, hint, body = m.pCapsules(st, inner, h, focused)
	case "passes":
		title, hint, body = m.pPasses(st, inner, h, focused)
	case "events":
		title, hint, body = m.pEvents(st, inner, h, focused)
	default:
		if isInsight(id) {
			title, hint, body = m.pInsight(st, id, inner, h, focused)
		}
	}
	if e := sectionError(st, id); e != "" {
		body = append([]string{paint(cBad, "reader error: "+clean(e))}, body...)
	}
	return box(id, title, hint, body, w, h, focused)
}

func sectionError(st *proto.State, id string) string {
	src := map[string]string{"gantt": "timeline", "cpu": "activity", "alerts": "", "agentview": "agent_view"}[id]
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
	case "agent_view":
		return st.AgentView.Error
	case "timeline":
		return firstNonEmpty(st.Timeline.Error, st.Jobs.Error)
	case "machine":
		return firstNonEmpty(st.Machine.Error, st.Resources.Error)
	case "activity":
		return firstNonEmpty(st.Activity.Error, st.Resources.Error)
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
	now := st.GeneratedAt
	label := func(s string) string { return fit(dim(s), 14) }
	// 1. Background grades: the ones grading now, else the last one that graded something.
	inFlight := 0
	for _, gr := range p.Grades {
		if gr.State != "grading" && gr.State != "unfinished" {
			continue
		}
		if inFlight < 2 {
			name := ""
			if inFlight == 0 {
				name = map[string]string{"grading": "grading now", "unfinished": "unfinished"}[gr.State]
			}
			out = append(out, label(name)+gradeLine(gr, now, w-14, st.Experiment.State == "LIVE"))
		}
		inFlight++
	}
	if inFlight > 2 {
		out = append(out, label("")+dim(fmt.Sprintf("+%d more grading", inFlight-2)))
	}
	if inFlight == 0 {
		for _, gr := range p.Grades {
			if n, ok := f(gr.NCompleted); ok && n > 0 {
				out = append(out, label("last grade")+gradeLine(gr, now, w-14, false))
				break
			}
		}
	}
	if len(p.Grades) == 0 && p.Published == nil {
		out = append(out, label("grades")+dim("no grade recorded yet"))
	}
	// 2. What the agent was last shown: qa/progress.json, which can trail the grades above.
	if pub := p.Published; pub != nil {
		g := paint(cText, intOr(pub.NCompleted, "—")) + dim("/") + paint(cText, intOr(pub.NSelected, "—")) + dim(" graded · ") +
			paint(cGood, intOr(pub.NPassed, "—")+" pass")
		if v := p.Verdict; v != nil && len(v.TiersGraded) > 0 {
			g += dim(" · " + strings.Join(v.TiersGraded, " "))
			if len(v.TiersNotRun) > 0 {
				g += dim(" only, ") + paint(cWarn, strings.Join(v.TiersNotRun, " ")+" not run")
			}
		}
		if t, ok := f(pub.UpdatedAt); ok && now > 0 {
			g += dim(" · " + dur(max(0, now-t)) + " old")
			newest := 0.0
			for _, gr := range p.Grades {
				if u, ok := f(gr.UpdatedAt); ok && u > newest {
					newest = u
				}
			}
			if newest-t > 600 {
				g += paint(cWarn, " · "+dur(newest-t)+" behind the grader")
			}
		}
		out = append(out, label("agent sees")+g)
	}
	// 3. Certificates kept in the tier ledger, and the joint L3 obligation.
	if c := p.Certificates; c != nil {
		parts := []string{}
		if c.L2 != nil {
			parts = append(parts, dim("L2 ")+paint(cGood, fmt.Sprintf("%.0f", c.L2["pass"])))
		}
		if c.L3 != nil {
			t := dim("L3 ") + paint(cGood, fmt.Sprintf("%.0f", c.L3["pass"]))
			if c.L3["pending"] > 0 {
				t += dim(fmt.Sprintf(" +%.0f pending", c.L3["pending"]))
			}
			parts = append(parts, t)
		}
		if ph := c.Physical; ph != nil {
			t := dim("L3 obligation ") + paint(cText, intOr(ph.Passed, "—")+"/"+intOr(ph.Required, "—"))
			if n, ok := f(ph.Failed); ok && n > 0 {
				t += paint(cBad, fmt.Sprintf(" %.0f failed", n))
			}
			parts = append(parts, t)
		}
		if len(parts) > 0 {
			out = append(out, label("certified")+strings.Join(parts, dim(" · ")))
		}
	}
	// 3. Results by cohort from the capsule ledger: pass, fail, in progress, not run yet.
	byStatus := map[string]map[string]float64{}
	for key, n := range st.Capsules.Counts["cohort_status"] {
		cohort, status, ok := strings.Cut(key, "/")
		if !ok {
			continue
		}
		if byStatus[cohort] == nil {
			byStatus[cohort] = map[string]float64{}
		}
		byStatus[cohort][status] += n
	}
	if len(p.Cohorts) > 0 {
		barW := max(6, w-14-34)
		for _, c := range p.Cohorts {
			total, ok := f(c.Total)
			if !ok || c.Name == "total" {
				continue
			}
			counts := byStatus[c.Name]
			pass, l2, fail, gated, busy := 0.0, 0.0, 0.0, 0.0, 0.0
			for k, v := range counts {
				switch k {
				case "pass", "passed":
					pass += v
				case "l2_pass":
					l2 += v
				case "fail", "failed", "mismatch", "error":
					fail += v
				case "gated":
					gated += v
				case "running", "queued", "pending", "in_progress", "grading":
					busy += v
				} // incomplete, not run, not recorded: not yet
			}
			segs := stackBar([]float64{pass, l2, fail, gated, busy}, []color.Color{cGood, cInfo, cBad, cVerify, cWarn}, total, barW)
			out = append(out, fit(paint(cText, clean(c.Name)), 14)+segs+paint(cGood, fmt.Sprintf(" %4.0f", pass))+paint(cInfo, fmt.Sprintf(" %4.0f", l2))+paint(cBad, fmt.Sprintf(" %4.0f", fail))+
				paint(cVerify, fmt.Sprintf(" %5.0f", gated))+dim(fmt.Sprintf("  /%.0f", total)))
		}
	} else {
		out = append(out, label("by cohort")+dim("not recorded"))
	}
	if s := p.Series["n_passed"]; len(s) > 0 {
		tail := " " + paint(cText, human(s[len(s)-1])+" pass") + dim(fmt.Sprintf(" · %d grades", len(s)))
		if p.SeriesScope != "" && p.SeriesScope != "background grades" {
			tail += dim(" (" + p.SeriesScope + ")")
		}
		stopped := 0
		for _, gr := range p.Grades {
			if gr.State == "stopped" {
				stopped++
			}
		}
		if stopped > 0 {
			tail += dim(fmt.Sprintf(" · %d stopped before grading", stopped))
		}
		out = append(out, label("pass / grade")+sparkline(s, max(4, min(len(s)*2, w-14-lipgloss.Width(tail))), cGood)+tail)
	}
	out = append(out, label("numerics")+paint(cGood, intOr(p.Numeric.Passed, "—"))+dim(" runs correct · ")+
		paint(cBad, intOr(p.Numeric.Failed, "—"))+dim(" wrong · across all grades"))
	if c := p.Completion; c != nil {
		out = append(out, label("completed")+paint(cText, clean(c.Reason))+dim(" after ")+paint(cText, durP(c.ElapsedS)))
	}
	// 4. Compiler stages with a known state only (unknown ones are noise here).
	chips, moved := []string{}, false
	for _, s := range st.Stages.Stages {
		if s.State == "unknown" || s.State == "" {
			continue
		}
		moved = moved || s.State != "pending"
		g, c := stageStyle(s.State)
		chips = append(chips, paint(c, g+" "+clean(s.ID)))
	}
	if moved {
		out = append(out, label("stages")+strings.Join(chips, dim(" · ")))
	}
	m.rows["progress"] = []Row{{Kind: "local", Local: localDetail("progress", p)}}
	legend := paint(cGood, "■") + dim("pass ") + paint(cInfo, "■") + dim("L2 pass, L3 pending ") + paint(cBad, "■") + dim("fail ") + paint(cVerify, "■") + dim("gated ") +
		paint(cWarn, "■") + dim("running ") + paint(cFaint, "·") + dim("not yet on this compiler")
	return "PROGRESS", legend, out
}

// gradeLine is one background grade: a bar of passed, failed and errored capsules against the
// selection, the counts, and how long ago it last reported (the sign that it is still moving).
func gradeLine(gr proto.Grade, now float64, w int, live bool) string {
	sel, _ := f(gr.NSelected)
	pass, _ := f(gr.NPassed)
	fail, _ := f(gr.NFailed)
	errs, _ := f(gr.NErrors)
	l2, _ := f(gr.NL2)
	counts := paint(cText, intOr(gr.NCompleted, "—")) + dim("/"+intOr(gr.NSelected, "—")) + " " +
		paint(cGood, fmt.Sprintf("%.0f✔", pass))
	if l2 > 0 {
		counts += " " + paint(cInfo, fmt.Sprintf("%.0f L2✔", l2)) + dim(" (L3 pending)")
	}
	counts += " " + paint(cBad, fmt.Sprintf("%.0f✖", fail))
	if errs > 0 {
		counts += " " + paint(cWarn, fmt.Sprintf("%.0f!", errs))
	}
	tail := ""
	if t, ok := f(gr.UpdatedAt); ok && now > 0 {
		switch gr.State {
		case "grading":
			c := cText
			if live && now-t > 1200 {
				c = cWarn
			}
			tail = dim(" · last result ") + paint(c, dur(max(0, now-t))+" ago")
		case "unfinished":
			tail = dim(" · last result " + clock(gr.UpdatedAt) + ", never finished")
		default:
			tail = dim(" · ended " + clock(gr.UpdatedAt) + " · " + gr.State)
		}
	}
	if s, ok := f(gr.StartedAt); ok && now > 0 && gr.State == "grading" {
		tail = dim(" · for "+dur(now-s)) + tail
	}
	barW := max(4, min(20, w-lipgloss.Width(counts)-lipgloss.Width(tail)-2))
	return stackBar([]float64{pass, l2, fail, errs}, []color.Color{cGood, cInfo, cBad, cWarn}, sel, barW) + " " + counts + tail
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
		if !m.matches(d.Title + " " + d.Detail + " " + d.Code + " " + d.Action) {
			continue
		}
		g, c := sevStyle(d.Severity)
		times := ""
		if d.Occurrences > 1 {
			times = dim(fmt.Sprintf(" ×%d", d.Occurrences))
		}
		lost := ""
		if v, ok := f(d.ImpactS); ok && v >= 60 {
			lost = dim(" · " + dur(v) + " at stake")
		}
		row := Row{Kind: "local", Key: d.Code, Local: append(localDetail(d.Code, d), d.Items...)}
		lines = append(lines, paint(c, g)+" "+bold(cText, clean(d.Title))+times+lost)
		rows = append(rows, row)
		if d.Detail != "" {
			lines = append(lines, "   "+dim(clean(d.Detail)))
			rows = append(rows, row)
		}
		if d.Action != "" {
			lines = append(lines, "   "+paint(cInfo, "→ "+clean(d.Action)))
			rows = append(rows, row)
		}
	}
	if len(lines) == 0 {
		lines = append(lines, paint(cGood, "nothing stands out in the recorded numbers"))
	}
	header := []string{}
	if v := st.Timeline.Verdict; v != nil {
		header = append(header, dim("critical path: ")+bold(cWarn, strings.ReplaceAll(v.Bound, "_", " ")+fmt.Sprintf(" %.0f%%", v.SharePct))+
			dim(" of the lead's time · "+clean(v.Explanation)))
	}
	hint := fmt.Sprintf("%d findings, worst first · ↵ detail", len(st.Diagnostics))
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
	if win := m.win().seconds; win > 0 {
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
	now := st.GeneratedAt
	// Fixed right-hand columns: state, last record, time worked, tokens, failed commands.
	// Narrow panels drop the token and failure columns before squeezing names.
	wide := w >= 84
	right := 28
	if wide {
		right = 48
	}
	nameW := max(12, w-right)
	m.agentWide = wide
	header := []string{}
	if wide {
		header = append(header, dim(fit("agent", nameW)+fmt.Sprintf("%-9s %9s %7s %7s %9s", "state", "last rec", "worked", "tokens", "failures")))
	} else {
		header = append(header, dim(fit("agent", nameW)+fmt.Sprintf("%-9s %9s %7s", "state", "last rec", "worked")))
	}
	// Agents by round, newest round first; a round with no agents still says what happened to it.
	byRound := map[int][]proto.Agent{}
	for _, ag := range a.Agents {
		r := -1
		if v, ok := f(ag.Round); ok {
			r = int(v)
		}
		byRound[r] = append(byRound[r], ag)
	}
	turnsOf := map[int][]proto.Turn{}
	for _, t := range st.Lifecycle.Turns {
		if v, ok := f(t.Round); ok {
			turnsOf[int(v)] = append(turnsOf[int(v)], t)
		}
	}
	rounds := []int{}
	seen := map[int]bool{}
	for r := range byRound {
		rounds, seen[r] = append(rounds, r), true
	}
	for r := range turnsOf {
		if !seen[r] {
			rounds = append(rounds, r)
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(rounds)))
	rows, lines := []Row{}, []string{}
	newestWithAgents := -2
	for _, r := range rounds {
		if len(byRound[r]) > 0 {
			newestWithAgents = r
			break
		}
	}
	for _, r := range rounds {
		agents := byRound[r]
		label := "agents"
		if r >= 0 {
			label = fmt.Sprintf("round %d", r)
		}
		summary := roundSummary(turnsOf[r], agents, now)
		lines = append(lines, bold(cHeading, label)+dim(" · ")+summary)
		rows = append(rows, Row{Kind: "lifecycle", Key: "turns"})
		if len(agents) == 0 {
			continue
		}
		if r != newestWithAgents && !(m.zoom && m.focus == "agents") {
			tok := 0.0
			for _, ag := range agents {
				if v, ok := f(ag.Tokens.Total); ok {
					tok += v
				}
			}
			lines = append(lines, dim(fmt.Sprintf("   %d agents · %s tokens · z to expand", len(agents), humanF(tok))))
			rows = append(rows, Row{Kind: "lifecycle", Key: "turns"})
			continue
		}
		m.agentTree(agents, now, nameW, h, &rows, &lines)
	}
	if len(a.Agents) == 0 && len(rounds) == 0 {
		lines = append(lines, dim("no agent records"))
	}
	t := a.Totals
	hint := fmt.Sprintf("%s working · helpers ≤%s · ↵ commands", intOr(t.Working, "—"), intOr(t.HelperCeiling, "?"))
	return "AGENT TREE", hint, m.list("agents", header, rows, lines, h, focused)
}

// roundSummary is one line for a round: how its turns went and how it ended.
func roundSummary(turns []proto.Turn, agents []proto.Agent, now float64) string {
	if len(turns) == 0 {
		return dim(fmt.Sprintf("%d agents", len(agents)))
	}
	worked := 0.0
	for _, t := range turns {
		if v, ok := f(t.AgentS); ok {
			worked += v
		}
	}
	last := turns[len(turns)-1]
	parts := []string{fmt.Sprintf("%d turn(s), agent worked %s", len(turns), dur(worked))}
	switch {
	case last.OpenStage != nil:
		return paint(actColor("startup"), "starting turn "+intOr(last.Turn, "?")+" · "+clean(last.OpenStage.Stage)) + dim(" · "+parts[0])
	case last.AgentEnd == nil && last.AgentStart != nil:
		return bold(cGood, "agent working") + dim(" · "+parts[0])
	}
	end := ""
	if v, ok := f(last.AgentEnd); ok && now > 0 {
		end = " " + dur(math.Max(0, now-v)) + " ago"
	}
	reason := clean(last.EndReason)
	if worked == 0 {
		return paint(cBad, "no agent ran") + dim(" · ended"+end+": "+reason)
	}
	return dim(parts[0] + " · ended" + end + ": " + reason)
}

// agentTree appends one round's agents as a tree (lead first, helpers under their parent).
func (m *Model) agentTree(agents []proto.Agent, now float64, nameW, h int, rows *[]Row, lines *[]string) {
	byParent := map[string][]proto.Agent{}
	ids := map[string]bool{}
	for _, ag := range agents {
		ids[ag.ID] = true
	}
	roots := []proto.Agent{}
	for _, ag := range agents {
		if ag.ParentID != "" && ids[ag.ParentID] && ag.ParentID != ag.ID {
			byParent[ag.ParentID] = append(byParent[ag.ParentID], ag)
		} else {
			roots = append(roots, ag)
		}
	}
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
		if ag.Path != "" && ag.Path != "/root" {
			path = dim(" " + clean(ag.Path))
		}
		head := " " + dim(prefix+branch) + paint(c, g) + " " + bold(cAgent, clean(name)) + path
		head = fit(head, nameW-1) + " "
		seen := "—"
		if v, ok := f(ag.LastActivityAt); ok && now > 0 {
			seen = dur(math.Max(0, now-v)) + " ago"
		}
		fails := ""
		if fc, ok := f(ag.FailedCommands); ok && fc > 0 {
			fails = paint(cBad, fmt.Sprintf("%9s", fmt.Sprintf("%.0f cmds", fc)))
		}
		cols := fit(paint(c, stateWords[ag.State]), 9) + dim(fmt.Sprintf(" %9s ", seen)) + paint(cText, fmt.Sprintf("%7s", durP(ag.ActiveS)))
		if m.agentWide {
			cols += paint(cText, fmt.Sprintf(" %7s", human(ag.Tokens.Total))) + " " + fails
		}
		if m.matches(name + " " + ag.Path + " " + ag.State) {
			*lines = append(*lines, fit(head, nameW)+cols)
			*rows = append(*rows, Row{Kind: "agents", Key: ag.ID})
			if ag.Current != "" && h > 10 && ag.State == "working" {
				*lines = append(*lines, dim(" "+prefix+"   ↳ ")+paint(cText, clean(ag.Current)))
				*rows = append(*rows, Row{Kind: "agents", Key: ag.ID})
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
}

// State words for agents: what the record says, not a guess.
var stateWords = map[string]string{
	"working": "working", "completed": "finished", "finished": "finished", "aborted": "aborted",
	"stopped": "stopped", "observed": "seen", "": "—",
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
		refused := []string{}
		for _, g := range gs {
			if !g.Comparable {
				refused = append(refused, clean(g.Subject)+": "+clean(g.Reason))
				continue
			}
			txt := fmt.Sprintf("vs golden %s: %s / %s = %s", clean(g.Subject), human(g.CandidateCycles), human(g.ReferenceCycles), ratio(g.Ratio))
			lines = append(lines, paint(cCalib, "★ ")+paint(cText, txt))
			sel = append(sel, Row{Kind: "local", Local: localDetail("golden", g)})
		}
		if len(refused) > 0 {
			// Refusals are one line: they say why no comparison is made, not a result per reference.
			lines = append(lines, paint(cCalib, "★ ")+dim(fmt.Sprintf("%d golden references not comparable yet · ↵ why", len(refused))))
			sel = append(sel, Row{Kind: "local", Key: "golden", Local: refused})
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
	// Name and failure matter most; shapes next; origin only when there is room.
	rest := max(20, w-36)
	nameW, shapeW, originW, failW := rest*36/100, rest*22/100, 0, 0
	if rest >= 90 {
		nameW, shapeW, originW = rest*30/100, rest*20/100, rest*20/100
	} else if rest < 40 {
		nameW, shapeW = rest*55/100, 0
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
		glyph, failure := tierGlyph(c.Status), paint(cBad, clean(c.Failure["category"]))
		if c.Stale {
			// The newest result graded an older compiler: shown faint, never as this compiler's failure.
			glyph, failure = paint(cFaint, "·"), dim("old compiler: "+clean(firstNonEmpty(c.Failure["category"], c.Status)))
		}
		lines = append(lines, glyph+" "+col(paint(cText, clean(c.Name)), nameW)+col(dim(clean(coh)), 5)+col(paint(cL2, clean(c.Op)), 9)+
			col(paint(cText, clean(shapes)), shapeW)+col(dim(clean(origin)), originW)+tiers+"  "+fmt.Sprintf("%8s ", cy)+
			fit(failure, failW))
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
	head := []string{}
	profiled := false
	for _, ps := range p.Applied {
		if _, ok := f(ps.ProfileWallS); ok {
			profiled = true
		}
	}
	counted := false
	for _, ps := range p.Applied {
		if _, ok := f(ps.ExercisedBy); ok {
			counted = true
		}
		if _, ok := f(ps.Churn); ok {
			counted = true
		}
	}
	cols := 0
	if counted {
		cols += 14
	}
	if profiled {
		cols += 8
	}
	nameW := max(12, w-6-cols)
	timeH := ""
	if profiled {
		timeH = fmt.Sprintf(" %7s", "time")
	}
	countH := ""
	if counted {
		countH = fmt.Sprintf("%5s %6s", "caps", "churn")
	}
	head = append(head, dim(fit("  #  pass, in pipeline order", nameW+5)+countH+timeH))
	sel, lines := []Row{}, []string{}
	for _, ps := range p.Applied {
		if !m.matches(ps.Name + " " + ps.Kind + " " + ps.Boundary) {
			continue
		}
		order := "  ·"
		if v, ok := f(ps.Order); ok {
			order = fmt.Sprintf("%3.0f", v)
		}
		kind := ""
		if ps.Kind != "command" {
			kind = dim(" (" + clean(ps.Kind) + ")")
		}
		if ps.Boundary != "" {
			kind += dim(" ⟂" + clean(ps.Boundary))
		}
		timeC := ""
		if profiled {
			timeC = fmt.Sprintf(" %7s", durP(ps.ProfileWallS))
		}
		countC := ""
		if counted {
			countC = fmt.Sprintf("%5s %6s", intOr(ps.ExercisedBy, "·"), intOr(ps.Churn, "·"))
		}
		lines = append(lines, paint(cBuild, order)+"  "+fit(paint(cText, clean(ps.Name))+kind, nameW)+countC+timeC)
		sel = append(sel, Row{Kind: "passes", Key: ps.Name})
	}
	if len(p.Applied) == 0 {
		lines = append(lines, dim("the compiler's pass pipeline is not recorded yet"))
	}
	foot := []string{dim("⟂ the shared-pass boundary the pass implements")}
	if counted {
		foot = append(foot, dim("caps: capsules whose result used it · churn: cached results it invalidated"))
	} else {
		foot = append(foot, dim("no per-pass usage or profile recorded yet"))
	}
	lines = append(lines, foot...)
	return "PASSES", fmt.Sprintf("%d applied by the compiler · ↵ detail", len(p.Applied)), m.list("passes", head, sel, lines, h, focused)
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
		tag, c := eventTag(e.Kind)
		mark := " "
		switch e.Severity {
		case "warn", "warning":
			mark = paint(cWarn, "▲")
		case "error":
			mark = paint(cBad, "✖")
		}
		who := ""
		if e.Agent != "" {
			who = paint(cAgent, clean(e.Agent)) + " "
		}
		lines = append(lines, dim(clock(e.T))+" "+bold(c, fit(tag, 7))+mark+" "+who+paint(cText, clean(e.Text)))
		sel = append(sel, Row{Kind: "local", Key: e.Kind, Local: localDetail("event", e)})
	}
	if len(lines) == 0 {
		lines = append(lines, dim("no events recorded"))
	}
	return "EVENTS", "newest first · author agent check sim grade", m.list("events", nil, sel, lines, h, focused)
}

func eventTag(kind string) (string, color.Color) {
	switch kind {
	case "author":
		return "AUTHOR", actColor("startup")
	case "agent":
		return "AGENT", cAgent
	case "check":
		return "CHECK", cVerify
	case "sim":
		return "SIM", cL3
	case "grade":
		return "GRADE", cGood
	case "action":
		return "ACTION", cBuild
	case "stop":
		return "STOP", cBad
	}
	return strings.ToUpper(kind), cDim
}
