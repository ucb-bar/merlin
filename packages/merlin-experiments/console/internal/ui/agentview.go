package ui

import (
	"fmt"
	"strings"

	"github.com/ucb-bar/merlin/packages/merlin-experiments/console/internal/proto"
)

// pAgentView shows what the author agent can see of its own feedback, from sources/agent_view.py.
func (m *Model) pAgentView(st *proto.State, w, h int, focused bool) (string, string, []string) {
	v := st.AgentView
	now := st.GeneratedAt
	lines, rows := []string{}, []Row{}
	add := func(line string, detail ...string) {
		lines = append(lines, line)
		rows = append(rows, Row{Kind: "local", Key: "agent_view", Local: append([]string{clean(v.Workspace)}, detail...)})
	}
	if v.Workspace == "" {
		add(dim("no agent workspace recorded for this run"))
		return "AGENT VIEW", "", m.list("agentview", nil, rows, lines, h, focused)
	}
	file := func(label string, a proto.AgentFile) {
		name := fit(dim(label), 15)
		switch {
		case !a.Present:
			add(name + dim("— absent"))
		case a.ModelGateOnly:
			add(name + paint(cWarn, "✗ model gate only") + dim(" · "+ago(now, a.At)))
		case a.Rows == nil || *a.Rows == 0:
			add(name + paint(cWarn, "✗ no rows") + dim(" · "+ago(now, a.At)))
		default:
			state := ""
			if a.SnapshotCurrent != nil {
				if *a.SnapshotCurrent {
					state = paint(cGood, " current")
				} else {
					state = dim(" not current")
				}
			}
			extra := ""
			if hr := a.Headroom; hr != nil {
				extra = dim(" · ranked ") + paint(cText, intOr(hr.Ranked, "—")) + dim(" util ") + paint(cText, pct(hr.MedianPct, 1))
			}
			add(name + paint(cGood, "✓ ") + paint(cText, intOr(a.Rows, "—")+" rows") + state + dim(" · "+ago(now, a.At)) + extra)
		}
		for _, n := range a.Notices {
			add("  " + paint(cInfo, "notice: "+clean(n)))
		}
	}
	file("verdict.json", v.Verdict)
	file("latest grade", v.LatestCompleted)
	file("progress.json", v.Progress)
	if g := v.NewestGrade; g != nil {
		delivered := paint(cBad, "reached nothing the agent reads")
		if len(g.Delivered) > 0 {
			delivered = paint(cGood, "→ "+strings.Join(g.Delivered, ", "))
		}
		add(fit(dim("newest grade"), 15)+paint(cText, intOr(g.Rows, "—")+" rows")+dim(" · "+ago(now, g.At)+" ")+delivered,
			"grade token "+clean(g.Token), "graded source "+clean(g.GradedSource))
	}
	sc := v.Selfchecks
	pending := ""
	if p, ok := f(sc.Pending); ok && p > 0 {
		pending = dim(" · ") + paint(cWarn, fmt.Sprintf("%.0f pending", p)) + dim(" oldest "+ago(now, sc.OldestPendingAt))
	}
	add(fit(dim("self-checks"), 15) + paint(cText, intOr(sc.Answered, "—")+"/"+intOr(sc.Requests, "—")+" answered") + pending +
		dim(" · last "+ago(now, sc.LastResponseAt)))
	if sc.LastError != "" {
		add("  "+paint(cBad, "last error: ")+dim(clean(sc.LastError)), clean(sc.LastError))
	}
	sg := v.Suggestions
	if sg.Present {
		add(fit(dim("suggestions"), 15) + paint(cText, intOr(sg.Suggestions, "—")+" supported · "+intOr(sg.Pending, "—")+" need IR · "+intOr(sg.Discovery, "—")+" open"))
	} else {
		add(fit(dim("suggestions"), 15) + paint(cWarn, "✗ not staged"))
	}
	for _, n := range v.Notes {
		add(dim(clean(n)))
	}
	return "AGENT VIEW", "what the agent can read · ↵ detail", m.list("agentview", nil, rows, lines, h, focused)
}

// volumeBadge is the run volume's free space for the header; empty when not measured.
func volumeBadge(v *proto.Volume) string {
	if v == nil {
		return ""
	}
	free, ok := f(v.FreeBytes)
	if !ok {
		return ""
	}
	text := fmt.Sprintf("disk %s free (%s) ", humanF(free), pct(v.FreePct, 0))
	switch v.Level {
	case "critical":
		return bold(cBad, text)
	case "warn":
		return bold(cWarn, text)
	}
	return dim(text)
}
