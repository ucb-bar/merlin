package ui

import (
	"encoding/json"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/ucb-bar/merlin/packages/merlin-experiments/console/internal/proto"
)

const fixture = `{"type":"snapshot","full":true,"generated_at":1792780000,"operator_private":true,
 "experiment":{"id":"/runs/r1","run_id":"exp-v1","task_name":"task-a","run_dir":"/runs/r1","state":"LIVE",
   "state_basis":"runner pid 1 runs this --run-id","started_at":1792776400,"last_record_at":1792779999},
 "sections":{
  "progress":{"state":"running","n_selected":250,"n_completed":20,"n_passed":3,"snapshot_current":false,
    "grades":[{"token":"2","state":"grading","started_at":1792778000,"updated_at":1792779000,"n_selected":250,"n_completed":20,"n_passed":3,"n_failed":16,"n_errors":1},
      {"token":"1","state":"stopped","started_at":1792776000,"updated_at":1792776100,"n_selected":250,"n_completed":0,"n_passed":0}],
    "published":{"updated_at":1792777000,"n_selected":250,"n_completed":6,"n_passed":1},
    "certificates":{"L2":{"pass":3,"pending":0,"fail":0},"L3":{"pass":1,"pending":2,"fail":0},"physical":{"required":153,"passed":0,"failed":0,"pending":153}},
    "cohorts":[{"name":"functional","total":153,"all":3,"L2":10,"L3":2,"refusal":0}],
    "numeric":{"completed":19,"passed":19,"failed":0},"series":{"t":[1792776400,1792778000,1792779000],"n_passed":[0,1,3]}},
  "stages":{"stages":[{"id":"frontend","title":"Frontend","state":"done","basis":"derived","actions":{"done":2,"failed":0,"running":0,"total":2},"capsules":{"required":10,"passing":4}},
     {"id":"lowering","title":"Lowering","state":"blocked","basis":"ledger","actions":{"done":0,"failed":1,"running":0,"total":1},"capsules":{"required":96,"passing":0}}]},
  "agents":{"agents":[{"id":"lead-1","name":"lead-1","role":"lead","state":"working","tokens":{"total":1500000},"active_s":600,
       "spans":[[1792776500,1792777000,"turn"],[1792778000,null,"tool"]],"current":"cat file"},
     {"id":"h1","parent_id":"lead-1","name":"Curie","path":"/root/frontend","role":"subagent","state":"completed","tokens":{"total":20000},"failed_commands":2}],
     "totals":{"agents":2,"working":1,"subagents":1,"helper_ceiling":16}},
  "tokens":{"total":{"fresh_input":1000,"cache_read":90000,"output":500,"reasoning":200,"total":91700},
     "series":{"t":[1792776400,1792778000,1792779000],"fresh_input":[0,500,1000],"cache_read":[0,40000,90000],"output":[0,200,500],"reasoning":[0,100,200]},
     "cache_hit_ratio":0.98,"budget":{"total":40000000,"used_pct":23.5}},
  "resources":{"mode":"live","cores":[{"cpu":9,"util_pct":80,"kind":"L2"},{"cpu":10,"util_pct":null,"kind":"unknown"}],"busy_cpus":0.8,"total_cpus":2,
     "queue":{"waiters":3},"series":{"cpu_pct":[10,20,30]}},
  "jobs":{"spans":[{"lane":"L2","label":"capsule_execution","start":1792778000,"end":1792778600,"status":"returned"}],
     "stages":[{"stage":"capsule_execution","count":5,"p50_s":10,"p95_s":40}],"sim":{"queue_wait_p50":50,"runtime_p50":10}},
  "events":{"events":[{"t":1792779990,"kind":"message","severity":"info","agent":"lead","text":"bad \u001b[2J record"}]},
  "capsules":{"capsules":[{"name":"SY_conv","cohort":"functional","kind":"layer","op":"conv2d","shapes":["x:1x56x56x64 i8"],
     "origin":{"model":"resnet","program":"prefill"},"tiers":{"L0":"pass","L1":"pass","L2":"fail"},"status":"fail","cycles":{"L2":1234},
     "failure":{"category":"NUMERIC_MISMATCH"}}],"counts":{"cohort":{"functional":1}}},
  "perf":{"rows":[{"capsule":"SY_conv","spike_model_cycles":11877,"gsim_cycles":12000,"floor_cycles":36,"efficiency_pct":0.3,
     "intensity":4,"achieved_macs_per_cycle":2,"divergence_pct":1.0,"comparable":true,"top_resource":{"name":"dma","activity_pct":70}}],
     "summary":{"modeled":1},"peak_macs_per_cycle":256},
  "passes":{"passes":[{"name":"lower_interface_to_target","kind":"command","declared":true,"used":true,"churn":120,"required_by":4}],
     "applied":[{"name":"lower_interface_to_target","kind":"command","declared":true,"used":true,"churn":120,"order":1}],
     "counts":{"library_modules":379,"declared":1,"used":1},"pipeline":["parse","lower_interface_to_target"]}
 },
 "diagnostics":[{"code":"QUEUE_DOMINATED","severity":"warn","scope":"jobs","title":"Simulation jobs wait longer than they run","detail":"median 50s vs 10s","occurrences":1}]}`

const experiments = `{"type":"experiments","items":[
 {"id":"/runs/r1","run_id":"exp-v1","task_name":"task-a","state":"LIVE","state_basis":"x","last_record_at":1792779999},
 {"id":"/runs/r0","run_id":"exp-v0","task_name":"task-a","state":"FINISHED","state_basis":"y","last_record_at":1792700000}],
 "selected":"/runs/r1"}`

func loaded(t *testing.T, w, h int) (*Model, *[]map[string]any) {
	t.Helper()
	sent := []map[string]any{}
	m := New(func(cmd map[string]any) { sent = append(sent, cmd) })
	m.W, m.H = w, h
	for _, line := range []string{experiments, fixture} {
		msg, err := proto.Decode([]byte(strings.ReplaceAll(line, "\n", "")))
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		m.Apply(msg)
	}
	return m, &sent
}

func TestFrameIsExactSizeWithEveryPanel(t *testing.T) {
	for _, size := range [][2]int{{220, 56}, {200, 50}, {150, 40}, {120, 34}} {
		m, _ := loaded(t, size[0], size[1])
		frame := m.Render()
		lines := strings.Split(frame, "\n")
		if len(lines) != size[1] {
			t.Fatalf("%v: %d lines", size, len(lines))
		}
		for i, l := range lines {
			if w := ansi.StringWidth(l); w != size[0] {
				t.Fatalf("%v: line %d has width %d", size, i, w)
			}
		}
		plain := ansi.Strip(frame)
		for _, title := range []string{"PROGRESS", "BOTTLENECKS", "TIMELINE", "AGENT TREE", "TOKENS", "CORES",
			"PERFORMANCE", "CAPSULES", "PASSES", "EVENTS"} {
			if !strings.Contains(plain, title) {
				t.Fatalf("%v: missing panel %s", size, title)
			}
		}
		if strings.Contains(frame, "\x1b[2J") {
			t.Fatal("record text reached the terminal as a control sequence")
		}
	}
}

func TestContentIsDrawnFromTheSnapshot(t *testing.T) {
	m, _ := loaded(t, 260, 60)
	plain := ansi.Strip(m.Render())
	for _, want := range []string{"LIVE", "exp-v1", "Curie", "lead", "SY_conv", "conv2d", "lower_interface_to_target",
		"Simulation jobs wait longer", "grading now", "20/250", "behind the grader", "L3 obligation 0/153",
		"stopped before grading", "frontend", "lowering", "exp-v0", "FINISHED", "23.5%"} {
		if !strings.Contains(plain, want) {
			t.Errorf("frame lacks %q", want)
		}
	}
	// The capsule table shows shapes, origin and failure in full when it has the room (zoomed).
	m.focus, m.zoom = "capsules", true
	zoomed := ansi.Strip(m.Render())
	for _, want := range []string{"1x56x56x64", "resnet/prefill", "NUMERIC_MISMATCH"} {
		if !strings.Contains(zoomed, want) {
			t.Errorf("zoomed capsules lack %q", want)
		}
	}
	// A missing value is a dash, never a zero.
	if !strings.Contains(plain, "—") {
		t.Error("no dash for missing values")
	}
}

func key(s string) tea.KeyPressMsg {
	switch s {
	case "tab":
		return tea.KeyPressMsg{Code: tea.KeyTab}
	case "enter":
		return tea.KeyPressMsg{Code: tea.KeyEnter}
	case "esc":
		return tea.KeyPressMsg{Code: tea.KeyEscape}
	}
	return tea.KeyPressMsg{Code: rune(s[0]), Text: s}
}

func TestKeysFocusZoomInspectAndSelect(t *testing.T) {
	m, sent := loaded(t, 220, 56)
	m.Render()
	m.focus = "capsules"
	m.Update(key("enter"))
	if len(*sent) == 0 || (*sent)[len(*sent)-1]["cmd"] != "inspect" || (*sent)[len(*sent)-1]["key"] != "SY_conv" {
		t.Fatalf("enter did not request inspection: %v", *sent)
	}
	if !m.inspectOpen {
		t.Fatal("inspector not open")
	}
	m.Apply(&proto.Inspect{RequestID: m.pendingReq, Title: "capsule SY_conv", Paths: []string{"/runs/r1/x.json"}})
	if !strings.Contains(ansi.Strip(m.Render()), "capsule SY_conv") {
		t.Fatal("inspector content not drawn")
	}
	m.Update(key("esc"))
	if m.inspectOpen {
		t.Fatal("esc did not close the inspector")
	}
	m.Update(key("z"))
	zoomed := ansi.Strip(m.Render())
	if !strings.Contains(zoomed, "CAPSULES") || strings.Contains(zoomed, "AGENT TREE") {
		t.Fatal("zoom did not isolate the focused panel")
	}
	m.Update(key("z"))
	before := m.focus
	m.Update(key("tab"))
	if m.focus == before {
		t.Fatal("tab did not move focus")
	}
	m.focus, m.sidebar = "experiments", 1
	m.Render()
	m.cursor["experiments"] = 0
	for i, r := range m.rows["experiments"] {
		if r.Key == "/runs/r0" {
			m.cursor["experiments"] = i
			break
		}
	}
	m.Update(key("enter"))
	last := (*sent)[len(*sent)-1]
	if last["cmd"] != "select" || last["id"] != "/runs/r0" {
		t.Fatalf("enter on an experiment did not select it: %v", last)
	}
}

func TestFilterNarrowsRows(t *testing.T) {
	m, _ := loaded(t, 220, 56)
	m.focus = "capsules"
	m.Update(key("/"))
	for _, r := range "nomatch" {
		m.Update(tea.KeyPressMsg{Code: r, Text: string(r)})
	}
	m.Update(key("enter"))
	m.Render()
	if len(m.rows["capsules"]) != 0 {
		t.Fatal("filter did not remove non-matching capsules")
	}
}

func TestDeltaSnapshotKeepsUnchangedSections(t *testing.T) {
	m, _ := loaded(t, 200, 50)
	delta := map[string]any{"type": "snapshot", "full": false, "generated_at": 1792780001,
		"experiment": map[string]any{"id": "/runs/r1", "run_id": "exp-v1", "state": "LIVE"},
		"sections":   map[string]any{"events": map[string]any{"events": []any{}}}}
	raw, _ := json.Marshal(delta)
	msg, err := proto.Decode(raw)
	if err != nil {
		t.Fatal(err)
	}
	m.Apply(msg)
	if len(m.State.Capsules.Capsules) != 1 || len(m.State.Events.Events) != 0 {
		t.Fatal("delta did not keep unchanged sections and replace changed ones")
	}
}

func TestBrowserSearchesAndOpensAnExperiment(t *testing.T) {
	m, sent := loaded(t, 220, 56)
	m.Update(key("b"))
	if !m.browse || !strings.Contains(ansi.Strip(m.Render()), "decided by") {
		t.Fatal("b did not open the experiment browser")
	}
	for _, r := range "v0" {
		m.Update(tea.KeyPressMsg{Code: r, Text: string(r)})
	}
	if items := m.browseItems(); len(items) != 1 || items[0].RunID != "exp-v0" {
		t.Fatalf("search did not narrow to exp-v0: %v", items)
	}
	m.Update(key("enter"))
	last := (*sent)[len(*sent)-1]
	if last["cmd"] != "select" || last["id"] != "/runs/r0" || m.browse {
		t.Fatalf("enter in the browser did not open exp-v0: %v", last)
	}
	m.Update(key("b"))
	m.Update(key("tab")) // live only
	if items := m.browseItems(); len(items) != 1 || items[0].State != "LIVE" {
		t.Fatalf("state tab did not filter to live: %v", items)
	}
}

func TestJumpToTheRunningExperiment(t *testing.T) {
	m, sent := loaded(t, 220, 56)
	m.State.Experiment.ID = "/runs/r0" // viewing the finished one
	m.State.Experiment.State = "FINISHED"
	if !strings.Contains(ansi.Strip(m.Render()), "1 running: exp-v1 [L]") {
		t.Fatal("header does not offer the running experiment")
	}
	m.Update(key("L"))
	last := (*sent)[len(*sent)-1]
	if last["cmd"] != "select" || last["id"] != "/runs/r1" {
		t.Fatalf("L did not open the running experiment: %v", last)
	}
}

func TestTimelineCursorReportsTheInstant(t *testing.T) {
	m, _ := loaded(t, 220, 56)
	m.focus = "gantt"
	m.Update(key("."))
	if m.tcursor < 0 {
		t.Fatal("the time cursor did not appear")
	}
	if !strings.Contains(ansi.Strip(m.Render()), "▏") {
		t.Fatal("the cursor report is missing")
	}
	m.Update(key("esc"))
	if m.tcursor != -1 {
		t.Fatal("esc did not clear the cursor")
	}
}

const lifecycleDelta = `{"type":"snapshot","full":false,"generated_at":1792780000,"operator_private":true,
 "experiment":{"id":"/runs/r1","run_id":"exp-v1","task_name":"task-a","run_dir":"/runs/r1","state":"LIVE",
   "started_at":1792776400,"last_record_at":1792779999},
 "sections":{
  "lifecycle":{"phase":"startup","phase_since":1792779000,"phase_detail":"library_overlay for 400s","author_running":false,
    "turns":[{"round":0,"turn":0,"startup_start":1792776500,"startup_end":1792776600,"agent_start":1792776600,"agent_end":1792778000,
      "end_reason":"development boundary"},{"round":0,"turn":1,"startup_start":1792779000,"open_stage":{"stage":"library_overlay","since":1792779600}}]},
  "activity":{"mode":"live","cpus":[9,10],
    "categories":[{"id":"author","code":"a","title":"a"},{"id":"build","code":"b","title":"b"},{"id":"L3","code":"3","title":"3"},
      {"id":"idle","code":".","title":"i"}],
    "history":{"t0":1792779880,"dt":2,"n":60,"since":1792779880,"cores":[
      {"cpu":9,"codes":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","util":"999999999999999999999999999999999999999999999999999999999999"},
      {"cpu":10,"codes":"333333333333333333333333333333..............................","util":"999999999999999999999999999999000000000000000000000000000000"}]},
    "now":[{"cpu":9,"util_pct":95,"code":"b","top":[{"activity":"build","label":"compiler call","pct":95}]},
      {"cpu":10,"util_pct":0,"code":".","top":[]}],
    "totals":{"build":120,"L3":60}}}}`

func TestNowLineAuthorLaneAndCoreGantt(t *testing.T) {
	m, _ := loaded(t, 220, 60)
	msg, err := proto.Decode([]byte(strings.ReplaceAll(lifecycleDelta, "\n", "")))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	m.Apply(msg)
	plain := ansi.Strip(m.Render())
	for _, want := range []string{"NOW", "starting a turn", "library_overlay for 400s", "author turns", "CORES · what each CPU",
		"compiler call", "build 1", "L3 sim", "│ now", "▶ starting", "round 0", "last 30m"} {
		if !strings.Contains(plain, want) {
			t.Errorf("frame lacks %q", want)
		}
	}
	// One row per core when there is room; Enter on a core asks the data process for its detail.
	m.Render()
	rows := m.rows["cpu"]
	if len(rows) != 2 || rows[0].Kind != "activity" || rows[0].Key != "9" {
		t.Fatalf("core rows: %+v", rows)
	}
}

const insightSnapshot = `{"type":"snapshot","full":false,"generated_at":1792780001,"operator_private":true,
 "experiment":{"id":"/runs/r1","run_id":"exp-v1","task_name":"task-a","run_dir":"/runs/r1","state":"LIVE"},
 "sections":{
  "performance":{"title":"PERFORMANCE","summary":[{"label":"test cases measured","value":"133","level":null,"key":null},
     {"label":"modeled vs measured","value":"median ×1.43, 33 off >2×","level":"warn","key":"model_gap"}],
     "table":{"columns":["family / case","n","speedup","util"],"rows":[{"cells":["dense","95","×1.00","3.6%"],"level":null,"key":"dense"},
       {"cells":["conv","25","×1.00","—"],"level":null,"key":"conv"}]},
     "series":[{"label":"geomean speedup vs first","t":[1,2,3],"v":[1.0,1.1,1.3]}],"notes":["cycles are measured gSIM barrier cycles"]},
  "grader":{"title":"GRADER","summary":[{"label":"verdicts finished / delivered","value":"13 / 0","level":"bad","key":"verdicts"}],
     "table":null,"series":[],"notes":[]},
  "integrity":{"error":"OSError: boom"},
  "cases":{"title":"CASES","summary":[{"label":"cases passing","value":"2/3","level":null,"key":null}],
     "table":{"columns":["test case","status","gap"],"rows":[{"cells":["case-big","L3 ✓","×9.0"],"level":"ok","key":"case-big"},
       {"cells":["case-small","L2 ✓","×1.2"],"level":null,"key":"case-small"},
       {"cells":["case-failing","fail numeric","—"],"level":"bad","key":"case-failing"}]},
     "series":[],"notes":[],
     "sorts":[{"label":"worst gap","order":[0,1,2]},{"label":"status","order":[2,0]}]}
 },"changed":["performance","grader","integrity","cases"]}`

func withInsights(t *testing.T, w, h int) (*Model, *[]map[string]any) {
	t.Helper()
	m, sent := loaded(t, w, h)
	msg, err := proto.Decode([]byte(strings.ReplaceAll(insightSnapshot, "\n", "")))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	m.Apply(msg)
	return m, sent
}

func TestInsightPageRendersEveryPanelAtEverySize(t *testing.T) {
	for _, size := range [][2]int{{220, 56}, {200, 50}, {150, 40}, {120, 34}, {90, 30}} {
		m, _ := withInsights(t, size[0], size[1])
		m.Update(key("v"))
		frame := m.Render()
		lines := strings.Split(frame, "\n")
		if len(lines) != size[1] {
			t.Fatalf("%v: %d lines", size, len(lines))
		}
		for i, l := range lines {
			if w := ansi.StringWidth(l); w != size[0] {
				t.Fatalf("%v: line %d has width %d", size, i, w)
			}
		}
		plain := ansi.Strip(frame)
		for _, title := range []string{"PERFORMANCE", "CASES", "GRADER", "TOOLS", "INTEGRITY", "ATTEMPTS",
			"CERTIFICATION", "FOCUS", "COST", "LIBRARY"} {
			if !strings.Contains(plain, title) {
				t.Fatalf("%v: missing insight panel %s", size, title)
			}
		}
		if strings.Contains(plain, "TIMELINE") {
			t.Fatalf("%v: the main page is still drawn", size)
		}
	}
}

func TestInsightContentKeysAndPageToggle(t *testing.T) {
	m, sent := withInsights(t, 200, 50)
	m.Update(key("v"))
	if m.focus != "performance" {
		t.Fatalf("insight page did not focus its first panel: %s", m.focus)
	}
	plain := ansi.Strip(m.Render())
	for _, want := range []string{"test cases measured", "133", "median ×1.43", "dense", "3.6%", "13 / 0",
		"reader error: OSError: boom", "waiting for the data process"} {
		if !strings.Contains(plain, want) {
			t.Errorf("insight page lacks %q", want)
		}
	}
	// The second summary line carries an inspect key for its source.
	m.cursor["performance"] = 1
	m.Update(key("enter"))
	last := (*sent)[len(*sent)-1]
	if last["cmd"] != "inspect" || last["kind"] != "performance" || last["key"] != "model_gap" {
		t.Fatalf("enter did not inspect the line's key: %v", last)
	}
	m.Update(key("esc"))
	m.Update(key("tab"))
	if m.focus != "cases" {
		t.Fatalf("tab on the insight page went to %s", m.focus)
	}
	m.Update(key("v"))
	if m.focus != "progress" || !strings.Contains(ansi.Strip(m.Render()), "TIMELINE") {
		t.Fatalf("v did not return to the main page (focus %s)", m.focus)
	}
	m.Update(key("v"))
	if m.focus != "cases" {
		t.Fatalf("the insight page did not keep its focus: %s", m.focus)
	}
}

func TestInsightSortsCycleTheTableOrder(t *testing.T) {
	m, sent := withInsights(t, 200, 50)
	m.Update(key("v"))
	m.Update(key("tab"))
	if m.focus != "cases" {
		t.Fatalf("focus %s", m.focus)
	}
	plain := ansi.Strip(m.Render())
	if !strings.Contains(plain, "s: worst gap") || strings.Index(plain, "case-big") > strings.Index(plain, "case-failing") {
		t.Fatalf("first sort not applied:\n%s", plain)
	}
	m.Update(key("s"))
	plain = ansi.Strip(m.Render())
	if !strings.Contains(plain, "s: status") {
		t.Fatalf("s did not cycle the sort:\n%s", plain)
	}
	// Order [2, 0]: the failing case first, then the big one, then the row the order omits.
	failing, big, small := strings.Index(plain, "case-failing"), strings.Index(plain, "case-big"), strings.Index(plain, "case-small")
	if !(failing < big && big < small) {
		t.Fatalf("status order not applied: %d %d %d", failing, big, small)
	}
	// The cursor selects rows in the displayed order: summary line, table header, then the first row.
	m.cursor["cases"] = 2
	m.Update(key("enter"))
	last := (*sent)[len(*sent)-1]
	if last["cmd"] != "inspect" || last["kind"] != "cases" || last["key"] != "case-failing" {
		t.Fatalf("enter did not inspect the displayed row: %v", last)
	}
	m.Update(key("s"))
	if !strings.Contains(ansi.Strip(m.Render()), "s: worst gap") {
		t.Fatalf("the sort did not wrap around")
	}
}

func TestCoreCellsCarryShortGapsButNotLongOnes(t *testing.T) {
	// Samples every third 2 s bucket (a throttled sampler), then a long silence.
	codes := "a  a  a" + strings.Repeat(" ", 20)
	util := "5  5  5" + strings.Repeat(" ", 20)
	cells, _ := coreCells(codes, util, 0, 2, 0, float64(2*len(codes)), len(codes))
	for i := 0; i < 7; i++ {
		if cells[i] != 'a' {
			t.Fatalf("cell %d not carried across a short gap: %q", i, cells[i])
		}
	}
	if cells[len(cells)-1] != 0 {
		t.Fatalf("a long silence must stay empty, got %q", cells[len(cells)-1])
	}
}
