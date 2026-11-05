// Package proto decodes the console data process's JSON lines (merlin.console.snapshot.v1).
//
// The Python process (merlin_experiments.tracking.console.serve) owns every record read; this
// package only mirrors the section contract documented in
// merlin_experiments/tracking/console/sources/__init__.py. Numbers that were not recorded arrive as
// JSON null and stay nil here: a missing value is never a zero.
package proto

import (
	"encoding/json"
	"fmt"
)

// Envelope is the discriminator shared by every message.
type Envelope struct {
	Type string `json:"type"`
}

// Hello is the first message of a data process.
type Hello struct {
	Schema          string `json:"schema"`
	OperatorPrivate bool   `json:"operator_private"`
	PID             int    `json:"pid"`
}

// Owner is the managed launch owner of an experiment, when one exists.
type Owner struct {
	Path      string `json:"path"`
	PID       *int   `json:"pid"`
	Alive     bool   `json:"alive"`
	Selection string `json:"selection"`
}

// Experiment is one discovered run with its derived state and the record that decided it.
type Experiment struct {
	ID           string   `json:"id"`
	RunID        string   `json:"run_id"`
	Kind         string   `json:"kind"`
	Task         string   `json:"task"`
	TaskName     string   `json:"task_name"`
	RunDir       string   `json:"run_dir"`
	Workspace    string   `json:"workspace"`
	Target       string   `json:"target"`
	Suite        string   `json:"suite"`
	LaunchDir    string   `json:"launch_dir"`
	Owner        *Owner   `json:"owner"`
	RunnerPID    *int     `json:"runner_pid"`
	StartedAt    *float64 `json:"started_at"`
	LastRecordAt *float64 `json:"last_record_at"`
	State        string   `json:"state"`
	StateBasis   string   `json:"state_basis"`
}

// Experiments is the discovery listing.
type Experiments struct {
	Items     []Experiment `json:"items"`
	Selected  string       `json:"selected"`
	Notices   []string     `json:"notices"`
	ScannedAt float64      `json:"scanned_at"`
}

// Tokens is a token account; any field may be unrecorded.
type Tokens struct {
	Input      *float64 `json:"input"`
	FreshInput *float64 `json:"fresh_input"`
	CacheRead  *float64 `json:"cache_read"`
	CacheWrite *float64 `json:"cache_write"`
	Output     *float64 `json:"output"`
	Reasoning  *float64 `json:"reasoning"`
	Total      *float64 `json:"total"`
}

type Core struct {
	CPU     int      `json:"cpu"`
	UtilPct *float64 `json:"util_pct"`
	Kind    string   `json:"kind"`
	Label   string   `json:"label"`
}

type Resources struct {
	Mode      string   `json:"mode"`
	Cores     []Core   `json:"cores"`
	BusyCPUs  *float64 `json:"busy_cpus"`
	TotalCPUs *float64 `json:"total_cpus"`
	Queue     struct {
		Waiters    *float64 `json:"waiters"`
		Leases     *float64 `json:"leases"`
		GsimActive *float64 `json:"gsim_active"`
	} `json:"queue"`
	Process struct {
		CPUPct    *float64 `json:"cpu_pct"`
		RSSBytes  *float64 `json:"rss_bytes"`
		Threads   *float64 `json:"threads"`
		Processes *float64 `json:"processes"`
		Load1     *float64 `json:"load1"`
		IO        *struct {
			ReadBps  *float64 `json:"read_bps"`
			WriteBps *float64 `json:"write_bps"`
		} `json:"io"`
	} `json:"process"`
	Series map[string][]*float64 `json:"series"`
	Notes  []string              `json:"notes"`
	Error  string                `json:"error"`
}

type Agent struct {
	ID             string   `json:"id"`
	ParentID       string   `json:"parent_id"`
	Name           string   `json:"name"`
	Path           string   `json:"path"`
	Role           string   `json:"role"`
	Depth          *float64 `json:"depth"`
	State          string   `json:"state"`
	StartedAt      *float64 `json:"started_at"`
	LastActivityAt *float64 `json:"last_activity_at"`
	ActiveS        *float64 `json:"active_s"`
	Turns          *float64 `json:"turns"`
	Commands       *float64 `json:"commands"`
	FailedCommands *float64 `json:"failed_commands"`
	Current        string   `json:"current"`
	LastMessage    string   `json:"last_message"`
	Model          string   `json:"model"`
	Effort         string   `json:"effort"`
	Tokens         Tokens   `json:"tokens"`
	Spans          [][]any  `json:"spans"`
	Stage          string   `json:"stage"`
	Round          *float64 `json:"round"`
}

type Agents struct {
	Agents []Agent `json:"agents"`
	Totals struct {
		Agents        *float64 `json:"agents"`
		Working       *float64 `json:"working"`
		Subagents     *float64 `json:"subagents"`
		HelperCeiling *float64 `json:"helper_ceiling"`
	} `json:"totals"`
	Source  string `json:"source"`
	Session *struct {
		Running *bool    `json:"running"`
		Phase   string   `json:"phase"`
		Since   *float64 `json:"since"`
		Detail  string   `json:"detail"`
	} `json:"session"`
	Notes []string `json:"notes"`
	Error string   `json:"error"`
}

type TokenSection struct {
	Total    Tokens                `json:"total"`
	Series   map[string][]*float64 `json:"series"`
	PerAgent []struct {
		ID     string   `json:"id"`
		Name   string   `json:"name"`
		Total  *float64 `json:"total"`
		Output *float64 `json:"output"`
		Share  *float64 `json:"share"`
	} `json:"per_agent"`
	RatePerMin struct {
		Input  *float64 `json:"input"`
		Output *float64 `json:"output"`
	} `json:"rate_per_min"`
	CacheHitRatio *float64 `json:"cache_hit_ratio"`
	Budget        *struct {
		Total   *float64 `json:"total"`
		Working *float64 `json:"working"`
		Reserve *float64 `json:"reserve"`
		UsedPct *float64 `json:"used_pct"`
	} `json:"budget"`
	RateLimit *struct {
		UsedPercent   *float64 `json:"used_percent"`
		WindowMinutes *float64 `json:"window_minutes"`
		ResetsAt      *float64 `json:"resets_at"`
	} `json:"rate_limit"`
	ContextWindow *float64 `json:"context_window"`
	Notes         []string `json:"notes"`
	Error         string   `json:"error"`
}

type Span struct {
	Lane    string   `json:"lane"`
	Label   string   `json:"label"`
	Capsule string   `json:"capsule"`
	Start   *float64 `json:"start"`
	End     *float64 `json:"end"`
	Status  string   `json:"status"`
	Source  string   `json:"source"`
}

type StageStat struct {
	Stage   string   `json:"stage"`
	Count   *float64 `json:"count"`
	P50     *float64 `json:"p50_s"`
	P95     *float64 `json:"p95_s"`
	Max     *float64 `json:"max_s"`
	Running *float64 `json:"running"`
}

type Jobs struct {
	Spans  []Span      `json:"spans"`
	Stages []StageStat `json:"stages"`
	Sim    struct {
		Requests     *float64 `json:"requests"`
		Queued       *float64 `json:"queued"`
		Running      *float64 `json:"running"`
		Done         *float64 `json:"done"`
		QueueWaitP50 *float64 `json:"queue_wait_p50"`
		QueueWaitP95 *float64 `json:"queue_wait_p95"`
		RuntimeP50   *float64 `json:"runtime_p50"`
		RuntimeP95   *float64 `json:"runtime_p95"`
	} `json:"sim"`
	Running []struct {
		Stage   string   `json:"stage"`
		Capsule string   `json:"capsule"`
		AgeS    *float64 `json:"age_s"`
		Tier    string   `json:"tier"`
	} `json:"running"`
	Notes []string `json:"notes"`
	Error string   `json:"error"`
}

type Event struct {
	T        *float64 `json:"t"`
	Kind     string   `json:"kind"`
	Severity string   `json:"severity"`
	Agent    string   `json:"agent"`
	Text     string   `json:"text"`
	Source   string   `json:"source"`
}

type Events struct {
	Events []Event  `json:"events"`
	Notes  []string `json:"notes"`
	Error  string   `json:"error"`
}

type Capsule struct {
	Name             string              `json:"name"`
	Stale            bool                `json:"stale"` // its newest result graded an older compiler
	Cohort           string              `json:"cohort"`
	Kind             string              `json:"kind"`
	Op               string              `json:"op"`
	Attrs            string              `json:"attrs"`
	Shapes           []string            `json:"shapes"`
	Dtype            string              `json:"dtype"`
	SourceRole       string              `json:"source_role"`
	Origin           map[string]any      `json:"origin"`
	Family           string              `json:"family"`
	Axis             string              `json:"axis"`
	PassRequirements []string            `json:"pass_requirements"`
	RequiredTiers    []string            `json:"required_tiers"`
	Tiers            map[string]string   `json:"tiers"`
	Status           string              `json:"status"`
	Numeric          string              `json:"numeric"`
	Cycles           map[string]*float64 `json:"cycles"`
	Failure          map[string]string   `json:"failure"`
	Timing           map[string]*float64 `json:"timing"`
	Path             string              `json:"path"`
}

type Capsules struct {
	Capsules []Capsule                     `json:"capsules"`
	Counts   map[string]map[string]float64 `json:"counts"`
	Notes    []string                      `json:"notes"`
	Error    string                        `json:"error"`
}

type Cohort struct {
	Name         string   `json:"name"`
	Total        *float64 `json:"total"`
	All          *float64 `json:"all"`
	L2           *float64 `json:"L2"`
	L3           *float64 `json:"L3"`
	Refusal      *float64 `json:"refusal"`
	L3Obligation string   `json:"L3_obligation"`
}

// Grade is one background grade of an immutable snapshot of the agent's compiler.
type Grade struct {
	Token      string   `json:"token"`
	State      string   `json:"state"` // grading | unfinished | done | superseded | stopped
	OwnerState string   `json:"owner_state"`
	Reason     string   `json:"reason"`
	StartedAt  *float64 `json:"started_at"`
	UpdatedAt  *float64 `json:"updated_at"`
	NSelected  *float64 `json:"n_selected"`
	NCompleted *float64 `json:"n_completed"`
	NPassed    *float64 `json:"n_passed"`
	NFailed    *float64 `json:"n_failed"`
	NErrors    *float64 `json:"n_errors"`
	NL2        *float64 `json:"n_l2_awaiting_l3"` // passed Spike L2, waiting for an L3 certificate
	Source     string   `json:"source"`
}

type Progress struct {
	Grades      []Grade  `json:"grades"`
	GradesTotal *float64 `json:"grades_total"`
	Published   *struct {
		UpdatedAt  *float64 `json:"updated_at"`
		NSelected  *float64 `json:"n_selected"`
		NCompleted *float64 `json:"n_completed"`
		NPassed    *float64 `json:"n_passed"`
	} `json:"published"`
	Certificates *struct {
		L2       map[string]float64 `json:"L2"`
		L3       map[string]float64 `json:"L3"`
		Physical *struct {
			Required *float64 `json:"required"`
			Passed   *float64 `json:"passed"`
			Failed   *float64 `json:"failed"`
			Pending  *float64 `json:"pending"`
		} `json:"physical"`
	} `json:"certificates"`
	SeriesScope string `json:"series_scope"`
	Verdict     *struct {
		Stage       string   `json:"stage"`
		TiersGraded []string `json:"tiers_graded"`
		TiersNotRun []string `json:"tiers_not_run"`
		AllPass     *bool    `json:"all_pass"`
	} `json:"verdict"`
	State            string   `json:"state"`
	NSelected        *float64 `json:"n_selected"`
	NCompleted       *float64 `json:"n_completed"`
	NPassed          *float64 `json:"n_passed"`
	NExecutionPassed *float64 `json:"n_execution_passed"`
	SnapshotCurrent  *bool    `json:"snapshot_current"`
	GradedSource     string   `json:"graded_source"`
	UpdatedAt        *float64 `json:"updated_at"`
	Cohorts          []Cohort `json:"cohorts"`
	Numeric          struct {
		Completed *float64 `json:"completed"`
		Passed    *float64 `json:"passed"`
		Failed    *float64 `json:"failed"`
	} `json:"numeric"`
	Series     map[string][]*float64 `json:"series"`
	Completion *struct {
		Reason         string   `json:"reason"`
		Converged      *bool    `json:"converged"`
		FormalComplete *bool    `json:"formal_complete"`
		ElapsedS       *float64 `json:"elapsed_s"`
	} `json:"completion"`
	Plateau       map[string]any `json:"plateau"`
	LatestRequest *struct {
		Source     string   `json:"source"`
		QueueWaitS *float64 `json:"queue_wait_s"`
		WallS      *float64 `json:"wall_s"`
		Stage      string   `json:"stage"`
	} `json:"latest_request"`
	Notes []string `json:"notes"`
	Error string   `json:"error"`
}

type Resource struct {
	Name        string   `json:"name"`
	ActivityPct *float64 `json:"activity_pct"`
}

type PerfRow struct {
	Capsule              string     `json:"capsule"`
	Cohort               string     `json:"cohort"`
	SpikeModelCycles     *float64   `json:"spike_model_cycles"`
	GsimCycles           *float64   `json:"gsim_cycles"`
	L2Cycles             *float64   `json:"l2_cycles"`
	FloorCycles          *float64   `json:"floor_cycles"`
	EfficiencyPct        *float64   `json:"efficiency_pct"`
	UsefulMacs           *float64   `json:"useful_macs"`
	PeakMacsPerCycle     *float64   `json:"peak_macs_per_cycle"`
	RequestedBytes       *float64   `json:"requested_bytes"`
	Intensity            *float64   `json:"intensity"`
	AchievedMacsPerCycle *float64   `json:"achieved_macs_per_cycle"`
	ModelStatus          string     `json:"model_status"`
	TimingMode           string     `json:"timing_mode"`
	TopResource          *Resource  `json:"top_resource"`
	Resources            []Resource `json:"resources"`
	DivergencePct        *float64   `json:"divergence_pct"`
	Comparable           bool       `json:"comparable"`
	Basis                string     `json:"basis"`
	Source               string     `json:"source"`
}

type Perf struct {
	Rows    []PerfRow `json:"rows"`
	Summary struct {
		Modeled     *float64 `json:"modeled"`
		Provisional *float64 `json:"provisional"`
		Unknown     *float64 `json:"unknown"`
		BestEff     *float64 `json:"best_eff"`
		WorstEff    *float64 `json:"worst_eff"`
		GeoEff      *float64 `json:"geo_eff"`
	} `json:"summary"`
	PeakMacsPerCycle *float64 `json:"peak_macs_per_cycle"`
	Notes            []string `json:"notes"`
	Error            string   `json:"error"`
}

type Golden struct {
	References []struct {
		Name            string   `json:"name"`
		Model           string   `json:"model"`
		ProgramIdentity string   `json:"program_identity"`
		Device          string   `json:"device"`
		Cycles          *float64 `json:"cycles"`
		Source          string   `json:"source"`
	} `json:"references"`
	Comparisons []struct {
		Subject         string   `json:"subject"`
		CandidateCycles *float64 `json:"candidate_cycles"`
		ReferenceCycles *float64 `json:"reference_cycles"`
		Ratio           *float64 `json:"ratio"`
		Comparable      bool     `json:"comparable"`
		Reason          string   `json:"reason"`
	} `json:"comparisons"`
	Notes []string `json:"notes"`
	Error string   `json:"error"`
}

type Pass struct {
	Name         string   `json:"name"`
	Kind         string   `json:"kind"`
	Declared     bool     `json:"declared"`
	Used         bool     `json:"used"`
	Status       string   `json:"status"`
	Boundary     string   `json:"boundary"`
	RequiredBy   *float64 `json:"required_by"`
	ExercisedBy  *float64 `json:"exercised_by"`
	Churn        *float64 `json:"churn"`
	ProfileWallS *float64 `json:"profile_wall_s"`
	Source       string   `json:"source"`
}

// AppliedPass is a pass the compiler runs (pipeline order for manifest commands).
type AppliedPass struct {
	Pass
	Order *float64 `json:"order"`
}

type Passes struct {
	Passes   []Pass              `json:"passes"`
	Applied  []AppliedPass       `json:"applied"`
	Counts   map[string]*float64 `json:"counts"`
	Pipeline []string            `json:"pipeline"`
	Notes    []string            `json:"notes"`
	Error    string              `json:"error"`
}

type Stage struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	State   string `json:"state"`
	Basis   string `json:"basis"`
	Actions struct {
		Done    *float64 `json:"done"`
		Failed  *float64 `json:"failed"`
		Running *float64 `json:"running"`
		Total   *float64 `json:"total"`
	} `json:"actions"`
	Capsules struct {
		Required *float64 `json:"required"`
		Passing  *float64 `json:"passing"`
	} `json:"capsules"`
	WallS   *float64       `json:"wall_s"`
	Tokens  *float64       `json:"tokens"`
	Metrics map[string]any `json:"metrics"`
	Items   []string       `json:"items"`
}

type Stages struct {
	Stages   []Stage  `json:"stages"`
	Contract string   `json:"contract"`
	Notes    []string `json:"notes"`
	Error    string   `json:"error"`
}

type Finding struct {
	Code        string   `json:"code"`
	Severity    string   `json:"severity"`
	Scope       string   `json:"scope"`
	Title       string   `json:"title"`
	Detail      string   `json:"detail"`
	Evidence    *string  `json:"evidence"`
	Occurrences int      `json:"occurrences"`
	Items       []string `json:"items"`
	Action      string   `json:"action"`
	ImpactS     *float64 `json:"impact_s"`
}

// TurnStage is one author-startup stage of a turn.
type TurnStage struct {
	Stage string   `json:"stage"`
	Start *float64 `json:"start"`
	End   *float64 `json:"end"`
	WallS *float64 `json:"wall_s"`
	Depth int      `json:"depth"`
}

// Turn is one author turn: admission, startup, agent work and how it ended (sources/lifecycle.py).
type Turn struct {
	Round        *float64    `json:"round"`
	Turn         *float64    `json:"turn"`
	Phase        string      `json:"phase"`
	AdmittedAt   *float64    `json:"admitted_at"`
	Allowance    *float64    `json:"allowance"`
	StartupStart *float64    `json:"startup_start"`
	StartupEnd   *float64    `json:"startup_end"`
	StartupS     *float64    `json:"startup_s"`
	Stages       []TurnStage `json:"stages"`
	OpenStage    *struct {
		Stage string  `json:"stage"`
		Since float64 `json:"since"`
	} `json:"open_stage"`
	AgentStart *float64 `json:"agent_start"`
	AgentEnd   *float64 `json:"agent_end"`
	AgentS     *float64 `json:"agent_s"`
	EndReason  string   `json:"end_reason"`
}

// Lifecycle is the author session's state (sources/lifecycle.py).
type Lifecycle struct {
	Phase         string   `json:"phase"`
	PhaseSince    *float64 `json:"phase_since"`
	PhaseDetail   string   `json:"phase_detail"`
	AuthorRunning *bool    `json:"author_running"`
	Turns         []Turn   `json:"turns"`
	Notes         []string `json:"notes"`
	Error         string   `json:"error"`
}

// ActivityCategory is one per-core activity with its one-character code.
type ActivityCategory struct {
	ID    string `json:"id"`
	Code  string `json:"code"`
	Title string `json:"title"`
}

// Activity is what each of the experiment's cores works on over time (sources/activity.py).
type Activity struct {
	Mode       string             `json:"mode"`
	Categories []ActivityCategory `json:"categories"`
	CPUs       []int              `json:"cpus"`
	History    *struct {
		T0    float64 `json:"t0"`
		DT    float64 `json:"dt"`
		N     int     `json:"n"`
		Since float64 `json:"since"`
		Cores []struct {
			CPU   int    `json:"cpu"`
			Codes string `json:"codes"`
			Util  string `json:"util"`
		} `json:"cores"`
	} `json:"history"`
	Now []struct {
		CPU     int      `json:"cpu"`
		UtilPct *float64 `json:"util_pct"`
		Code    string   `json:"code"`
		Top     []struct {
			Activity string  `json:"activity"`
			Label    string  `json:"label"`
			Pct      float64 `json:"pct"`
		} `json:"top"`
	} `json:"now"`
	Totals map[string]float64 `json:"totals"`
	Notes  []string           `json:"notes"`
	Error  string             `json:"error"`
}

// Timeline is the wall-time attribution of the experiment (sources/timeline.py).
type Timeline struct {
	Window *struct {
		Start   float64 `json:"start"`
		End     float64 `json:"end"`
		Seconds float64 `json:"seconds"`
	} `json:"window"`
	Lead *struct {
		ID     string  `json:"id"`
		ThinkS float64 `json:"think_s"`
		ToolS  float64 `json:"tool_s"`
		WaitS  float64 `json:"wait_s"`
		SetupS float64 `json:"setup_s"`
		IdleS  float64 `json:"idle_s"`
		Basis  string  `json:"basis"`
	} `json:"lead"`
	Recorded *struct {
		Method          string   `json:"method"`
		ThinkS          float64  `json:"think_s"`
		ToolWaitS       float64  `json:"tool_wait_s"`
		BetweenSessionS *float64 `json:"between_session_s"`
		MeasuredSpanS   *float64 `json:"measured_span_s"`
		ThinkPct        *float64 `json:"think_pct"`
		OutputPerThinkS *float64 `json:"output_tokens_per_think_s"`
		Source          string   `json:"source"`
	} `json:"recorded"`
	Overlap struct {
		SimBusyS          *float64 `json:"sim_busy_s"`
		QueueS            *float64 `json:"queue_s"`
		LeadBlockedOnSimS *float64 `json:"lead_blocked_on_sim_s"`
		SimWhileThinkS    *float64 `json:"sim_while_think_s"`
	} `json:"overlap"`
	Parallelism struct {
		AgentsAvg     *float64 `json:"agents_avg"`
		AgentsPeak    *float64 `json:"agents_peak"`
		HelpersUsed   *float64 `json:"helpers_used"`
		HelperCeiling *float64 `json:"helper_ceiling"`
		JobsAvg       *float64 `json:"jobs_avg"`
		JobsPeak      *float64 `json:"jobs_peak"`
	} `json:"parallelism"`
	Concurrency *struct {
		T0     float64   `json:"t0"`
		DT     float64   `json:"dt"`
		Agents []float64 `json:"agents"`
		Sim    []float64 `json:"sim"`
		Jobs   []float64 `json:"jobs"`
	} `json:"concurrency"`
	Markers []struct {
		T     float64 `json:"t"`
		Kind  string  `json:"kind"`
		Label string  `json:"label"`
	} `json:"markers"`
	Verdict *struct {
		Bound       string  `json:"bound"`
		SharePct    float64 `json:"share_pct"`
		Explanation string  `json:"explanation"`
		Basis       string  `json:"basis"`
	} `json:"verdict"`
	Notes []string `json:"notes"`
	Error string   `json:"error"`
}

// MachineCore is one online CPU with its holders.
type MachineCore struct {
	CPU     int      `json:"cpu"`
	UtilPct *float64 `json:"util_pct"`
	Holders []string `json:"holders"`
	Mine    bool     `json:"mine"`
	Console bool     `json:"console"`
}

// Machine is every CPU and its allocation (sources/machine.py).
type Machine struct {
	Cores       []MachineCore `json:"cores"`
	Allocations []struct {
		Name      string   `json:"name"`
		Kind      string   `json:"kind"`
		CPUs      string   `json:"cpus"`
		Partition string   `json:"partition"`
		Processes *float64 `json:"processes"`
		PID       *int     `json:"pid"`
	} `json:"allocations"`
	BusyCPUs       *float64 `json:"busy_cpus"`
	TotalCPUs      *float64 `json:"total_cpus"`
	ExperimentCPUs []int    `json:"experiment_cpus"`
	ConsoleCPUs    []int    `json:"console_cpus"`
	Volume         *Volume  `json:"volume"`
	Notes          []string `json:"notes"`
	Error          string   `json:"error"`
}

// AgentFile is one agent-visible feedback file (sources/agent_view.py).
type AgentFile struct {
	Present         bool     `json:"present"`
	At              *float64 `json:"at"`
	Rows            *float64 `json:"rows"`
	SnapshotCurrent *bool    `json:"snapshot_current"`
	GradedSource    string   `json:"graded_source"`
	GradedAt        *float64 `json:"graded_at"`
	Notices         []string `json:"notices"`
	ModelGateOnly   bool     `json:"model_gate_only"`
	Headroom        *struct {
		Ranked    *float64 `json:"ranked"`
		PassingL2 *float64 `json:"passing_l2"`
		MedianPct *float64 `json:"median_utilization_pct"`
	} `json:"headroom"`
}

// AgentView is what the author agent can see of its own feedback (sources/agent_view.py).
type AgentView struct {
	Workspace       string    `json:"workspace"`
	Live            bool      `json:"live"`
	Verdict         AgentFile `json:"verdict"`
	LatestCompleted AgentFile `json:"latest_completed"`
	Progress        AgentFile `json:"progress"`
	Suggestions     struct {
		Present     bool     `json:"present"`
		Suggestions *float64 `json:"suggestions"`
		Pending     *float64 `json:"pending_workload_check"`
		Discovery   *float64 `json:"discovery"`
	} `json:"suggestions"`
	Selfchecks struct {
		Requests        *float64 `json:"requests"`
		Answered        *float64 `json:"answered"`
		Pending         *float64 `json:"pending"`
		OldestPendingAt *float64 `json:"oldest_pending_at"`
		LastResponseAt  *float64 `json:"last_response_at"`
		RecentErrors    *float64 `json:"recent_errors"`
		LastError       string   `json:"last_error"`
	} `json:"selfchecks"`
	NewestGrade *struct {
		Token        string   `json:"token"`
		At           *float64 `json:"at"`
		Rows         *float64 `json:"rows"`
		GradedSource string   `json:"graded_source"`
		Delivered    []string `json:"delivered"`
	} `json:"newest_grade"`
	Notes []string `json:"notes"`
	Error string   `json:"error"`
}

// Volume is the free space of the filesystem holding the run directory (sources/machine.py).
type Volume struct {
	Path       string   `json:"path"`
	FreeBytes  *float64 `json:"free_bytes"`
	TotalBytes *float64 `json:"total_bytes"`
	FreePct    *float64 `json:"free_pct"`
	Level      string   `json:"level"`
}

// Console is the console's own cost and placement.
type Console struct {
	CPUPct    *float64 `json:"cpu_pct"`
	RSSBytes  *float64 `json:"rss_bytes"`
	CPUs      []int    `json:"cpus"`
	Avoided   []int    `json:"avoided"`
	Policy    string   `json:"policy"`
	Shared    bool     `json:"shared"`
	IntervalS *float64 `json:"interval_s"`
	Throttled bool     `json:"throttled"`
	BudgetPct *float64 `json:"budget_pct"`
	Error     string   `json:"error"`
}

// Snapshot is one experiment's state. Sections are decoded lazily: a delta message carries only the
// sections that changed, and Merge keeps the rest.
type Snapshot struct {
	Full            bool                       `json:"full"`
	GeneratedAt     float64                    `json:"generated_at"`
	Experiment      Experiment                 `json:"experiment"`
	OperatorPrivate bool                       `json:"operator_private"`
	Sections        map[string]json.RawMessage `json:"sections"`
	Changed         []string                   `json:"changed"`
	Timings         map[string]float64         `json:"timings"`
	Diagnostics     []Finding                  `json:"diagnostics"`
	Notices         []string                   `json:"notices"`
	Console         *Console                   `json:"console"`
}

// Inspect is the drill-down record for one row.
type Inspect struct {
	RequestID int    `json:"request_id"`
	Kind      string `json:"kind"`
	Key       string `json:"key"`
	Title     string `json:"title"`
	Sections  []struct {
		Title string   `json:"title"`
		Lines []string `json:"lines"`
	} `json:"sections"`
	Paths  []string `json:"paths"`
	Notice string   `json:"notice"`
	Error  string   `json:"error"`
}

// Error is a data-process complaint.
type Error struct {
	Message string `json:"message"`
}

// InsightNames are the insight panels (merlin_experiments.tracking.console.model.INSIGHTS), in display
// order. Every one shares the generic Insight shape below.
var InsightNames = []string{"performance", "cases", "tools", "grader", "integrity", "attempts", "certification", "focus", "cost", "library"}

// InsightLine is one summary line of an insight panel: display text, an optional level
// (ok|info|warn|bad) and the inspect key of the row it summarises.
type InsightLine struct {
	Label string `json:"label"`
	Value string `json:"value"`
	Level string `json:"level"`
	Key   string `json:"key"`
}

// InsightRow is one table row of an insight panel; cells are display text ("—" when not recorded).
type InsightRow struct {
	Cells []string `json:"cells"`
	Level string   `json:"level"`
	Key   string   `json:"key"`
}

// InsightTable is the optional table of an insight panel.
type InsightTable struct {
	Columns []string     `json:"columns"`
	Rows    []InsightRow `json:"rows"`
}

// InsightSeries is one downsampled time series of an insight panel.
type InsightSeries struct {
	Label string    `json:"label"`
	T     []float64 `json:"t"`
	V     []float64 `json:"v"`
}

// InsightSort is one row order a panel offers for its table: Order lists table row indices, first
// shown first. The frontend's sort key (s) cycles a panel's sorts.
type InsightSort struct {
	Label string `json:"label"`
	Order []int  `json:"order"`
}

// Insight is the generic panel shape of the insight sources (sources/insight_common.py). Their
// source-specific fields are not decoded here, except the optional table orders (Sorts).
type Insight struct {
	Title   string          `json:"title"`
	Summary []InsightLine   `json:"summary"`
	Table   *InsightTable   `json:"table"`
	Series  []InsightSeries `json:"series"`
	Notes   []string        `json:"notes"`
	Sorts   []InsightSort   `json:"sorts"`
	Error   string          `json:"error"`
}

// State is everything the frontend knows about the selected experiment, decoded per section.
type State struct {
	Experiment  Experiment
	GeneratedAt float64
	Operator    bool
	Timings     map[string]float64
	Diagnostics []Finding
	Notices     []string
	Resources   Resources
	Agents      Agents
	Tokens      TokenSection
	Jobs        Jobs
	Events      Events
	Capsules    Capsules
	Progress    Progress
	Perf        Perf
	Golden      Golden
	Passes      Passes
	Stages      Stages
	Timeline    Timeline
	Machine     Machine
	Lifecycle   Lifecycle
	Activity    Activity
	AgentView   AgentView
	Insights    map[string]Insight
	Console     Console
	Have        map[string]bool
	DecodeErr   map[string]string
}

// NewState is an empty state for one experiment.
func NewState() *State {
	return &State{Have: map[string]bool{}, DecodeErr: map[string]string{}, Timings: map[string]float64{},
		Insights: map[string]Insight{}}
}

// Merge applies a snapshot message: replace changed sections, keep the others.
func (s *State) Merge(snap *Snapshot) {
	if snap.Full || snap.Experiment.ID != s.Experiment.ID {
		*s = *NewState()
	}
	s.Experiment = snap.Experiment
	s.GeneratedAt = snap.GeneratedAt
	s.Operator = snap.OperatorPrivate
	s.Timings = snap.Timings
	s.Diagnostics = snap.Diagnostics
	s.Notices = snap.Notices
	if snap.Console != nil {
		s.Console = *snap.Console
	}
	targets := map[string]any{
		"resources": &s.Resources, "agents": &s.Agents, "tokens": &s.Tokens, "jobs": &s.Jobs,
		"events": &s.Events, "capsules": &s.Capsules, "progress": &s.Progress, "perf": &s.Perf,
		"golden": &s.Golden, "passes": &s.Passes, "stages": &s.Stages, "timeline": &s.Timeline,
		"machine": &s.Machine, "lifecycle": &s.Lifecycle, "activity": &s.Activity,
		"agent_view": &s.AgentView,
	}
	for _, name := range InsightNames {
		targets[name] = &Insight{}
	}
	for name, raw := range snap.Sections {
		target, ok := targets[name]
		if !ok {
			continue
		}
		if err := decodeInto(raw, target); err != nil {
			s.DecodeErr[name] = err.Error()
			continue
		}
		delete(s.DecodeErr, name)
		s.Have[name] = true
		if insight, ok := target.(*Insight); ok {
			s.Insights[name] = *insight
		}
	}
}

func decodeInto(raw json.RawMessage, target any) error {
	// Reset to zero first so a section that lost a field does not keep the stale value.
	switch t := target.(type) {
	case *Resources:
		*t = Resources{}
	case *Agents:
		*t = Agents{}
	case *TokenSection:
		*t = TokenSection{}
	case *Jobs:
		*t = Jobs{}
	case *Events:
		*t = Events{}
	case *Capsules:
		*t = Capsules{}
	case *Progress:
		*t = Progress{}
	case *Perf:
		*t = Perf{}
	case *Golden:
		*t = Golden{}
	case *Passes:
		*t = Passes{}
	case *Stages:
		*t = Stages{}
	case *Timeline:
		*t = Timeline{}
	case *Machine:
		*t = Machine{}
	case *Lifecycle:
		*t = Lifecycle{}
	case *Activity:
		*t = Activity{}
	case *AgentView:
		*t = AgentView{}
	case *Insight:
		*t = Insight{}
	default:
		return fmt.Errorf("unknown section target %T", target)
	}
	return json.Unmarshal(raw, target)
}

// Decode turns one JSON line into a typed message.
func Decode(line []byte) (any, error) {
	var env Envelope
	if err := json.Unmarshal(line, &env); err != nil {
		return nil, err
	}
	var msg any
	switch env.Type {
	case "hello":
		msg = &Hello{}
	case "experiments":
		msg = &Experiments{}
	case "snapshot":
		msg = &Snapshot{}
	case "inspect":
		msg = &Inspect{}
	case "error":
		msg = &Error{}
	default:
		return nil, fmt.Errorf("unknown message type %q", env.Type)
	}
	if err := json.Unmarshal(line, msg); err != nil {
		return nil, err
	}
	return msg, nil
}
