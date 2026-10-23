// Command merlin-console is the Bubble Tea frontend of the Merlin experiment command console.
//
// It spawns the read-only data process (python -m merlin_experiments.tracking.console.serve) and
// renders its JSON-line snapshots: experiments, progress, stages, bottlenecks, a timeline of agents and
// jobs, the agent tree, tokens, CPUs and queues, performance, capsules, passes and events in one view.
//
//	merlin-console [--python PY] [--operator-private] [--task DIR]... [--experiment RUN]
//	merlin-console --render-once 200x55 ...   print one frame (no TTY needed) and exit
//	merlin-console --replay FILE ...          render recorded JSON lines instead of a data process
//
// Everything after "--" is passed to the data process unchanged.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/ucb-bar/merlin/packages/merlin-experiments/console/internal/proto"
	"github.com/ucb-bar/merlin/packages/merlin-experiments/console/internal/ui"
)

type multi []string

func (m *multi) String() string     { return strings.Join(*m, ",") }
func (m *multi) Set(v string) error { *m = append(*m, v); return nil }

func main() {
	python := flag.String("python", envOr("MERLIN_CONSOLE_PYTHON", "python3"), "Python interpreter with merlin-experiments installed")
	operator := flag.Bool("operator-private", false, "Show capsule names, paths and golden references (operator only)")
	experiment := flag.String("experiment", "", "Open this experiment first (run id or run directory)")
	tasksRoot := flag.String("tasks-root", "", "Directory holding the task folders")
	interval := flag.Float64("interval", 1.0, "Fast refresh seconds for running experiments")
	renderOnce := flag.String("render-once", "", "Print one frame of WIDTHxHEIGHT after the first snapshot and exit")
	replay := flag.String("replay", "", "Render recorded data-process JSON lines from FILE")
	var tasks, runs multi
	flag.Var(&tasks, "task", "Limit discovery to this task folder (repeatable)")
	flag.Var(&runs, "run-dir", "Also show this run or record directory (repeatable)")
	flag.Parse()

	args := []string{"-m", "merlin_experiments.tracking.console.serve", "--interval", strconv.FormatFloat(*interval, 'f', -1, 64)}
	if *operator {
		args = append(args, "--operator-private")
	}
	if *experiment != "" {
		args = append(args, "--experiment", *experiment)
	}
	if *tasksRoot != "" {
		args = append(args, "--tasks-root", *tasksRoot)
	}
	for _, t := range tasks {
		args = append(args, "--task", t)
	}
	for _, r := range runs {
		args = append(args, "--run-dir", r)
	}
	args = append(args, flag.Args()...)

	if *renderOnce != "" {
		os.Exit(once(*python, args, *replay, *renderOnce))
	}
	os.Exit(interactive(*python, args, *replay))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// bridge runs the data process (or a replay file) and delivers decoded messages to deliver.
type bridge struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	mu     sync.Mutex
	stderr []string
}

func (b *bridge) send(cmd map[string]any) {
	if b.stdin == nil {
		return
	}
	raw, err := json.Marshal(cmd)
	if err != nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	_, _ = b.stdin.Write(append(raw, '\n'))
}

func (b *bridge) tail() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]string{}, b.stderr...)
}

func start(python string, args []string, replay string, deliver func(any), done func(error)) (*bridge, error) {
	b := &bridge{}
	var stdout io.Reader
	if replay != "" {
		f, err := os.Open(replay)
		if err != nil {
			return nil, err
		}
		stdout = f
	} else {
		b.cmd = exec.Command(python, args...)
		b.cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
		in, err := b.cmd.StdinPipe()
		if err != nil {
			return nil, err
		}
		out, err := b.cmd.StdoutPipe()
		if err != nil {
			return nil, err
		}
		errp, err := b.cmd.StderrPipe()
		if err != nil {
			return nil, err
		}
		if err := b.cmd.Start(); err != nil {
			return nil, fmt.Errorf("starting the data process (%s): %w", python, err)
		}
		b.stdin, stdout = in, out
		go func() {
			sc := bufio.NewScanner(errp)
			for sc.Scan() {
				b.mu.Lock()
				b.stderr = append(b.stderr, sc.Text())
				if len(b.stderr) > 40 {
					b.stderr = b.stderr[len(b.stderr)-40:]
				}
				b.mu.Unlock()
			}
		}()
	}
	go func() {
		rd := bufio.NewReaderSize(stdout, 1<<20)
		for {
			line, err := rd.ReadBytes('\n')
			if len(line) > 1 {
				if msg, derr := proto.Decode(line); derr == nil {
					deliver(msg)
				}
			}
			if err != nil {
				var werr error
				if b.cmd != nil {
					werr = b.cmd.Wait()
				}
				done(werr)
				return
			}
		}
	}()
	return b, nil
}

func interactive(python string, args []string, replay string) int {
	var prog *tea.Program
	ready := make(chan struct{})
	var b *bridge
	model := ui.New(func(cmd map[string]any) {
		if b != nil {
			b.send(cmd)
		}
	})
	prog = tea.NewProgram(model)
	var err error
	b, err = start(python, args, replay, func(msg any) {
		<-ready
		prog.Send(ui.LineMsg{Msg: msg})
	}, func(werr error) {
		<-ready
		text := ""
		if werr != nil {
			text = werr.Error()
		}
		if replay == "" {
			prog.Send(ui.CollectorExitMsg{Err: text, Stderr: b.tail()})
		}
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "merlin-console:", err)
		return 2
	}
	close(ready)
	if _, err := prog.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "merlin-console:", err)
		return 1
	}
	if b.cmd != nil && b.cmd.Process != nil {
		b.send(map[string]any{"cmd": "quit"})
		time.AfterFunc(2*time.Second, func() { _ = b.cmd.Process.Kill() })
	}
	return 0
}

// once prints a single frame after the first snapshot (plus the slow sections) arrives.
func once(python string, args []string, replay, size string) int {
	w, h := 200, 55
	if a, c, ok := strings.Cut(size, "x"); ok {
		if v, err := strconv.Atoi(a); err == nil {
			w = v
		}
		if v, err := strconv.Atoi(c); err == nil {
			h = v
		}
	}
	model := ui.New(nil)
	model.W, model.H = w, h
	msgs := make(chan any, 64)
	finished := make(chan error, 1)
	b, err := start(python, args, replay, func(msg any) { msgs <- msg }, func(e error) { finished <- e })
	if err != nil {
		fmt.Fprintln(os.Stderr, "merlin-console:", err)
		return 2
	}
	timeout := time.After(120 * time.Second)
	snapshots := 0
loop:
	for {
		select {
		case msg := <-msgs:
			model.Apply(msg)
			if snap, ok := msg.(*proto.Snapshot); ok {
				snapshots++
				state := snap.Experiment.State
				live := state == "LIVE" || state == "ACTIVE" || state == "STALLED"
				// A live run needs a second snapshot for CPU deltas; a stored run is whole in one.
				if replay == "" && (snapshots >= 2 || !live) {
					break loop
				}
			}
		case e := <-finished:
			for len(msgs) > 0 {
				model.Apply(<-msgs)
			}
			if snapshots == 0 {
				fmt.Fprintln(os.Stderr, "merlin-console: data process ended before a snapshot", e, strings.Join(b.tail(), "\n"))
				return 1
			}
			break loop
		case <-timeout:
			fmt.Fprintln(os.Stderr, "merlin-console: no snapshot within 120 s")
			break loop
		}
	}
	if b.cmd != nil && b.cmd.Process != nil {
		b.send(map[string]any{"cmd": "quit"})
		_ = b.cmd.Process.Kill()
	}
	fmt.Println(model.Render())
	return 0
}
