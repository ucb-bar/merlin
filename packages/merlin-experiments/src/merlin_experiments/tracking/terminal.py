"""Animated read-only experiment console: cores, jobs, agent tree and performance.

Run ``python -m merlin_experiments.tracking.terminal --task TASK_DIR``.
Sampling is independent of animation; no agents, graders or hardware jobs are
launched, and no campaign files are modified. No new terminal dependency is needed.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import select
import shutil
import sys
import threading
import time
from pathlib import Path

from . import terminal_views as V


class Feed:
    """Slow record reads never block keyboard input or animation."""

    def __init__(self, sampler, metadata, interval, health=None):
        self.sampler, self.metadata, self.interval = sampler, metadata, interval
        self.health = health
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.sample, self.details, self.error = {}, {}, None
        self.generation = 0
        self.thread = threading.Thread(target=self.run, name='terminal-read-only-sampler', daemon=True)

    def run(self):
        next_metadata = 0.0
        run_id = None
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                sample = self.sampler.sample()
                current_run = sample.get('campaign', {}).get('run_id')
                if current_run != run_id:
                    with self.lock:
                        self.details = {}
                    run_id, next_metadata = current_run, 0.0
                with self.lock:
                    self.sample, self.error = sample, None
                    self.generation += 1
                if started >= next_metadata:
                    details = self.metadata.sample(sample)
                    if self.health is not None:
                        details["progress_health"] = self.health.sample(sample)
                    with self.lock:
                        self.details = details
                    next_metadata = started + max(3.0, self.interval)
            except Exception as exc:
                with self.lock:
                    self.error = f'Read unavailable: {type(exc).__name__}: {V.clean(exc, 160)}'
            self.stop.wait(max(0.05, self.interval - (time.monotonic() - started)))

    def latest(self):
        with self.lock:
            return self.sample, self.details, self.error, self.generation


@contextlib.contextmanager
def terminal_mode():
    """Always restore the user's screen, cursor and input mode."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        sys.stdout.write('\x1b[?1049h\x1b[?25l\x1b[H')
        sys.stdout.flush()
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        sys.stdout.write('\x1b[0m\x1b[?25h\x1b[?1049l')
        sys.stdout.flush()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--task', type=Path, required=True, help='Managed campaign task directory; follow its latest native owner')
    result.add_argument('--selection', type=Path, help='Optional exact selection instead of following the latest launch')
    result.add_argument('--operator-private', action='store_true', help='Show operator capsule, agent and performance details; never sent to the author loop')
    result.add_argument('--interval', type=float, default=1.0, help='Resource sample interval in seconds (default: 1)')
    result.add_argument('--window', type=int, default=120, help='Seconds displayed in the core timeline (default: 120)')
    result.add_argument('--view', choices=['overview', *[v.lower() for v in V.TABS]], default='overview', help='Unified overview by default; optional focused plain-text reports')
    result.add_argument('--once', action='store_true', help='Print one sampled frame and exit')
    result.add_argument('--no-color', action='store_true')
    result.add_argument('--width', type=int, help='Override detected terminal columns')
    result.add_argument('--height', type=int, help='Override detected terminal rows')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0.2 <= args.interval <= 60:
        raise SystemExit('--interval must be between 0.2 and 60 seconds')
    if not 10 <= args.window <= 3600:
        raise SystemExit('--window must be between 10 and 3600 seconds')
    if args.width is not None and args.width < 45:
        raise SystemExit('--width must be at least 45 columns')
    if args.height is not None and args.height < 12:
        raise SystemExit('--height must be at least 12 rows')
    from .terminal_probe import Sampler
    from .terminal_metadata import Metadata
    from .progress_health import ProgressHealth

    sampler = Sampler(args.task, operator_private=args.operator_private, selection=args.selection)
    metadata = Metadata(args.task, operator_private=args.operator_private)
    health = ProgressHealth(args.task, operator_private=args.operator_private)
    history = V.History(args.window)
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.once
    color = not args.no_color and 'NO_COLOR' not in os.environ and sys.stdout.isatty()
    tab = -1 if args.view == 'overview' else V.TABS.index(args.view.upper())
    offset, frame, paused = 0, 0, False
    paused_history = None

    def draw(sample, details, error=None):
        size = shutil.get_terminal_size((120, 40))
        return V.render(sample, details, paused_history if paused else history, tab=tab, width=args.width or size.columns,
                        height=args.height or (size.lines if interactive else max(34, size.lines)),
                        color=color, frame=frame, paused=paused, offset=offset, error=error)

    if not interactive:
        sampler.sample()  # Establish counters; an unmeasured first sample is never zero utilization.
        time.sleep(args.interval)
        sample = sampler.sample()
        sample['operator_private'] = args.operator_private
        details = metadata.sample(sample)
        details["progress_health"] = health.sample(sample)
        history.add(sample)
        print(draw(sample, details))
        return 0

    feed = Feed(sampler, metadata, args.interval, health)
    feed.thread.start()
    shown, shown_details, last_generation, last_frame = {}, {}, -1, ''
    try:
        with terminal_mode() as fd:
            while True:
                sample, details, error, generation = feed.latest()
                if generation != last_generation:
                    sample = {**sample, 'operator_private': args.operator_private}
                    history.add(sample)
                    last_generation = generation
                if not paused:
                    shown, shown_details = {**sample, 'operator_private': args.operator_private}, details
                text = draw(shown, shown_details, error)
                if text != last_frame:
                    # Only changed rows travel over SSH; never clear between animation frames.
                    rows, previous_rows = text.splitlines(), last_frame.splitlines()
                    changes = []
                    for index in range(max(len(rows), len(previous_rows))):
                        row = rows[index] if index < len(rows) else ''
                        previous = previous_rows[index] if index < len(previous_rows) else None
                        if row != previous:
                            changes.append(f'\x1b[{index + 1};1H' + row + '\x1b[K')
                    sys.stdout.write(''.join(changes))
                    sys.stdout.flush()
                    last_frame = text
                ready, _, _ = select.select([fd], [], [], 0.12)
                if ready:
                    key = os.read(fd, 32).decode(errors='ignore')
                    if 'q' in key or '\x03' in key:
                        break
                    if tab != -1 and key in ('1', '2', '3', '4', '5'):
                        tab, offset = int(key) - 1, 0
                    elif tab != -1 and key == '\t':
                        tab, offset = (tab + 1) % len(V.TABS), 0
                    elif key in ('j', '\x1b[B'):
                        offset = min(10000, offset + 1)
                    elif key in ('k', '\x1b[A'):
                        offset = max(0, offset - 1)
                    elif key == ' ':
                        paused = not paused
                        paused_history = copy.deepcopy(history) if paused else None
                frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop.set()
        feed.thread.join(timeout=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
