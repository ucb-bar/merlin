"""Color terminal views of sampled resources and recorded experiment evidence."""

from __future__ import annotations

import math
import time
import unicodedata
from collections import deque

from . import terminal_health

RESET = '\x1b[0m'
COLORS = {'agent': '38;5;213', 'L2': '38;5;45', 'L3': '38;5;82',
          'build': '38;5;214', 'verify': '38;5;141', 'calibration': '38;5;220',
          'host': '38;5;75', 'unknown': '38;5;244', 'idle': '38;5;238'}
GLYPHS = {'agent': 'A', 'L2': '2', 'L3': '3', 'build': 'B', 'verify': 'V',
          'calibration': 'C', 'host': 'H', 'unknown': '?', 'idle': '·'}
TABS = ('CORES', 'JOBS', 'AGENTS', 'PERF', 'HISTORY')


def clip_ansi(line, width):
    """Clip display columns without cutting an ANSI color sequence in half."""
    result, used, index = [], 0, 0
    while index < len(line):
        if line[index:index + 2] == '\x1b[':
            end = index + 2
            while end < len(line) and not ('@' <= line[end] <= '~'):
                end += 1
            result.append(line[index:end + 1])
            index = end + 1
            continue
        char = line[index]
        size = 0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1)
        if used + size > width:
            break
        result.append(char)
        used += size
        index += 1
    return ''.join(result) + (RESET if '\x1b[' in line else '')


def visible_width(line):
    index, width = 0, 0
    while index < len(line):
        if line[index:index + 2] == '\x1b[':
            index += 2
            while index < len(line) and not ('@' <= line[index] <= '~'):
                index += 1
        else:
            char = line[index]
            width += 0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1)
        index += 1
    return width


def pad(line, width):
    line = clip_ansi(line, width)
    return line + ' ' * max(0, width - visible_width(line))


def clean(value, limit=200):
    """Records cannot inject terminal escapes, newlines or unbounded strings."""
    return ''.join(c if c.isprintable() and ord(c) < 127 else ' ' for c in str(value or ''))[:limit]


def number(value, digits=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return '—'
    return f'{value:,.{digits}f}'


def paint(value, code, color=True):
    return f'\x1b[{code}m{value}{RESET}' if color else str(value)


def kind(action):
    value = str(action or '').lower()
    # Resource labels are display categories only; they grant no execution verdict.
    if 'l3' in value or 'gsim' in value or 'rtl_sim' in value:
        return 'L3'
    if 'l2' in value or 'spike' in value:
        return 'L2'
    if 'calibr' in value:
        return 'calibration'
    if any(token in value for token in ('agent', 'codex', 'claude')):
        return 'agent'
    if any(token in value for token in ('verif', 'policy', 'snapshot', 'preflight')):
        return 'verify'
    if any(token in value for token in ('build', 'compile', 'emit', 'clang', 'link')):
        return 'build'
    if 'host' in value:
        return 'host'
    return 'unknown'


def elapsed(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return '—'
    if value >= 3600:
        return f'{int(value // 3600)}h{int(value % 3600 // 60):02}m'
    return f'{int(value // 60)}m{int(value % 60):02}s' if value >= 60 else f'{value:.1f}s'


class History:
    """Bounded samples collected only while this display is open."""

    def __init__(self, seconds=120):
        self.seconds = seconds
        self.samples = deque(maxlen=max(600, seconds * 10))
        self.run_id = None

    def add(self, sample):
        run_id = sample.get('campaign', {}).get('run_id')
        if self.run_id != run_id:
            self.samples.clear()
            self.run_id = run_id
        stamp = sample.get('monotonic', time.monotonic())
        if self.samples and stamp == self.samples[-1][0]:
            return
        self.samples.append((stamp, {c['cpu']: c for c in sample.get('cores', [])}))
        while self.samples and self.samples[0][0] < stamp - self.seconds:
            self.samples.popleft()

    def cells(self, cpu, width):
        if not self.samples:
            return [None] * width
        end = self.samples[-1][0]
        start = end - self.seconds
        cells = [None] * width
        for stamp, cores in self.samples:
            index = min(width - 1, max(0, int((stamp - start) / self.seconds * width)))
            row = cores.get(cpu)
            if row is not None:
                # Keep the highest measured utilization if multiple polls share a cell.
                old = cells[index]
                if old is None or (row.get('measured_utilization_pct') or 0) >= (old.get('measured_utilization_pct') or 0):
                    cells[index] = row
        return cells


def dominant(row):
    threads = [t for t in row.get('threads', []) if (t.get('cpu_time_delta_s') or 0) > 0]
    return max(threads, key=lambda t: t.get('cpu_time_delta_s', 0), default={})


def activity(row):
    utilization = row.get('measured_utilization_pct')
    if utilization is None:
        return 'unknown'
    if utilization < 2:
        return 'idle'
    return kind(dominant(row).get('action'))


def _cores(sample, history, width, color, offset):
    rows = sample.get('cores', [])
    chart_width = max(12, min(100, int((width - 17) * .55)))
    detail_width = max(12, width - chart_width - 17)
    axis = f'-{history.seconds}s'.ljust(max(1, chart_width - 3)) + 'now'
    lines = [paint(f' CPU   {axis}   busy   sampled action / capsule', '2', color)]
    for row in rows[offset:]:
        cells = []
        for item in history.cells(row['cpu'], chart_width):
            state = activity(item) if item else None
            cells.append(' ' if state is None else paint(GLYPHS[state], COLORS[state], color))
        task = dominant(row)
        state = activity(row)
        label = clean(task.get('action') or ('idle' if state == 'idle' else 'unattributed'))
        if task.get('capsule'):
            label += ' / ' + clean(task['capsule'])
        elif task.get('pid'):
            label += f" / pid {task['pid']}"
        busy = number(row.get('measured_utilization_pct'), 0)
        lines.append(f" {row['cpu']:>3} │{''.join(cells)}│ {busy:>4}%  " + paint(label[:detail_width], COLORS[state], color))
    if not rows:
        lines.append(' CPU reservation not recorded yet.')
    return lines


def _jobs(sample, width, color, offset):
    lines = [' STATE / STAGE              CAPSULE / OWNER                         CPU / WAIT']
    jobs = sample.get('jobs') or []
    for job in jobs[offset:]:
        stage = clean(job.get('stage') or job.get('action') or job.get('state') or 'unknown', 25)
        capsule = clean(job.get('capsule') or job.get('label') or job.get('owner') or 'not attributed', 40)
        detail = clean(job.get('wait_reason') or job.get('detail') or job.get('state'), 45)
        lines.append(paint(f' {stage:<26}', COLORS[kind(stage)], color) + f'{capsule:<41} {detail}')
    if not jobs:
        lines.append(' No job-stage records available. Observed processes below:')
    lines.append('')
    lines.append(' PID       PPID      OBSERVED PROCESS / ACTION               CPU TIME THIS SAMPLE')
    processes = sample.get('processes') or []
    for process in processes[offset:offset + 40]:
        action = clean(process.get('action') or process.get('name') or process.get('comm') or 'unknown', 42)
        delta = process.get('cpu_time_delta_s')
        lines.append(f" {str(process.get('pid', '—')):<10}{str(process.get('ppid', '—')):<10}" +
                     paint(f'{action:<44}', COLORS[kind(action)], color) + elapsed(delta))
    return lines


def _agents(metadata, color, offset):
    agents = metadata.get('agents') or []
    lines = [' RECORDED AGENT RELATIONSHIPS  ·  execution status is separate from compiler branches']
    if not agents:
        return lines + [' No parent/child agent records available yet. Process activity is in JOBS.']
    indexed = {str(a.get('id') or a.get('agent_id') or i): a for i, a in enumerate(agents)}
    children = {}
    roots = []
    for key, item in indexed.items():
        parent = str(item.get('parent_id') or item.get('parent') or '')
        if parent in indexed and parent != key:
            children.setdefault(parent, []).append(key)
        else:
            roots.append(key)
    visited = set()

    def walk(key, prefix='', last=True, depth=0):
        if key in visited or depth > 20:
            return
        visited.add(key)
        item = indexed[key]
        path = item.get('agent_path')
        raw_name = str(path).rsplit('/', 1)[-1] if path else item.get('name') or item.get('role') or key
        name = clean('author ' + key[:8] if raw_name == key else raw_name, 32)
        state = clean(item.get('status') or item.get('state') or 'unknown', 20)
        branch = clean(item.get('branch') or item.get('source_sha256') or '', 36)
        lines.append(prefix + ('└─ ' if last else '├─ ') + paint(name, COLORS['agent'], color) + f'  {state}' + (f'  [{branch}]' if branch else ''))
        nested = children.get(key, [])
        for i, child in enumerate(nested):
            walk(child, prefix + ('   ' if last else '│  '), i == len(nested) - 1, depth + 1)
    for i, root in enumerate(roots):
        walk(root, last=i == len(roots) - 1)
    for key in indexed:
        if key not in visited:
            walk(key)
    return lines[:1] + lines[1 + offset:]


def _performance(metadata, width, color, offset):
    lines = [' CAPSULE                         CYCLES       FLOOR      FLOOR/CYCLES  ENGINE / MODEL / SCOPE',
             ' Ratios require matched timing scope. A provisional estimate is not an RTL measurement.']
    rows = metadata.get('capsules') or []
    for row in rows[offset:]:
        name = clean(row.get('capsule') or row.get('name') or 'not recorded', 31)
        cycles = row.get('cycles')
        floor = row.get('roofline_cycles')
        ratio = row.get('roofline_efficiency_pct')
        # Never create a cross-source ratio here; the evidence reader must qualify it.
        ratio_text = number(ratio) + ('%' if isinstance(ratio, (int, float)) else '')
        label = ' / '.join(clean(row.get(key), 26) for key in ('engine', 'model_status', 'scope') if row.get(key))
        state = clean(row.get('status') or 'not recorded', 25)
        lines.append(f' {name:<32}{number(cycles, 0):>11} {number(floor, 0):>11} ' +
                     paint(f'{ratio_text:>13}', COLORS['L2'], color) + f'  {label or state}')
        source = row.get('source_sha256')
        if source:
            lines.append(paint(f'   compiler {clean(source, 16)}  {state}', '2', color))
    if not rows:
        lines.append(' No qualified cycle/roofline records yet. Missing values are not zero.')
    return lines


def _attempts(metadata, color, offset):
    lines = [' OPTIMIZATION / COMPILER HISTORY  ·  proposals and measured improvements remain distinct']
    rows = metadata.get('attempts') or []
    if not rows:
        return lines + [' No optimization-attempt records available yet.']
    for row in rows[offset:]:
        stamp = clean(row.get('time') or row.get('at') or row.get('timestamp'), 24)
        status = clean(row.get('status') or row.get('outcome') or 'recorded', 22)
        title = clean(row.get('title') or row.get('optimization') or row.get('action') or row.get('summary'), 140)
        lines.append(paint(f' {stamp:<25}{status:<23}', COLORS['build'], color) + title)
        source = clean(row.get('source_sha256') or row.get('revision') or row.get('compiler_sha256'), 30)
        if source:
            lines.append(paint('   compiler ' + source, '2', color))
    return lines


def render(sample, metadata, history, *, tab=-1, width=120, height=40, color=True,
           frame=0, paused=False, offset=0, error=None):
    campaign = sample.get('campaign') or {}
    width, height = max(45, width), max(12, height)
    run = clean(campaign.get('run_id') or 'waiting for campaign', max(20, width - 44))
    pulse = '●' if paused else ('◐', '◓', '◑', '◒')[frame % 4]
    mode = 'OPERATOR' if sample.get('operator_private') else 'PUBLIC'
    lines = [paint(f' {pulse} MERLIN LIVE  ', '1;38;5;45', color) + run + paint(f'   {mode}', '2', color)]
    cores = sample.get('cores') or []
    known = [c['measured_utilization_pct'] for c in cores if isinstance(c.get('measured_utilization_pct'), (int, float))]
    busy = f'{sum(known) / 100:.1f}/{len(cores)}' if known else 'unknown'
    coverage = metadata.get('coverage') or {}
    if coverage.get('n_selected') is not None:
        current = '' if coverage.get('snapshot_current') is True else ' (retained snapshot)'
        progress = f"Checked {coverage.get('n_completed', '—')}/{coverage['n_selected']}  Passed {coverage.get('n_passed', '—')}{current}"
    else:
        progress = clean(coverage.get('summary') or campaign.get('state') or 'waiting for recorded results', 75)
    lines.append(f' CPU busy {busy}   {progress}')
    if tab == -1:
        return overview(sample, metadata, history, lines=lines, width=width, height=height,
                        color=color, frame=frame, paused=paused, error=error)
    if tab == 0:
        body = _cores(sample, history, width, color, offset)
        legend = '  '.join(paint(f'{GLYPHS[k]} {k}', COLORS[k], color)
                           for k in ('agent', 'L2', 'L3', 'build', 'verify', 'calibration', 'unknown', 'idle'))
        lines.append(legend)
    elif tab == 1:
        body = _jobs(sample, width, color, offset)
    elif tab == 2:
        body = _agents(metadata, color, offset)
    elif tab == 3:
        body = _performance(metadata, width, color, offset)
    else:
        body = _attempts(metadata, color, offset)
    room = max(1, height - len(lines) - 3)
    lines += body[:room]
    lines += [''] * max(0, height - 3 - len(lines))
    notice = error or next(iter(metadata.get('notices') or []), '')
    if not isinstance(notice, str):
        notice = str(notice)
    if not notice:
        notice = 'Per-core activity is sampled attribution; CPU busy is measured separately. Blank history predates this display.'
    lines.append(paint(' ' + clean(notice, width - 2), '38;5;220' if error else '2', color))
    lines.append(paint(' Read-only | 1-5 / Tab views   j/k scroll   Space pause view   q quit', '2', color))
    lines.append(paint(' ' + ('VIEW PAUSED; collection continues' if paused else 'LIVE') +
                       '   ' + time.strftime('%H:%M:%S UTC', time.gmtime()) +
                       '   No missing metric is treated as a pass.', '2', color))
    return '\n'.join(clip_ansi(line, width - 1) for line in lines)


def overview(sample, metadata, history, *, lines, width, height, color, frame, paused, error):
    """One screen keeps cores, agent lineage, performance, jobs and attempts visible."""
    left_width = max(45, int((width - 3) * .61))
    right_width = max(20, width - left_width - 3)
    legend = ' '.join(paint(f'{GLYPHS[k]}={k}', COLORS[k], color)
                      for k in ('agent', 'L2', 'L3', 'build', 'verify', 'calibration'))
    lines.extend(clean(row, width - 1) for row in terminal_health.summary(metadata.get('progress_health')))
    lines.append(legend + paint('  · idle  ? unattributed', '2', color))
    left = _cores(sample, history, left_width, color, 0)
    right = [paint(' AGENT TREE', '1;38;5;213', color)]
    tree = _agents(metadata, color, 0)[1:]
    right.extend(tree[:4])
    if len(tree) > 4:
        right.append(paint(f' + {len(tree) - 4} further recorded agents', '2', color))
    right += ['', paint(' PERFORMANCE  cycles / floor / efficiency', '1;38;5;45', color)]
    capsules = metadata.get('capsules') or []
    ordered = sorted(capsules, key=lambda r: r.get('roofline_efficiency_pct')
                     if isinstance(r.get('roofline_efficiency_pct'), (int, float)) else float('inf'))
    for row in ordered[:2]:
        name = clean(row.get('capsule') or row.get('name') or 'unknown', right_width - 2)
        ratio = row.get('roofline_efficiency_pct')
        ratio_text = number(ratio) + ('%' if isinstance(ratio, (int, float)) else '')
        right.append(' ' + name)
        right.append(paint(f" {number(row.get('cycles'), 0)} / {number(row.get('roofline_cycles'), 0)} / {ratio_text}", COLORS['L2'], color))
        right.append(paint(' ' + clean(' / '.join(str(row.get(k) or '') for k in ('engine', 'model_status')), right_width - 2), '2', color))
    if not capsules:
        right += [' No qualified cycles recorded yet.', ' Roofline efficiency: not recorded.']
    right += [paint(' structural floor / estimate; not HW utilization', '2', color), '',
              paint(' RECENT OPTIMIZATION ATTEMPTS', '1;38;5;214', color)]
    attempts = metadata.get('attempts') or []
    for attempt in attempts[:4]:
        status = clean(attempt.get('status') or attempt.get('outcome') or 'recorded', 18)
        title = clean(attempt.get('title') or attempt.get('optimization') or attempt.get('action') or attempt.get('summary'), right_width - 4)
        right += [paint(' ' + status, COLORS['build'], color) + ' ' + title]
    if not attempts:
        right += [' No attempt records available yet.']
    available = max(4, height - len(lines) - 7)
    body_rows = min(max(len(left), len(right)), available)
    for index in range(body_rows):
        a = left[index] if index < len(left) else ''
        b = right[index] if index < len(right) else ''
        lines.append(pad(a, left_width) + paint(' │ ', '2', color) + clip_ansi(b, right_width - 1))
    if len(left) > body_rows:
        lines.append(paint(f' {len(left) - body_rows} more core rows: enlarge terminal to at least {len(left) + 10} lines.', '38;5;220', color))
    job_count = len(sample.get('jobs') or [])
    lines.append(paint(f' CURRENT JOBS / QUEUE  ({job_count} records)', '1;38;5;141', color))
    jobs = sorted(sample.get('jobs') or [], key=lambda j: ({'setup_stage': 0, 'cpu_waiter': 1, 'cpu_lease': 2}.get(j.get('kind'), 3), -(j.get('age_s') or 0)))
    remaining = max(1, height - len(lines) - 3)
    for job in jobs[:min(remaining, 4)]:
        stage = clean(job.get('stage') or job.get('action') or job.get('state') or 'unknown', 22)
        capsule = clean(job.get('capsule') or job.get('label') or job.get('owner') or 'capsule not attributed', 52)
        wait = clean(job.get('wait_reason') or job.get('detail') or job.get('state'), 70)
        if job.get('age_s') is not None:
            wait += ' ' + elapsed(job['age_s'])
        lines.append(paint(f' {stage:<23}', COLORS[kind(stage)], color) + capsule + '  ' + wait)
    if not jobs:
        lines.append(' No active job-stage records; collection continues.')
    lines += [''] * max(0, height - 3 - len(lines))
    health_alert = terminal_health.alert(metadata.get('progress_health'))
    note = error or health_alert or 'CPU busy is measured; task-to-core attribution is sampled. Metrics retain their recorded model/scope.'
    lines.append(paint(' ' + clean(note, width - 2), '38;5;220' if error or health_alert else '2', color))
    lines.append(paint(' READ ONLY  |  Space pause display  q quit  |  blank timeline = before this display opened', '2', color))
    lines.append(paint(' ' + ('DISPLAY PAUSED; sampling continues' if paused else 'LIVE') + '  ' +
                       time.strftime('%H:%M:%S UTC', time.gmtime()), '2', color))
    return '\n'.join(clip_ansi(line, width - 1) for line in lines[:height])
