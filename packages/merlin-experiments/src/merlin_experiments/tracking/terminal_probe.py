#!/usr/bin/env python3
"""Bounded read-only Linux resource observations for an operator terminal.

``Sampler(task, operator_private=False).sample()`` returns JSON-compatible data.
No commands, locks, or writes are performed. Resource observations are display
telemetry, never numerical or performance evidence. Core utilization comes from
/proc/stat; thread last_cpu is only a point sample. Opaque worker attribution
stays unknown. ``proc_root`` and ``cgroup_root`` support harmless fixtures.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path
import time



def read(path, limit=1 << 20):
    try:
        with Path(path).open('rb') as f:
            data = f.read(limit + 1)
        return data if len(data) <= limit else None
    except (OSError, ValueError, TypeError):
        return None


def document(path, limit=4 << 20):
    try:
        raw = read(path, limit)
        return json.loads(raw) if raw else None
    except (ValueError, TypeError):
        return None


def cpus(value):
    out = set()
    try:
        for item in value.split(','):
            a, sep, b = item.partition('-')
            low, high = int(a), int(b if sep else a)
            if low < 0 or high < low or high > 65535:
                return []
            out.update(range(low, high + 1))
    except (ValueError, AttributeError):
        return []
    return sorted(out)


def task_stat(pid, tid=None, proc_root=Path("/proc")):
    raw = read(proc_root / str(pid) / (f'task/{tid}/stat' if tid else 'stat'), 16384)
    if not raw:
        return None
    try:
        rest = raw.decode().rsplit(')', 1)[1].split()
        return {'pid': pid, 'tid': tid or pid, 'state': rest[0], 'ppid': int(rest[1]),
                'cpu_ticks': int(rest[11]) + int(rest[12]), 'start_ticks': int(rest[19]),
                'last_cpu': int(rest[36])}
    except (ValueError, IndexError):
        return None


def cpu_counters(proc_root=Path("/proc")):
    raw = read(proc_root / 'stat', 1 << 20)
    result = {}
    for line in (raw or b'').decode().splitlines():
        fields = line.split()
        if (fields and fields[0].startswith('cpu')
                and fields[0][3:].isascii() and fields[0][3:].isdecimal()):
            try:
                counts = list(map(int, fields[1:9]))
                result[int(fields[0][3:])] = (sum(counts), counts[3] + counts[4])
            except (ValueError, IndexError):
                continue
    return result


def locks(proc_root=Path("/proc")):
    result = {}
    for line in (read(proc_root / 'locks', 4 << 20) or b'').decode().splitlines():
        f = line.split()
        if len(f) < 8 or '->' in f:
            continue
        try:
            major, minor, inode = f[5].split(':')
            result.setdefault((os.makedev(int(major, 16), int(minor, 16)), int(inode)), []).append(
                {'pid': int(f[4]), 'lock_kind': f[1], 'mode': f[3]})
        except (ValueError, IndexError):
            pass
    return result


def lock_rows(path, inventory):
    try:
        s = path.stat()
        return inventory.get((s.st_dev, s.st_ino), [])
    except OSError:
        return []


def classify(argv, selection):
    if not argv:
        return 'unknown', 'argv unavailable'
    executable = Path(argv[0]).name
    # A broker schedules work; its name is not evidence of an engine running.
    module = None
    if executable.startswith('python') and '-m' in argv:
        index = argv.index('-m')
        if index + 1 < len(argv):
            module = argv[index + 1]
    services = {
        'merlin_experiments.phase1.brokers.selfcheck': 'host_selfcheck_broker',
        'merlin_experiments.phase1.brokers.simjob': 'host_simjob_broker',
        'merlin_experiments.phase1.brokers.model_check': 'host_model_check_broker',
        'merlin_experiments.phase1.brokers.model_check_worker': 'host_model_check_worker',
        'merlin_experiments.phase1.brokers.isa_tools': 'host_isa_tools_broker',
        'merlin_experiments.phase1.brokers.cca': 'host_cca_broker',
        'merlin_experiments.phase1.feedback.selfcheck': 'host_selfcheck_worker',
        'merlin_experiments.phase1.joint_calibration_service': 'calibration_service',
    }
    if module in services:
        return services[module], 'exact Python -m service identity; execution stage unknown'
    script = next((value for value in argv[1:4] if not value.startswith('-')), '')
    if (executable.startswith('python') and '-m' not in argv and '-c' not in argv
            and Path(script).parts[-3:] == ('merlin', 'common', 'read_lease_digest.py')):
        return 'verification_lease_service', 'exact helper script suffix; custody service, not target execution'
    if executable in ('codex-code-mode-host', 'code-mode-host'):
        return 'agent_tool_host', 'exact tool host executable; individual tool action unknown'
    if executable in ('bwrap', 'bubblewrap'):
        return 'host_sandbox_wrapper', 'exact sandbox wrapper executable; child action not inferred'
    if executable in ('sleep', 'bash', 'sh', 'dash'):
        return 'host_' + executable, 'exact host executable; busy state comes only from CPU deltas'
    env = selection.get('environment', {})
    names = {Path(x).name for x in argv[:4]}
    gsim_executables = (value for key, value in env.items()
                        if key.startswith('MERLIN_GSIM_EMU_') or key.endswith('_GSIM_EMU'))
    if any(isinstance(value, str) and value and value in argv for value in gsim_executables):
        return 'L3', 'selected gSIM executable in argv'
    if any(x in names for x in ('spike', 'spike-timing')) or (env.get('MERLIN_SPIKE') and env['MERLIN_SPIKE'] in argv):
        return 'L2_or_cycle_observer', 'Spike executable; numerical versus timing purpose unavailable'
    if 'codex' in names or 'claude' in names:
        return 'agentic', 'agent executable in argv; remote inference time is not local CPU work'
    if any(x.startswith(('clang', 'riscv64-unknown-elf-gcc', 'ld.lld', 'llc', 'mlir-opt')) for x in names):
        return 'compile', 'compiler executable in argv'
    if 'merlin_experiments.phase1.joint_calibration_owner' in argv:
        return 'calibration_supervisor', 'exact module argv'
    if 'merlin_experiments.phase1' in argv:
        return 'native_evaluator', 'exact module argv; worker thread action unknown'
    return 'unknown', 'no exact stage mapping'


def capsule_from_paths(argv, cwd):
    found = set()
    for value in [*argv, cwd]:
        if not isinstance(value, str) or not value.startswith('/'):
            continue
        parts = Path(value).parts
        for i, part in enumerate(parts[:-1]):
            if part.endswith('-capsule-bench'):
                found.add(parts[i + 1])
    return next(iter(found)) if len(found) == 1 else None


class Sampler:
    def __init__(self, task: Path, operator_private=False, *, selection=None,
                 proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup"),
                 max_processes=256, max_threads=1024, max_timing_records=256):
        self.task = Path(task)
        self.operator_private = operator_private
        self.proc_root, self.cgroup_root = Path(proc_root), Path(cgroup_root)
        self._timing_iterator = None
        self._timing_root = None
        self._timing_active = {}
        self._workspace_key = None
        self._workspace = None
        self.explicit_selection = Path(selection) if selection else None
        self.max_processes, self.max_threads = max_processes, max_threads
        self.max_timing_records = max_timing_records
        self.previous_time = None
        self.previous_cpus = {}
        self.previous_threads = {}
        self.metadata_key = None
        self.metadata = None
        self.owner_path = None

    def selected(self):
        # Bounded one-level operator owner discovery, never recursive run/log scans.
        rows = []
        for p in itertools.islice((self.task / '.tmp').glob('*/process/owner.json'), 128):
            d = document(p, 256 << 10)
            if not isinstance(d, dict) or not isinstance(d.get('pid'), int):
                continue
            if self.explicit_selection and str(self.explicit_selection) != d.get('selection', {}).get('path'):
                continue
            live = task_stat(d['pid'], proc_root=self.proc_root)
            live = bool(live and live['state'] != 'Z' and self._owner_alive(d))
            rows.append((live, d.get('started_at_unix', 0), p, d))
        if not rows:
            return None, None, None
        _, _, path, owner = max(rows, key=lambda r: (r[0], r[1]))
        binding = owner.get('selection', {})
        selected_path = self.explicit_selection or Path(binding.get('path', ''))
        key = (str(selected_path), binding.get('sha256'))
        if key != self.metadata_key:
            raw = read(selected_path, 4 << 20)
            if not raw or hashlib.sha256(raw).hexdigest() != binding.get('sha256'):
                return owner, None, path
            try:
                self.metadata = json.loads(raw)
            except ValueError:
                return owner, None, path
            if not isinstance(self.metadata, dict):
                return owner, None, path
            self.metadata_key = key
        return owner, self.metadata, path

    def _owner_alive(self, owner):
        expected = owner.get('argv')
        if not isinstance(expected, list) or not expected:
            return False
        if Path(expected[0]).name == 'taskset' and expected[1:2] == ['-c']:
            expected = expected[3:]
        raw = read(self.proc_root / str(owner['pid']) / 'cmdline', 65536)
        actual = [x.decode(errors='replace') for x in (raw or b'').split(b'\0') if x]
        return actual == expected

    def _workspace_path(self, run_dir):
        # Read one bounded existing record, not a guessed directory or a YAML loader.
        path = run_dir / 'environment.yaml'
        try:
            st = path.stat()
            key = (str(path), st.st_size, st.st_mtime_ns)
        except OSError:
            return None
        if key != self._workspace_key:
            raw = read(path, 2 << 20) or b''
            matches = set()
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped.startswith(b'- /'):
                    value = stripped[2:]
                    if b'/agent-workspaces/' in value and value.endswith(b'/workspace'):
                        matches.add(value)
            self._workspace = next(iter(matches)).decode(errors='replace') if len(matches) == 1 else None
            self._workspace_key = key
        return self._workspace

    @staticmethod
    def _timing_entries(root):
        # Fixed-depth incremental traversal; no recursive corpus/log discovery.
        def leaves(directory):
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        yield Path(entry.path) if entry.is_file(follow_symlinks=False) else None
            except OSError:
                return
        yield from leaves(root / 'profile_package_execution/setup-timing')
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    yield None  # Charge directory discovery against the same bounded scan.
                    if entry.name.startswith('fruns_') and entry.is_dir(follow_symlinks=False):
                        yield from leaves(Path(entry.path) / 'host-stage-timing/setup-timing')
        except OSError:
            return

    def _timing_records(self, root):
        # Refresh bounded active records and advance a bounded directory iterator.
        # No full-directory sort/stat on every UI refresh.
        if root != self._timing_root:
            if self._timing_iterator is not None:
                self._timing_iterator.close()
            self._timing_iterator = None
            self._timing_active.clear()
            self._timing_root = root
        paths = set(self._timing_active)
        if self._timing_iterator is None:
            try:
                self._timing_iterator = self._timing_entries(root)
            except OSError:
                return []
        try:
            for _ in range(self.max_timing_records):
                entry = next(self._timing_iterator)
                if entry is not None and entry.name.endswith('.json'):
                    paths.add(entry)
        except (StopIteration, OSError):
            self._timing_iterator.close()
            self._timing_iterator = None
        for path in paths:
            doc = document(path, 16384)
            if (isinstance(doc, dict) and doc.get('schema') == 'merlin.compiler_setup_timing.v1'
                    and isinstance(doc.get('stages'), list)
                    and any(isinstance(s, dict) and s.get('status') == 'running' for s in doc['stages'])):
                self._timing_active[path] = doc
            else:
                self._timing_active.pop(path, None)
        while len(self._timing_active) > self.max_timing_records:
            self._timing_active.pop(next(iter(self._timing_active)))
        return list(self._timing_active.values())

    def sample(self):
        started = time.monotonic()
        wall = time.time()
        owner, selection, owner_path = self.selected()
        if not selection:
            return {'schema': 'core_timeline.v1', 'timestamp': wall, 'status': 'selection_unavailable',
                    'cores': [], 'threads': [], 'processes': [], 'jobs': [], 'campaign': {},
                    'operator_private': self.operator_private, 'read_only': True}
        env = selection.get('environment', {})
        native_pid = owner['pid']
        owner_alive = self._owner_alive(owner)
        raw = read(self.proc_root / str(native_pid) / 'cgroup', 16384)
        cgroup = None
        for line in (raw if owner_alive and raw else b'').decode().splitlines():
            if line.startswith('0::'):
                cgroup = line[3:]
        assigned = selection.get('taskset_cpus') or cpus(env.get('MERLIN_COMPUTE_CPUS', ''))
        group_root = self.cgroup_root / cgroup.lstrip('/') if cgroup else None
        raw_pids = read(group_root / 'cgroup.procs', 65536) if group_root else None
        pids = sorted({int(x) for x in (raw_pids if raw_pids is not None else (str(native_pid).encode() if owner_alive else b'')).split() if x.isdigit()})
        limits = []
        if len(pids) > self.max_processes:
            limits.append('process roster truncated')
        pids = pids[:self.max_processes]
        counters = cpu_counters(self.proc_root)
        elapsed = started - self.previous_time if self.previous_time is not None else None
        hz = os.sysconf('SC_CLK_TCK')
        threads, processes, next_threads = [], [], {}
        for pid in pids:
            argbytes = read(self.proc_root / str(pid) / 'cmdline', 65536)
            argv = [x.decode(errors='replace') for x in (argbytes or b'').split(b'\0') if x]
            try:
                cwd = os.readlink(self.proc_root / str(pid) / 'cwd')
            except OSError:
                cwd = ''
            action, basis = classify(argv, selection)
            capsule = capsule_from_paths(argv, cwd) if self.operator_private else None
            processes.append({'pid': pid, 'executable': Path(argv[0]).name if argv else None,
                              'action': action, 'action_basis': basis, 'capsule': capsule,
                              'capsule_basis': 'unique native capsule directory in argv/cwd' if capsule else 'unknown'})
            try:
                tids = sorted(int(p.name) for p in (self.proc_root / str(pid) / 'task').iterdir() if p.name.isdigit())
            except OSError:
                continue
            for tid in tids:
                if len(threads) >= self.max_threads:
                    limits.append('thread roster truncated')
                    break
                row = task_stat(pid, tid, self.proc_root)
                if row is None:
                    continue
                key = (pid, tid, row['start_ticks'])
                old = self.previous_threads.get(key)
                delta = max(0, row['cpu_ticks'] - old) / hz if old is not None and elapsed else None
                next_threads[key] = row['cpu_ticks']
                status = read(self.proc_root / str(pid) / 'task' / str(tid) / 'status', 16384)
                affinity = next((cpus(line.split(':', 1)[1].strip()) for line in (status or b'').decode().splitlines()
                                 if line.startswith('Cpus_allowed_list:')), [])
                row.update(cpu_time_delta_s=delta, usage_pct=100 * delta / elapsed if delta is not None else None,
                           affinity_cpus=affinity, action=action, action_basis=basis, capsule=capsule,
                           cpu_attribution_basis='last_cpu is a point sample; CPU-time delta is per-thread, not per-core')
                threads.append(row)
        inventory = locks(self.proc_root)
        jobs = []
        pool = Path(env['MERLIN_COMPUTE_CPU_SLOTS']) if env.get('MERLIN_COMPUTE_CPU_SLOTS') else None
        if pool:
            for cpu in assigned:
                for held in lock_rows(pool / f'cpu-{cpu}.lock', inventory):
                    jobs.append({'kind': 'cpu_lease', 'state': 'kernel_lock_held', 'cpu': cpu, **held,
                                 'basis': '/proc/locks inode match; PID owner, TID unknown; assignment is not utilization'})
            try:
                tickets = list(itertools.islice((pool / 'waiters').iterdir(), 512))
            except OSError:
                tickets = []
            for ticket in tickets:
                fields = ticket.name.removesuffix('.wait').split('-')
                if (not ticket.name.endswith('.wait') or len(fields) != 5
                        or not all(value.isascii() and value.isdecimal() for value in fields[:4])
                        or not fields[4] or any(c not in '0123456789abcdef' for c in fields[4])):
                    continue
                priority, queued, pid, tid = map(int, fields[:4])
                body = document(ticket, 4096) or {}
                held = lock_rows(ticket, inventory)
                jobs.append({'kind': 'cpu_waiter', 'state': 'locked_wait_ticket' if held else 'ticket_liveness_unknown',
                             'pid': pid, 'tid': tid, 'priority': priority, 'queued_ns': queued,
                             'eligible_cpus': body.get('cpus'), 'basis': 'bounded waiter filename/payload and kernel lock lookup'})
        run_dir = Path(selection['run_dir'])
        timing = run_dir / '_qa_work'
        live_threads = {(t['pid'], t['tid']): t for t in threads}
        associations = {}
        boot_epoch = None
        for line in (read(self.proc_root / 'stat') or b'').splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == b'btime' and fields[1].isdigit():
                boot_epoch = int(fields[1])
        for d in self._timing_records(timing):
            source = d.get('source_sha256')
            source_present = (isinstance(source, str) and len(source) == 64
                              and all(c in '0123456789abcdef' for c in source))
            for stage in d.get('stages', []):
                if not isinstance(stage, dict) or stage.get('status') != 'running':
                    continue
                start_ns = stage.get('started_ns')
                pid, tid = d.get('pid'), stage.get('native_tid', d.get('native_tid'))
                if not isinstance(tid, int):
                    tid = None
                capsule = d.get('capsule') if self.operator_private else None
                job = {'kind': 'setup_stage', 'state': 'recorded_running', 'pid': pid, 'tid': tid,
                       'stage': stage.get('stage'), 'capsule': capsule,
                       'snapshot_id': d.get('snapshot_id') if self.operator_private else None,
                       'request_id': d.get('request_id') if self.operator_private else None,
                       'tier': d.get('tier'), 'tier_basis': 'recorded request only; not execution evidence',
                       'source_sha256': source if self.operator_private else None,
                       'source_identity_present': source_present,
                       'age_s': max(0, wall - start_ns / 1e9) if isinstance(start_ns, (int, float)) else None,
                       'basis': 'recorded setup diagnostics; no kernel action inferred',
                       'thread_association': 'unknown'}
                thread = live_threads.get((pid, tid))
                # btime has one-second precision. Require the stage to be later than
                # the latest possible thread birth, guarding stale/reused native IDs.
                if (thread is not None and source_present and boot_epoch is not None
                        and isinstance(start_ns, int) and thread['state'] not in ('Z', 'X')
                        and boot_epoch + thread['start_ticks'] / hz + 1 <= start_ns / 1e9 <= wall):
                    job['thread_association'] = 'pid_native_tid_source_and_lifetime_match'
                    prior = associations.get((pid, tid))
                    if prior is None or start_ns > prior[0]:
                        associations[(pid, tid)] = (start_ns, job)
                jobs.append(job)
        for key, (_, job) in associations.items():
            thread = live_threads[key]
            name = job['stage']
            verification = isinstance(name, str) and (name.startswith('invocation_snapshot')
                                                       or name.startswith('package_policy'))
            thread.update(action='verification' if verification else 'setup', stage=name,
                          action_basis='running setup diagnostic matched PID/native TID/source identity/thread lifetime; not kernel evidence',
                          capsule=job['capsule'], source_identity_present=True,
                          source_sha256=job['source_sha256'])
        cores = []
        for cpu in assigned:
            prior, now = self.previous_cpus.get(cpu), counters.get(cpu)
            utilization = None
            if prior and now and now[0] > prior[0]:
                idle_delta = now[1] - prior[1]
                total_delta = now[0] - prior[0]
                if 0 <= idle_delta <= total_delta:
                    utilization = 100 * (1 - idle_delta / total_delta)
            cores.append({'cpu': cpu, 'measured_utilization_pct': utilization,
                          'utilization_basis': '/proc/stat busy delta; includes all work on this core',
                          'threads': [t for t in threads if t['last_cpu'] == cpu],
                          'sampled_tids': [t['tid'] for t in threads if t['last_cpu'] == cpu],
                          'lease_owner_pids': [j['pid'] for j in jobs if j['kind'] == 'cpu_lease' and j['cpu'] == cpu]})
        self.previous_time, self.previous_cpus, self.previous_threads = started, counters, next_threads
        return {'schema': 'core_timeline.v1', 'timestamp': wall, 'monotonic': started,
                'sample_interval_s': elapsed, 'collection_s': time.monotonic() - started, 'status': 'observed' if owner_alive else 'owner_not_running',
                'operator_private': self.operator_private,
                'campaign': {'run_id': selection.get('run_id'), 'run_dir': str(run_dir),
                             'workspace': self._workspace_path(run_dir),
                             'native_pid': native_pid, 'owner_path': str(owner_path),
                             'selection_pin': owner.get('selection') if self.operator_private else {'present': True}, 'cgroup': cgroup, 'assigned_cpus': assigned},
                'cores': cores, 'threads': threads, 'processes': processes, 'jobs': jobs,
                'limitations': limits + ['No exact per-core thread residency trace; last_cpu is sampled.',
                    'Opaque evaluator worker threads have no invented capsule or stage assignment.',
                    'No argv bodies, environment values, expected outputs or transcripts are emitted.',
                    'Setup inventory is bounded and incrementally scanned; it may be incomplete.',
                    'Running stage records may be stale after abnormal termination; they are not CPU measurements.'],
                'read_only': True}

