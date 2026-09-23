"""The port-publish gate must fail on the thing that actually happened.

WHY THIS EXISTS, 2026-09-21. A SigNoz stack published its UI on ``0.0.0.0:8080`` and its OTLP
collector on ``0.0.0.0:4317``/``4318`` of a publicly routable host, carried
``restart: unless-stopped`` so it returned 16 seconds after each reboot, and was reachable for
4.5 days before anyone looked. Nothing had ever used it.

A gate that only passes on a clean tree is indistinguishable from a gate that cannot fail, so
every test here is a mutation: it plants a spelling that must be refused, or one that must not be.
The two central cases are the exact lines this repo shipped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_no_public_port_bindings.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("port_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _yaml(gate, tmp_path: Path, body: str):
    p = tmp_path / "compose.yml"
    p.write_text(body)
    return gate._scan_compose(p, body.splitlines())


def _text(gate, tmp_path: Path, body: str):
    p = tmp_path / "doc.md"
    p.write_text(body)
    return gate._scan_text(p, body.splitlines())


# --- the two spellings this repo actually shipped -------------------------------------------


def test_the_mlflow_compose_spelling_is_refused(gate, tmp_path):
    """THE POINT, compose half. `"5000:5000"` binds every interface, and MLflow ships with no
    authentication, so this is an unauthenticated server on a public address."""
    found = _yaml(gate, tmp_path, 'services:\n  mlflow:\n    ports:\n      - "5000:5000"\n')
    assert len(found) == 1
    assert "every interface" in found[0].why


def test_the_docker_run_spelling_is_refused(gate, tmp_path):
    """THE POINT, docs half. The collector invocation documented in `claude_code_otel.md`."""
    found = _text(gate, tmp_path, "docker run --rm -p 4317:4317 -p 4318:4318 otel/collector\n")
    assert len(found) == 2, "both published ports must be named, not just the first"


# --- what must NOT be flagged ----------------------------------------------------------------


def test_loopback_passes(gate, tmp_path):
    """The control. Without it this suite would pass for a gate that refuses every publish, which
    would make the loopback fix itself a violation and leave no green state to reach."""
    assert _yaml(gate, tmp_path, 'services:\n  a:\n    ports:\n      - "127.0.0.1:5000:5000"\n') == []
    assert _text(gate, tmp_path, "docker run -p 127.0.0.1:4318:4318 x\n") == []


def test_a_bare_container_port_is_not_a_host_binding(gate, tmp_path):
    """`ports: ["8080"]` and an image's EXPOSE metadata publish nothing to the host. The SigNoz
    ZooKeeper container listed four such ports in `docker ps` and was reachable on none of them --
    reading those as exposure is how a reviewer wastes a day on the wrong container."""
    assert _yaml(gate, tmp_path, 'services:\n  a:\n    ports:\n      - "8080"\n') == []


def test_an_in_container_bind_is_not_flagged(gate, tmp_path):
    """A server binding 0.0.0.0 INSIDE a container is correct -- one binding 127.0.0.1 there is
    unreachable even through its own published port. Only the publish decides host exposure, so
    'hardening' the receiver would break the service and leave the exposure untouched."""
    body = "receivers:\n  otlp:\n    protocols:\n      http:\n        endpoint: 0.0.0.0:4318\n"
    assert _yaml(gate, tmp_path, body) == []
    assert _text(gate, tmp_path, body) == []


# --- spellings that must not slip past --------------------------------------------------------


def test_explicit_all_interfaces_is_refused(gate, tmp_path):
    """`0.0.0.0:9000:9000` is the same exposure written out; a gate keyed only on the two-part
    form would wave it through."""
    found = _yaml(gate, tmp_path, 'services:\n  a:\n    ports:\n      - "0.0.0.0:9000:9000"\n')
    assert len(found) == 1 and "0.0.0.0" in found[0].why


def test_long_form_is_refused(gate, tmp_path):
    """Compose's long form carries `host_ip` as its own key; a scanner that only split strings
    would see no colon and report nothing."""
    body = "services:\n  a:\n    ports:\n      - target: 7000\n        published: 7000\n        host_ip: 0.0.0.0\n"
    assert len(_yaml(gate, tmp_path, body)) == 1


def test_equals_form_is_refused(gate, tmp_path):
    """`--publish=0.0.0.0:3002:3002` attaches the value to the flag token."""
    assert len(_text(gate, tmp_path, "docker run --publish=0.0.0.0:3002:3002 x\n")) == 1


def test_a_continued_line_is_still_scanned(gate, tmp_path):
    """The real invocation wraps across lines with a backslash, so the publish flag sits on a line
    that does not itself contain `docker run`."""
    body = "docker run --rm \\\n  -p 4318:4318 \\\n  otel/collector\n"
    assert len(_text(gate, tmp_path, body)) == 1


# --- the escape hatch ------------------------------------------------------------------------


def test_the_inline_rationale_exempts_one_line(gate, tmp_path):
    """A refusal with no way past it is a wall, not a gate. A port that genuinely must be off-box
    declares itself in place, which turns a silent default into a reviewed claim."""
    body = "docker run -p 3003:3003 x  # public-port-ok: bench needs off-box access\n"
    assert _text(gate, tmp_path, body) == []


def test_the_allowlist_only_shrinks(gate):
    """Matches the other ratcheted gates. The single entry is THIS file, which is exempt for the
    same reason markers.py is exempt from the regex gate: its fixtures are the violations, so
    scanning it would make the suite unable to pass. Any second entry is a reviewable regression.
    """
    text = gate.ALLOW_FILE.read_text()
    assert "ONLY SHRINKS" in text
    entries = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert len(entries) <= 1, "the allowlist grew; justify each entry in review"
    for line in entries:
        assert "#" in line, f"an allowlist entry must carry its rationale: {line!r}"
    assert [ln.split("#", 1)[0].strip() for ln in entries] == ["merlin/tests/infra/test_no_public_port_bindings.py"]


def test_an_allowlisted_file_is_actually_skipped(gate):
    """The carve-out has to work, or the entry above is decoration and the suite fails anyway."""
    flagged = {v.path.name for v in gate.scan(staged=False)}
    assert "test_no_public_port_bindings.py" not in flagged


# --- the tree itself -------------------------------------------------------------------------


def test_the_tracked_tree_is_clean(gate):
    """The standing assertion. If this fails, something published a port on every interface."""
    found = gate.scan(staged=False)
    assert found == [], "\n".join(str(v) for v in found)
