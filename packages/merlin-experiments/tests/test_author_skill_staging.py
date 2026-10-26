"""Harness author skills reach both drivers byte-identically, are recorded by digest and sealed on resume.

Generic over every ``phase1/author_skills/*`` directory: a skill added later is checked by the same
rules (frontmatter both drivers list, a short body, stdlib scripts that compile and answer ``--help``,
no target names anywhere in author-visible bytes) without editing this file.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1 import run_inputs as RI
from merlin_experiments.phase1 import session, task_disclosure
from merlin_experiments.phase1 import launch_config as lc
from merlin_experiments.phase1 import skill_staging as SK
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.providers import codex_context as CTX

from merlin.common.paths import repo_root

CODEX = ["--run-id", "test", "--driver", "codex", "--sandbox", "bwrap"]
CLAUDE = ["--run-id", "test", "--driver", "claudecode"]
OWN = ("fix-failure-cluster", "feedback-files", "delegate-to-helpers")
TASK = "# Task\n\nBuild it.\n\n## Available tools\n\n" + "- `python agent_selfcheck.py`: checks.\n" * 60


def _record(options) -> dict:
    request = type("Request", (), {"options": options, "launcher_argv": (),
                                   "context": type("Context", (), {"readback_policy": None})()})()
    return session.RunRequest.run_config.fget(request)


def _gate():
    sys.path.insert(0, str(repo_root() / "build_tools" / "scripts"))
    try:
        import check_no_target_name
    finally:
        sys.path.pop(0)
    return check_no_target_name


def _stage(tmp_path) -> tuple[Path, Path, dict]:
    ws, run = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run.mkdir()
    record = SK.selection(parse_options(CODEX))
    return ws, run, SK.stage(ws, run, record)


def test_every_skill_has_listable_frontmatter_and_a_short_procedural_body():
    skills = SK.collect()
    assert set(OWN) <= set(skills)
    for name, files in skills.items():
        meta, body = SK.frontmatter(files[SK.SKILL_FILE].decode("utf-8"))
        assert meta["name"] == name and len(meta["description"]) <= SK.MAX_DESCRIPTION
        assert 0 < len(body.encode("utf-8")) <= SK.BODY_BUDGET, name
        assert not name.startswith(task_disclosure.NAME_PREFIX) and name != task_disclosure.FEEDBACK_SKILL
        assert set(files) <= {SK.SKILL_FILE} | {path for path in files if path.startswith(("scripts/", "reference/"))}
        for relative, data in files.items():
            data.decode("utf-8")  # author-visible text
            if relative.startswith("reference/"):
                assert 0 < len(data) <= SK.REFERENCE_BUDGET, (name, relative)
        assert sum(len(data) for data in files.values()) <= SK.SKILL_BUDGET, name
    assert SK.FEEDBACK_SKILL in skills


def test_the_packaged_layout_is_skill_md_scripts_and_reference_plain_files(tmp_path):
    def skill(root, name="demo"):
        directory = root / name
        (directory / "reference").mkdir(parents=True)
        (directory / SK.SKILL_FILE).write_text(f'---\nname: {name}\ndescription: "Use for demos."\n---\n\nSee '
                                               "`reference/table.md`.\n")
        (directory / "reference" / "table.md").write_text("| a | b |\n")
        return directory

    good = skill(tmp_path / "good")
    (good / "scripts").mkdir()
    (good / "scripts" / "tool.py").write_text("print(1)\n")
    assert set(SK.collect(tmp_path / "good")["demo"]) == {SK.SKILL_FILE, "reference/table.md", "scripts/tool.py"}
    for label, edit in (("nested", lambda d: (d / "reference" / "deep").mkdir() or (d / "reference" / "deep" / "x.md")
                         .write_text("x")),
                        ("suffix", lambda d: (d / "reference" / "table.txt").write_text("x")),
                        ("toplevel", lambda d: (d / "notes.md").write_text("x")),
                        ("link", lambda d: (d / "reference" / "alias.md").symlink_to(d / "reference" / "table.md"))):
        directory = skill(tmp_path / label)
        edit(directory)
        with pytest.raises(ValueError, match="unpackaged file|symlink"):
            SK.collect(tmp_path / label)
    pyproject = (repo_root() / "packages" / "merlin-experiments" / "pyproject.toml").read_text()
    for subdirectory, suffix in SK.SUBDIRECTORIES.items():
        assert f'"phase1/author_skills/*/{subdirectory}/*{suffix}"' in pyproject
    assert '"phase1/author_skills/*/SKILL.md"' in pyproject


#: Skills on branches not merged yet; the coordinator empties this after merging.
PENDING_SKILLS: frozenset[str] = frozenset()
#: Words that, right before a backticked name, say the name is a skill ("open `x`", "see `x`").
_BEFORE = frozenset({"open", "opens", "opening", "see", "skill", "skills"})
_LIST_SEPARATORS = frozenset({",", "and", "or", ",and", ",or"})


def _is_name(text: str) -> bool:
    return (bool(text) and set(text) <= SK._NAME_CHARS and text[0].isalpha() and not text.endswith("-")
            and "--" not in text)


def mentioned_skills(body: str) -> set[str]:
    """Backticked names a Markdown body refers to as skills, parsed from its inline code spans.

    A span is a mention when the word before it is ``open``/``see``/``skill(s)``, the word after it is
    ``skill(s)``, or it continues a comma/and/or list that began with a mention.
    """
    pieces = body.split("`")
    found: set[str] = set()
    previous_mention = False
    for index in range(1, len(pieces), 2):
        span, before = pieces[index].strip(), pieces[index - 1]
        after = pieces[index + 1] if index + 1 < len(pieces) else ""
        before_words = before.lower().replace("(", " ").replace(":", " ").split()
        after_words = after.lower().replace(",", " ").replace(".", " ").split()
        continues = previous_mention and "".join(before.split()) in _LIST_SEPARATORS
        mention = _is_name(span) and (continues or (before_words and before_words[-1] in _BEFORE)
                                      or (after_words and after_words[0] in ("skill", "skills")))
        if mention:
            found.add(span)
        previous_mention = bool(mention)
    return found


def test_mention_parser_reads_open_see_lists_and_trailing_skill():
    body = ("Open `a-one`, `b-two` and `c-three` for this; see `d-four`. The `e-five` skill helps. Run "
            "`python x.py --cluster 2`, then `qa/verdict.json`; load with `json`.")
    assert mentioned_skills(body) == {"a-one", "b-two", "c-three", "d-four", "e-five"}


def test_every_skill_a_body_points_at_exists_or_is_pending_merge():
    skills = SK.collect()
    names = set(skills)
    for name, files in skills.items():
        _, body = SK.frontmatter(files[SK.SKILL_FILE].decode("utf-8"))
        missing = mentioned_skills(body) - names - PENDING_SKILLS
        assert not missing, f"{name} points at skills that do not exist: {sorted(missing)}"
    _, body = SK.frontmatter(skills["fix-failure-cluster"][SK.SKILL_FILE].decode("utf-8"))
    assert {"delegate-to-helpers", "read-command-buffer"} <= mentioned_skills(body)


def test_skill_sources_name_no_target_and_scripts_use_no_regex():
    gate = _gate()
    names = gate.TARGET_NAMES | gate.SUBSTRATE_NAMES
    for name, files in SK.collect().items():
        for relative, data in files.items():
            text = data.decode("utf-8")
            assert gate._name_in(text, names) is None, (name, relative, gate._name_in(text, names))
            if relative.endswith(".py"):
                imported = {alias.name.split(".")[0] for node in ast.walk(ast.parse(text))
                            if isinstance(node, (ast.Import, ast.ImportFrom))
                            for alias in getattr(node, "names", ())}
                imported |= {node.module.split(".")[0] for node in ast.walk(ast.parse(text))
                             if isinstance(node, ast.ImportFrom) and node.module}
                # A script may import the author's own editable compiler library (``merlin``, from
                # submission/compiler_library), never the harness that grades it.
                assert "re" not in imported and "merlin_experiments" not in imported, (name, relative)


def test_both_drivers_get_identical_bytes_and_runnable_scripts(tmp_path):
    ws, run, staged = _stage(tmp_path)
    trees = {}
    for driver, root in SK.ROOTS.items():
        base = ws / root
        trees[driver] = {path.relative_to(base).as_posix(): path.read_bytes()
                         for path in sorted(base.rglob("*")) if path.is_file()}
    codex, claude = trees["codex"], trees["claudecode"]
    assert codex == claude and set(path.split("/")[0] for path in codex) == set(SK.collect())
    assert len(staged["files"]) == 2 * len(codex) and staged["sha256"] == SK.digest(SK.collect())
    scripts = sorted(path for path in (ws / SK.ROOTS["claudecode"]).rglob("scripts/*.py"))
    assert scripts, "at least one skill ships a script"
    for script in scripts:
        assert os.access(script, os.R_OK | os.X_OK)
        compiled = subprocess.run([sys.executable, "-B", "-m", "py_compile", str(script)], capture_output=True,
                                  text=True, check=False)
        assert compiled.returncode == 0, compiled.stderr
        # -I: stdlib only, independent of the harness's PYTHONPATH, run from the workspace as the author does.
        helped = subprocess.run([sys.executable, "-B", "-I", str(script.relative_to(ws)), "--help"], cwd=ws,
                                capture_output=True, text=True, check=False, timeout=60)
        assert helped.returncode == 0 and "usage" in helped.stdout.lower(), (script, helped.stderr)


def test_staging_seals_the_files_and_refuses_overwrites(tmp_path):
    ws, run, bundle = tmp_path / "ws", tmp_path / "run", tmp_path / "bundle"
    for path in (ws, run, bundle):
        path.mkdir()
    for path in (ws / "TASK.md", run / "TASK.md"):
        path.write_text(TASK)
    for path in (bundle / "input_bundle_manifest.yaml", run / "input_bundle_manifest.yaml"):
        path.write_text("allowed: []\n")
    before = RI.treatment_snapshot_record(ws, run, bundle, ["tool"])
    record = SK.selection(parse_options(CODEX))
    # With author skills the generated feedback guide is left out; feedback-files carries it.
    disclosure = task_disclosure.stage(ws, run, feedback=False)
    assert disclosure["feedback_skill"] is False
    assert task_disclosure.FEEDBACK_SKILL not in {row["name"] for row in disclosure["skills"]}
    staged = SK.stage(ws, run, record)
    names = sorted(path.name for path in (ws / ".agents" / "skills").iterdir())
    assert names == sorted([*(row["name"] for row in disclosure["skills"]), *SK.collect()])
    sealed = RI.treatment_snapshot_record(ws, run, bundle, ["tool"])
    rows = {row["name"] for row in sealed["files"]}
    assert f"archived/{SK.RECORD}" in rows and {f"served/{row['path']}" for row in staged["files"]} <= rows
    assert len(sealed["files"]) == len(before["files"]) + 1 + len(disclosure["skills"]) + 1 + len(staged["files"])
    RI.verify_treatment_snapshot(sealed, ws, run, bundle, ["tool"])
    edited = ws / ".claude" / "skills" / SK.FEEDBACK_SKILL / SK.SKILL_FILE
    edited.write_text(edited.read_text() + "\nmore")
    with pytest.raises(RuntimeError, match="drifted"):
        RI.verify_treatment_snapshot(sealed, ws, run, bundle, ["tool"])
    with pytest.raises(RuntimeError, match="already exists"):
        SK.stage(tmp_path / "ws", tmp_path / "run", record)
    with pytest.raises(RuntimeError, match="changed between"):
        SK.stage(tmp_path / "other", run, {**record, "sha256": "0" * 64})
    (run / SK.RECORD).write_text('{"schema": "other"}')
    with pytest.raises(RuntimeError, match="malformed"):
        SK.staged_paths(ws, run)


def test_selection_defaults_on_for_native_skill_drivers_and_is_refused_elsewhere():
    for argv in (CODEX, CLAUDE, ["--run-id", "t"]):  # auto with an Anthropic model drives the claude CLI
        record = _record(parse_options(argv))[SK.KEY]
        assert record["enabled"] is True and record["source"] == "default" and set(OWN) <= set(record["skills"])
        assert record["sha256"] == SK.digest(SK.collect()) and record["roots"] == {
            "claudecode": ".claude/skills", "codex": ".agents/skills"}
    off = _record(parse_options([*CLAUDE, "--author-skills", "off"]))[SK.KEY]
    assert off == {"schema": SK.SCHEMA, "enabled": False, "source": "option"}
    converse = ["--run-id", "t", "--driver", "converse"]
    assert SK.KEY not in _record(parse_options(converse))
    assert SK.KEY not in _record(parse_options([*converse, "--author-skills", "off"]))
    with pytest.raises(SystemExit):
        parse_options([*converse, "--author-skills", "on"])
    # The Codex context names the guide the lead will find.
    assert CTX.selection(parse_options(CODEX))["feedback_skill"] == SK.FEEDBACK_SKILL
    note = CTX.round_note(CTX.selection(parse_options(CODEX)), None)
    assert f"with {SK.FEEDBACK_SKILL} describing" in note
    plain = CTX.selection(parse_options([*CODEX, "--author-skills", "off"]))
    assert "feedback_skill" not in plain
    assert f"with {task_disclosure.FEEDBACK_SKILL} describing" in CTX.round_note(plain, None)


def test_resume_keeps_the_record_and_refuses_an_explicit_change():
    options = parse_options(CLAUDE)
    record = _record(options)
    SK.verify_resume(options, record)
    SK.verify_resume(dataclasses.replace(options, author_skills="on"), record)
    with pytest.raises(RuntimeError, match="author skills selection changed"):
        SK.verify_resume(dataclasses.replace(options, author_skills="off"), record)
    legacy = {key: value for key, value in record.items() if key != SK.KEY}
    SK.verify_resume(options, legacy)  # launched before the setting: omitted option keeps it
    with pytest.raises(RuntimeError, match="author skills selection changed"):
        SK.verify_resume(dataclasses.replace(options, author_skills="on"), legacy)
    codex = parse_options(CODEX)
    with pytest.raises(RuntimeError, match="author skills selection changed"):
        session._verify_client_selection(dataclasses.replace(codex, author_skills="off"),
                                         {"run_config": _record(codex)})


def test_launch_config_renders_the_option_for_skill_drivers_only(tmp_path):
    import shutil

    launch_path = "examples/gemmini/phase1/launch/el4-fire-aligned.yaml"
    release_path = "examples/gemmini/phase0/release/fire-aligned-v10.yaml"

    def checkout(name, edit=None):
        root = tmp_path / name
        for relative in (Path(launch_path).parent, Path(release_path).parent):
            shutil.copytree(repo_root() / relative, root / relative)
        if edit is not None:
            document = yaml.safe_load((root / launch_path).read_text())
            edit(document)
            (root / launch_path).write_text(yaml.safe_dump(document, sort_keys=False))
        return root / launch_path

    def argv(path):
        return lc.build(lc.load(path), task_root=tmp_path, prep=tmp_path, environ={"HOME": "/h"}).argv

    assert "--author-skills" not in argv(checkout("plain"))
    rendered = argv(checkout("off", lambda d: d["author"].update(skills=False)))
    assert rendered[rendered.index("--author-skills") + 1] == "off"
    assert set(lc.SKILL_FLAGS) == set(SK.ROOTS)
    with pytest.raises(lc.ConfigError, match="skills"):
        lc.load(checkout("bad", lambda d: d["author"].update(skills="yes")))
