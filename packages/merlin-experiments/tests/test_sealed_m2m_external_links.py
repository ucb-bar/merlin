"""Base-only external links bind text without importing host /etc or /dev bytes."""

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import readonly_runtime, runtime_store
from merlin_experiments.capture_execution import sealed_m2m as M
from merlin_experiments.capture_execution import sealed_static as S


_EXTERNAL_LINKS = {
    "share/X11/rgb.txt": "/etc/X11/rgb.txt",
    "share/locale/locale.alias": "/etc/locale.alias",
    "share/groff/site-tmac": "/etc/groff",
    "lib/ssl/private": "/etc/ssl/private",
    "lib/ssl/certs": "/etc/ssl/certs",
    "lib/ssl/openssl.cnf": "/etc/ssl/openssl.cnf",
    "lib/ssl/cert.pem": "/etc/ssl/certs/ca-certificates.crt",
    "lib/environment.d/99-environment.conf": "/etc/environment",
    "lib/python3.12/sitecustomize.py": "/etc/python3.12/sitecustomize.py",
    **{
        f"lib/systemd/system/{name}.service": "/dev/null"
        for name in (
            "hwclock", "sudo", "cryptdisks-early", "x11-common", "cryptdisks",
            "screen-cleanup", "multipath-tools-boot",
        )
    },
}


def _inventory(root):
    return M._source_tree(root, allow_dangling_symlinks=True)


def _external_tree(tmp_path, *, directory, absolute):
    outside = tmp_path / "outside"
    if directory:
        outside.mkdir()
        (outside / "secret").write_bytes(b"host-only content")
    else:
        outside.write_bytes(b"host-only content")
    root = tmp_path / "usr"
    root.mkdir()
    target = str(outside) if absolute else "../outside"
    (root / "alias").symlink_to(target)
    return root, outside, target


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_external_link_text_survives_inventory_copy_and_guest(tmp_path, directory, absolute):
    root, outside, target = _external_tree(tmp_path, directory=directory, absolute=absolute)
    selected = _inventory(root)
    assert selected["bytes"] == 0
    assert selected["members"] == 2  # No children of the external directory.
    expected = {"kind": "external_symlink", "target": target}
    assert M._tree(root, allow_dangling_symlinks=True)["alias"] == expected
    staged = tmp_path / "staged"
    M._copy_source_tree(root, staged, allow_dangling_symlinks=True)
    guest = tmp_path / "guest"
    runtime_store._link_entry(staged, guest)
    for tree in (staged, guest):
        assert os.readlink(tree / "alias") == target
        assert M._snapshot_tree(tree, allow_dangling_symlinks=True) == selected
    # Host content and liveness are not part of an opaque link's identity.
    if directory:
        shutil.rmtree(outside)
    else:
        outside.unlink()
    assert _inventory(root) == selected
    alias = root / "alias"
    alias.unlink()
    alias.symlink_to(target.rsplit("/", 1)[0] + "/./" + target.rsplit("/", 1)[1])
    assert _inventory(root) != selected
    assert M._snapshot_tree(root, allow_dangling_symlinks=True) != selected


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("operation", ["source", "snapshot", "copy", "store"])
def test_external_metadata_never_resolves_or_stats_target(tmp_path, monkeypatch, directory, operation):
    root, _, target = _external_tree(tmp_path, directory=directory, absolute=True)
    alias = root / "alias"
    readlink, original_stat = os.readlink, Path.stat
    observed = []

    def observe(path):
        observed.append(Path(path))
        return readlink(path)

    def guarded_stat(path, *args, **kwargs):
        assert path != alias or kwargs.get("follow_symlinks") is False
        return original_stat(path, *args, **kwargs)

    def refuse_resolve(*args, **kwargs):
        pytest.fail("external metadata must not resolve a target")

    monkeypatch.setattr(os, "readlink", observe)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "resolve", refuse_resolve)
    if operation == "source":
        assert _inventory(root)["bytes"] == 0
    elif operation == "snapshot":
        assert M._tree(root, allow_dangling_symlinks=True)["alias"]["target"] == target
    elif operation == "copy":
        M._copy_source_tree(root, tmp_path / "copied", allow_dangling_symlinks=True)
    else:
        runtime_store._link_entry(root, tmp_path / "guest")
    assert observed == [alias]


@pytest.mark.parametrize("directory", [False, True])
def test_source_and_output_scans_still_refuse_external_links(tmp_path, directory):
    root, _, _ = _external_tree(tmp_path, directory=directory, absolute=True)
    with pytest.raises(ValueError, match="link"):
        M._source_tree(root)
    for scan in (M._tree, M._snapshot_tree, S._tree):
        with pytest.raises(ValueError, match="link"):
            scan(root)
    # Allowing internal directory aliases does not enable base metadata policy.
    with pytest.raises(ValueError, match="link"):
        S._tree(root, allow_directory_symlinks=True)


@pytest.mark.parametrize("directory", [False, True])
def test_selected_closure_member_through_external_link_is_refused(tmp_path, directory):
    source, runtime = tmp_path / "source", tmp_path / "runtime"
    roots = {
        "venv": runtime / "opt/capture-venv", "base": runtime / "usr",
        "m2m": source / "m2m-src/m2m", "workload": source / "workload",
        "schemas": source / "merlin-src/merlin/_data/schemas",
    }
    for root in roots.values():
        root.mkdir(parents=True)
    closure = roots["venv"] / "selected"
    outside = tmp_path / "host"
    if directory:
        closure.mkdir()
        (closure / "member.py").write_bytes(b"selected bytes")
        shutil.copytree(closure, outside)
    else:
        closure.write_bytes(b"selected bytes")
        shutil.copy2(closure, outside)
    plan = {
        "base": "/usr", "schemas_root": "/separate-schemas", "merlin_root": "/merlin",
        "selected_trees": {name: M._source_tree(root) for name, root in roots.items()},
    }
    if directory:
        shutil.rmtree(closure)
    else:
        closure.unlink()
    closure.symlink_to(outside)
    with pytest.raises(M.SealedM2MError, match="staged venv bytes differ"):
        M._verify_staged_selection(
            plan, source, runtime, source_rows=M._tree(source),
            runtime_rows=M._tree(runtime, allow_dangling_symlinks=True),
        )
    with pytest.raises(M.SealedM2MError, match="link"):
        M._source_tree(roots["venv"])


def test_compute_1_external_link_census_round_trip(tmp_path):
    root = tmp_path / "usr"
    for member, target in _EXTERNAL_LINKS.items():
        link = root / member
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    real = root / "real"
    real.mkdir()
    (real / "selected.py").write_text("value = 1\n")
    (root / "chain1").symlink_to("real")
    (root / "chain2").symlink_to("chain1")
    (root / "chain3").symlink_to("chain2/../chain1")
    (root / "chain4").symlink_to("chain3")
    (root / "chain5").symlink_to("chain4")
    unreadable = root / "share/chrony/chrony.keys"
    unreadable.parent.mkdir(parents=True)
    unreadable.write_bytes(b"never copied")
    unreadable.chmod(0)
    assert len(_EXTERNAL_LINKS) == 16
    selected = _inventory(root)
    staged, guest = tmp_path / "staged", tmp_path / "guest"
    M._copy_source_tree(root, staged, allow_dangling_symlinks=True)
    runtime_store._link_entry(staged, guest)
    for tree in (root, staged, guest):
        rows = M._tree(tree, allow_dangling_symlinks=True)
        assert rows["share/chrony/chrony.keys"]["kind"] == "unreadable_file"
        for name, target in _EXTERNAL_LINKS.items():
            assert rows[name] == {"kind": "external_symlink", "target": target}
        if tree is not root:
            assert not os.path.lexists(tree / "share/chrony/chrony.keys")
        assert _inventory(tree) == selected
        assert M._snapshot_tree(tree, allow_dangling_symlinks=True) == selected
        assert {name: os.readlink(tree / name) for name in _EXTERNAL_LINKS} == _EXTERNAL_LINKS


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("overlay_version", [None, 2, 3])
@pytest.mark.parametrize("absolute", [False, True])
def test_external_base_links_survive_runtime_store_and_readonly_reuse(
    tmp_path, monkeypatch, directory, overlay_version, absolute,
):
    base, _, target = _external_tree(tmp_path, directory=directory, absolute=absolute)
    venv = tmp_path / "venv"
    venv.mkdir()
    plan = {
        "venv": str(venv), "base": str(base), "system_libs": [],
        "selected_trees": {"venv": M._source_tree(venv), "base": _inventory(base)},
    }
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    entry = runtime_store.store_entry(plan)
    cached = runtime_store.verified_cached_entry(plan, tmp_path)
    assert cached is not None
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime_store.link_runtime(plan, runtime, verified_cache=cached)
    assert os.readlink(runtime / base.relative_to("/") / "alias") == target
    options = {}
    if overlay_version is not None:
        predicted = tmp_path / "predicted"
        overlay = predicted / "nested/code.py"
        overlay.parent.mkdir(parents=True)
        overlay.write_text("selected = True\n")
        plan["selected_trees"]["venv"] = M._source_tree(predicted)
        options = {"overlays": {"opt/capture-venv/nested/code.py": overlay}, "overlay_version": overlay_version}
    marker = entry.parent / (entry.name + ".complete.json")
    selection = readonly_runtime.select(
        plan, entry=entry, marker_sha256=S._file_digest(marker), destination=tmp_path, **options,
    )
    assert readonly_runtime.verify(plan, selection, tmp_path) == selection


def test_external_sitecustomize_is_not_executed_by_sealed_python_startup(tmp_path):
    # Relocate a real interpreter so its real startup search includes our link.
    base = tmp_path / "usr"
    python = base / "bin/python"
    python.parent.mkdir(parents=True)
    shutil.copy2(sys._base_executable, python)
    stdlib = base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    shutil.copytree(
        sysconfig.get_path("stdlib"), stdlib,
        ignore=shutil.ignore_patterns("site-packages", "__pycache__", "test", "tests", "tkinter", "idlelib", "ensurepip"),
    )
    marker = tmp_path / "sitecustomize-executed"
    target = tmp_path / "etc/sitecustomize.py"
    target.parent.mkdir()
    target.write_text(f"open({str(marker)!r}, 'w').write('host startup executed')\n")
    link = stdlib / "sitecustomize.py"
    link.unlink(missing_ok=True)
    link.symlink_to(target)
    assert S._base_symlink(link, base) == {"kind": "external_symlink", "target": str(target)}
    commands = (M._command(Path('/capture-out')), M._command_v2(Path('/capture-out'), dtype='fp32', recipe=False))
    for command in commands:
        assert command[1:5] == ("-I", "-S", "-B", "-c")
        result = subprocess.run(
            [str(python), *command[1:5], "import sys; assert sys.flags.no_site and 'sitecustomize' not in sys.modules"],
            env={}, capture_output=True, text=True, check=True,
        )
        assert not result.stderr
        assert not marker.exists()
    # Positive control proves this exact external link executes without -S.
    subprocess.run([str(python), '-I', '-B', '-c', 'pass'], env={}, capture_output=True, check=True)
    assert marker.read_text() == 'host startup executed'


@pytest.mark.parametrize("target", ["chain1", "chain1/../chain1", "./chain1"])
def test_internal_directory_chain_is_lexical_and_not_walked(tmp_path, target):
    root = tmp_path / "usr"
    (root / "real").mkdir(parents=True)
    (root / "real/data").write_bytes(b"bound target bytes")
    (root / "chain1").symlink_to("real")
    (root / "chain2").symlink_to(target)
    rows = M._tree(root, allow_dangling_symlinks=True)
    assert rows["chain2"] == {"kind": "symlink", "target": target}
    assert "chain2/data" not in rows
    selected = _inventory(root)
    copied = tmp_path / "copy"
    M._copy_source_tree(root, copied, allow_dangling_symlinks=True)
    assert os.readlink(copied / "chain2") == target
    assert M._snapshot_tree(copied, allow_dangling_symlinks=True) == selected
    (root / "chain2").unlink()
    (root / "chain2").symlink_to("./" + target)
    assert _inventory(root) != selected
    for scan in (M._source_tree, M._snapshot_tree):
        with pytest.raises(ValueError, match="symlink component"):
            scan(root)


@pytest.mark.parametrize("operation", ["source", "snapshot", "copy", "store"])
@pytest.mark.parametrize("target", ["b", "a/tail"])
def test_base_cycles_are_refused(tmp_path, operation, target):
    root = tmp_path / "usr"
    root.mkdir()
    (root / "a").symlink_to(target)
    (root / "b").symlink_to("a")
    with pytest.raises(ValueError, match="cyclic"):
        if operation == "source":
            _inventory(root)
        elif operation == "snapshot":
            M._snapshot_tree(root, allow_dangling_symlinks=True)
        elif operation == "copy":
            M._copy_source_tree(root, tmp_path / "copy", allow_dangling_symlinks=True)
        else:
            runtime_store._link_entry(root, tmp_path / "guest")


def test_internal_chain_that_escapes_is_external_without_host_reads(tmp_path, monkeypatch):
    root = tmp_path / "usr"
    root.mkdir()
    (root / "a").symlink_to("b/secret")
    (root / "b").symlink_to("../outside")
    original = Path.lstat

    def guarded(path, *args, **kwargs):
        assert path.is_relative_to(root)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", guarded)
    rows = M._tree(root, allow_dangling_symlinks=True)
    assert rows["a"]["kind"] == rows["b"]["kind"] == "external_symlink"
    assert rows["a"]["target"] == "b/secret"


def test_unreadable_metadata_is_bound_and_absent_in_guest(tmp_path):
    root = tmp_path / "usr"
    root.mkdir()
    member = root / "chrony.keys"
    member.write_bytes(b"host secret never copied")
    member.chmod(0)
    info = member.stat()
    expected = {"kind": "unreadable_file", "mode": 0, "size": info.st_size,
                "uid": info.st_uid, "gid": info.st_gid}
    assert M._tree(root, allow_dangling_symlinks=True)["chrony.keys"] == expected
    selected = _inventory(root)
    staged, guest = tmp_path / "staged", tmp_path / "guest"
    M._copy_source_tree(root, staged, allow_dangling_symlinks=True)
    runtime_store._link_entry(staged, guest)
    for tree in (staged, guest):
        # Never materialized: a guest open() fails with ENOENT.
        assert not os.path.lexists(tree / "chrony.keys")
        assert "chrony.keys" not in os.listdir(tree)
        assert M._tree(tree, allow_dangling_symlinks=True)["."]["members"] == ["chrony.keys"]
        assert M._snapshot_tree(tree, allow_dangling_symlinks=True) == selected
        assert M._tree(tree, allow_dangling_symlinks=True)["chrony.keys"] == expected
        for scan in (M._source_tree, M._snapshot_tree, S._tree):
            with pytest.raises(ValueError, match="unreadable"):
                scan(tree)
    member.chmod(0o200)
    assert _inventory(root) != selected
    assert M._snapshot_tree(root, allow_dangling_symlinks=True) != selected
    for scan in (M._source_tree, M._snapshot_tree, S._tree):
        with pytest.raises(ValueError, match="unreadable"):
            scan(root)


@pytest.mark.parametrize("kind", ["chain", "unreadable"])
@pytest.mark.parametrize("member", [
    "bin/python", "pyvenv.cfg", "lib/python3.12/site-packages/startup.pth",
    "package/direct_import.py", "package/dependency.so", "data/weights.bin",
])
def test_staged_closure_rejects_metadata_substitution(tmp_path, kind, member):
    source, runtime = tmp_path / "source", tmp_path / "runtime"
    roots = {
        "venv": runtime / "opt/capture-venv", "base": runtime / "usr",
        "m2m": source / "m2m-src/m2m", "workload": source / "workload",
        "schemas": source / "merlin-src/merlin/_data/schemas",
    }
    for root in roots.values():
        root.mkdir(parents=True)
    selected = roots["venv"] / member
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_bytes(b"selected closure bytes")
    plan = {"base": "/usr", "schemas_root": "/separate", "merlin_root": "/merlin",
            "selected_trees": {name: M._source_tree(root) for name, root in roots.items()}}
    if kind == "chain":
        real = roots["venv"] / "real"
        real.mkdir()
        selected.unlink()
        selected.symlink_to(os.path.relpath(real, selected.parent))
        alias = roots["venv"] / "alias"
        alias.symlink_to("real")
        selected.unlink()
        selected.symlink_to(os.path.relpath(alias, selected.parent))
    else:
        selected.chmod(0)
    with pytest.raises(ValueError):
        M._source_tree(roots["venv"])
    with pytest.raises(M.SealedM2MError, match="staged venv bytes differ"):
        M._verify_staged_selection(plan, source, runtime, source_rows=M._tree(source),
                                  runtime_rows=M._tree(runtime, allow_dangling_symlinks=True))


def test_guest_mounts_keep_etc_absent_and_dev_minimal(tmp_path):
    argv = M._sandbox_argv(Path("/usr/bin/bwrap"), tmp_path / "runtime", tmp_path / "source",
                           tmp_path / "out", ("/bin/true",), Path("/capture-out"))
    assert argv[argv.index("--dev") + 1] == "/dev"
    assert "/etc" not in argv


def test_absolute_internal_directory_chain_survives_base_relocation(tmp_path):
    root = tmp_path / "usr"
    (root / "real").mkdir(parents=True)
    (root / "real/data").write_bytes(b"bound bytes")
    (root / "alias").symlink_to(root / "real")
    (root / "chain").symlink_to("alias")
    selected = _inventory(root)
    staged = tmp_path / "staged/usr"
    M._copy_source_tree(root, staged, allow_dangling_symlinks=True)
    assert M._snapshot_tree(staged, allow_dangling_symlinks=True) == selected
    guest = tmp_path / "guest"
    runtime_store._link_entry(staged.parent, guest)
    assert M._snapshot_tree(guest / "usr", allow_dangling_symlinks=True) == selected
    assert os.readlink(guest / "usr/alias") == str(root / "real")


@pytest.mark.parametrize("kind", ["unreadable", "chain", "external"])
@pytest.mark.parametrize("tree", [False, True])
def test_declared_input_refuses_base_metadata(tmp_path, kind, tree):
    root = tmp_path / "input"
    root.mkdir()
    member = root / "data"
    if kind == "unreadable":
        member.write_bytes(b"secret")
        member.chmod(0)
    elif kind == "chain":
        (root / "real").mkdir()
        (root / "alias").symlink_to("real")
        member.symlink_to("alias")
    else:
        member.symlink_to("/etc/environment")
    with pytest.raises(ValueError):
        M._input_selection(root if tree else member, "model")


@pytest.mark.parametrize("kind", ["external", "chain", "unreadable"])
def test_selected_base_interpreter_refuses_metadata(tmp_path, kind):
    base = tmp_path / "usr"
    binary = base / "bin/python3.12"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"selected interpreter")
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\n")
    if kind == "external":
        outside = tmp_path / "host-python"
        binary.rename(outside)
        binary.symlink_to(outside)
    elif kind == "chain":
        binary.rename(base / "bin/real")
        (base / "bin/alias").symlink_to("real")
        binary.symlink_to("alias")
    else:
        binary.chmod(0)
    with pytest.raises(M.SealedM2MError, match="symlink component|unreadable"):
        M._venv_home(venv)


@pytest.mark.parametrize("field", ["mode", "size", "uid", "gid"])
def test_omitted_unreadable_marker_digest_binds_original_metadata(tmp_path, field):
    import json
    root = tmp_path / "usr"
    root.mkdir()
    source = root / "secret"
    source.write_bytes(b"host secret")
    source.chmod(0)
    staged = tmp_path / "staged"
    M._copy_source_tree(root, staged, allow_dangling_symlinks=True)
    selected = _inventory(staged)
    assert selected == _inventory(root)
    attribute = S._unreadable_xattr("secret")
    row = json.loads(os.getxattr(staged, attribute))
    row[field] += 1
    os.setxattr(staged, attribute, json.dumps(row).encode())
    assert _inventory(staged) != selected
    assert M._snapshot_tree(staged, allow_dangling_symlinks=True) != selected


@pytest.mark.parametrize("substitute", ["file", "link"])
def test_omitted_unreadable_member_cannot_reappear_with_bytes(tmp_path, substitute):
    root = tmp_path / "usr"
    root.mkdir()
    source = root / "secret"
    source.write_bytes(b"host secret")
    source.chmod(0)
    staged = tmp_path / "staged"
    M._copy_source_tree(root, staged, allow_dangling_symlinks=True)
    if substitute == "file":
        (staged / "secret").write_bytes(b"unbound bytes")
    else:
        (staged / "secret").symlink_to("/etc/environment")
    for scan in (_inventory, lambda tree: M._snapshot_tree(tree, allow_dangling_symlinks=True)):
        with pytest.raises(ValueError, match="omitted unreadable member is present"):
            scan(staged)


def test_forged_omission_marker_is_refused(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    os.setxattr(staged, S._unreadable_xattr("other"), S._json(
        {"kind": "unreadable_file", "name": "secret", "mode": 0, "size": 1, "uid": 0, "gid": 0}))
    with pytest.raises(ValueError, match="invalid omitted"):
        _inventory(staged)


def test_strict_scans_refuse_omission_markers(tmp_path):
    root = tmp_path / "usr"
    root.mkdir()
    (root / "secret").write_bytes(b"host secret")
    (root / "secret").chmod(0)
    staged = tmp_path / "staged"
    M._copy_source_tree(root, staged, allow_dangling_symlinks=True)
    for scan in (M._source_tree, M._snapshot_tree, S._tree):
        with pytest.raises(ValueError, match="unreadable"):
            scan(staged)
    with pytest.raises(ValueError, match="unreadable"):
        M._copy_source_tree(staged, tmp_path / "again")


@pytest.mark.parametrize("kind", ["external", "dangling", "chain", "unreadable"])
def test_selected_system_library_refuses_base_metadata(tmp_path, kind):
    base = tmp_path / "usr"
    lib = base / "lib/x86_64-linux-gnu"
    lib.mkdir(parents=True)
    real = lib / "libreal.so.1"
    real.write_bytes(b"\x7fELF selected")
    selected = lib / "libselected.so.1"
    if kind == "external":
        outside = tmp_path / "etc/alternatives/libselected.so.1"
        outside.parent.mkdir(parents=True)
        outside.symlink_to(real)  # live on the host, but only via an external hop
        selected.symlink_to(outside)
    elif kind == "dangling":
        selected.symlink_to("libabsent.so.1")
    elif kind == "chain":
        (base / "lib/alias1").symlink_to("x86_64-linux-gnu")
        (base / "lib/alias2").symlink_to("alias1")
        selected = base / "lib/alias2/libreal.so.1"
    else:
        selected.write_bytes(b"secret")
        selected.chmod(0)
    with pytest.raises(M.SealedM2MError, match="selected closure path"):
        M._require_plain_base_resolution(selected, base)


def test_selected_system_library_plain_aliases_are_accepted(tmp_path):
    base = tmp_path / "usr"
    lib = base / "lib/x86_64-linux-gnu"
    lib.mkdir(parents=True)
    (lib / "libz.so.1.3").write_bytes(b"\x7fELF")
    (lib / "libz.so.1").symlink_to("libz.so.1.3")
    top = tmp_path / "lib"
    top.symlink_to("usr/lib")  # host top-level alias outside the base
    M._require_plain_base_resolution(top / "x86_64-linux-gnu/libz.so.1", base)
    M._require_plain_base_resolution(lib / "libz.so.1", base)


def test_runtime_store_refuses_library_write_through_base_link(tmp_path, monkeypatch):
    base = tmp_path / "usr"
    (base / "lib").mkdir(parents=True)
    victim = tmp_path / "host-victim"
    (base / "lib/libx.so").symlink_to(victim)
    library = tmp_path / "selected-lib.so"
    library.write_bytes(b"lib")
    venv = tmp_path / "venv"
    venv.mkdir()
    plan = {"venv": str(venv), "base": str(base), "system_libs": [str(base / "lib/libx.so")],
            "selected_trees": {"venv": M._source_tree(venv), "base": _inventory(base)}}
    monkeypatch.setattr(runtime_store.shutil, "copy2", lambda *a, **k: pytest.fail("wrote through link"))
    with pytest.raises(M.SealedM2MError, match="staged base link"):
        runtime_store._build(plan, tmp_path / "dest")
    assert not victim.exists()
