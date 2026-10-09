"""Real unprivileged controls against the explicitly selected public manager."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    root = Path(fixture["root"])
    native_events = []

    def observe(event, values):
        if event in ("socket.connect", "socket.connect_ex", "socket.sendto", "socket.bind"):
            raise RuntimeError("network attempted outside local source control scope")
        if event == "subprocess.Popen":
            native_events.append({"program": str(values[0]), "argv": [str(token) for token in values[1]]})

    sys.addaudithook(observe)
    from fabric.api import cd, execute, prefix, settings, shell_env, warn_only
    from runtools.firesim_topology_with_passes import instance_liveness
    from runtools.runtime_config import RuntimeConfig
    from runtools.utils import check_script
    from util import local_transport as T

    declarations = []

    def configuration(name, *, mode="local", hostname="localhost", changes=None):
        recipe = yaml.safe_load(Path(fixture["recipe"]).read_text())
        spec = recipe["args"]["run_farm_host_specs"][0]
        spec = next(iter(spec.values()))
        if mode is not None:
            spec.update(
                command_transport=mode,
                local_command_environment={"PATH": "/usr/bin:/bin", "HOME": str(root / "empty-home"), "LANG": "C"},
                local_command_timeout_seconds=5,
            )
        if changes:
            spec.update(changes)
        key = next(iter(recipe["args"]["run_farm_hosts_to_use"][0].values()))
        recipe["args"]["run_farm_hosts_to_use"] = [{hostname: key}]
        recipe_path = root / (name + "-recipe.yaml")
        recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
        runtime = yaml.safe_load(Path(fixture["runtime"]).read_text())
        runtime["run_farm"]["base_recipe"] = str(recipe_path)
        runtime["run_farm"]["recipe_arg_overrides"]["default_simulation_dir"] = str(root / "simulation")
        path = root / (name + "-runtime.yaml")
        path.write_text(yaml.safe_dump(runtime, sort_keys=False))
        declarations.extend((str(recipe_path), str(path)))
        return RuntimeConfig(
            argparse.Namespace(
                hwdbconfigfile=fixture["hwdb"],
                runtimeconfigfile=str(path),
                overrideconfigdata="",
                task="runworkload",
                buildrecipesconfigfile=fixture["build_recipes"],
            )
        )

    outcomes = []

    def passed(name):
        outcomes.append({"control": name, "outcome": "passed original assertion"})

    def refused(name, action, kind=ValueError, message=None):
        try:
            action()
        except kind as error:
            if message is not None and message not in str(error):
                raise AssertionError(name + " has no original defect attribution") from error
            passed(name)
        else:
            raise AssertionError(name + " did not refuse")

    baseline = configuration("default", mode=None)
    hosts = baseline.run_farm.get_all_bound_host_nodes()
    assert len(hosts) == 1
    server = hosts[0].sim_slots[0]
    baseline_roster = (
        hosts[0].get_host(),
        len(hosts[0].sim_slots),
        server.get_job_name(),
        server.get_bootbin_name(),
        server.get_required_files_local_paths(),
    )
    with settings(host_string="localhost"), mock.patch.object(T.ssh, "run", return_value="original-ssh") as original:
        assert T.run("declared-default", timeout=1) == "original-ssh"
        original.assert_called_once_with("declared-default", timeout=1)
    passed("default SSH command delegation unchanged; no SSH executed")
    selected = configuration("local")
    host = selected.run_farm.get_all_bound_host_nodes()[0]
    server = host.sim_slots[0]
    assert baseline_roster == (
        host.get_host(),
        len(host.sim_slots),
        server.get_job_name(),
        server.get_bootbin_name(),
        server.get_required_files_local_paths(),
    )
    passed("actual complete host slot job ELF membership unchanged")
    with settings(host_string="localhost"):
        result = T.run("printf ordinary-local", quiet=True)
        assert str(result) == "ordinary-local" and result.succeeded and result.return_code == 0
        passed("actual local process stdout and status")
        result = T.run("printf stdout; printf stderr >&2", combine_stderr=False, quiet=True)
        assert str(result) == "stdout" and result.stderr == "stderr"
        passed("separate stdout stderr")
        spaced = root / "directory with spaces"
        spaced.mkdir()
        with cd(str(spaced)), prefix("printf prefix-"), shell_env(CONTROL_VALUE="declared"):
            result = T.run('printf "%s:%s" "$CONTROL_VALUE" "$PWD"', quiet=True)
        assert str(result) == "prefix-declared:" + str(spaced)
        passed("public cd prefix shell_env command contexts")
        result = T.run('printf "%s" "${SSH_AUTH_SOCK-unset}:${BASH_ENV-unset}:${HOME}"', quiet=True)
        assert str(result) == "unset:unset:" + str(root / "empty-home")
        passed("explicit environment without agent or shell startup inheritance")
        with warn_only():
            result = T.run("exit 7")
        assert result.failed and result.return_code == 7
        passed("original warn_only failure result")
        refused("ordinary nonzero process refusal", lambda: T.run("exit 9"), RuntimeError)
        refused("bounded direct-process timeout", lambda: T.run("sleep 3", timeout=0.1), subprocess.TimeoutExpired)
        for options in ({"pty": True}, {"stdin": None}, {"shell": False}, {"timeout": 0}):
            refused(
                "unsupported local run " + str(options), lambda options=options: T.run("printf forbidden", **options)
            )
        source = root / "source.bin"
        source.write_bytes(bytes(range(256)))
        source.chmod(0o640)
        destination = root / "destination with spaces.bin"
        transfer = T.put(str(source), str(destination), mirror_local_mode=True)
        assert transfer.succeeded and destination.read_bytes() == source.read_bytes()
        assert destination.stat().st_mode & 0o777 == 0o640
        received = root / "received"
        received.mkdir()
        T.get(str(destination), str(received))
        assert (received / destination.name).read_bytes() == source.read_bytes()
        passed("ordinary regular file put get and mode preservation")
        absent = root / "must-remain-absent"
        refused("unsupported privileged transfer", lambda: T.put(str(source), str(absent), use_sudo=True))
        refused("invalid transfer mode before side effect", lambda: T.put(str(source), str(absent), mode="bad"))
        assert not absent.exists()
        source_directory = root / "directory-source"
        source_directory.mkdir()
        (source_directory / "data.bin").write_bytes(b"declared-copy")
        target = root / "rsync-target"
        target.mkdir()
        T.rsync_project(str(target), str(source_directory) + "/", capture=True, ssh_opts="-o StrictHostKeyChecking=no")
        assert (target / "data.bin").read_bytes() == b"declared-copy"
        target_nested = root / "rsync-nested"
        target_nested.mkdir()
        T.rsync_project(str(target_nested), str(source_directory), capture=True)
        assert (target_nested / source_directory.name / "data.bin").read_bytes() == b"declared-copy"
        returned = root / "rsync-returned"
        returned.mkdir()
        T.rsync_project(str(target) + "/", str(returned), capture=True, upload=False)
        assert (returned / "data.bin").read_bytes() == b"declared-copy"
        passed("actual stock local rsync direction and trailing slash semantics")
        refused("remote rsync address", lambda: T.rsync_project("remote.invalid:/data", str(source_directory)))
        refused(
            "credentialed rsync option",
            lambda: T.rsync_project(str(target), str(source_directory), ssh_opts="-i excluded-key"),
        )
        refused(
            "unsupported rsync remote shell",
            lambda: T.rsync_project(str(target), str(source_directory), extra_opts="--rsh=ssh"),
        )
        comparison = root / "source-comparison"
        comparison.mkdir()
        script = root / "harmless-source-check"
        script.write_text("ordinary source bytes\n")
        (comparison / script.name).write_bytes(script.read_bytes())
        check_script(str(script), comparison)
        passed("actual unchanged stock check_script via local get")
        (comparison / script.name).write_text("genuine mismatching source\n")
        refused(
            "actual unchanged stock source mismatch",
            lambda: check_script(str(script), comparison),
            Exception,
            message="differs from the current FireSim version",
        )
    execute(instance_liveness, hosts=["localhost"])
    passed("actual public Fabric execute and ordinary instance_liveness over selected local host")
    with settings(host_string="remote.invalid"):
        refused("host absent from original local farm", lambda: T.run("printf forbidden"))
    original_selection = host._local_command_selection
    host._local_command_selection = (original_selection[0], original_selection[1] + 1)
    with settings(host_string="localhost"):
        refused("mutated original host selection", lambda: T.run("printf forbidden"))
    host._local_command_selection = original_selection
    for name, hostname, changes in (
        ("remote", "remote.invalid", None),
        ("alias", "127.0.0.1", None),
        (
            "credential environment",
            "localhost",
            {"local_command_environment": {"PATH": "/usr/bin:/bin", "HOME": str(root), "SSH_AUTH_SOCK": "/excluded"}},
        ),
        ("missing environment", "localhost", {"local_command_environment": None}),
        ("invalid timeout", "localhost", {"local_command_timeout_seconds": False}),
    ):
        refused(
            "unsupported actual manager configuration " + name,
            lambda name=name, hostname=hostname, changes=changes: configuration(
                name, hostname=hostname, changes=changes
            ),
        )
    report = {
        "schema": "private.public_manager_local_control.v1",
        "outcomes": outcomes,
        "complete_original_roster": list(baseline_roster),
        "actual_native_process_events": native_events,
        "created_configuration_files": declarations,
        "scope": "public manager host selection, unprivileged command and local file mechanisms only",
        "unknowns": [
            "privileged utilities",
            "terminal equivalence",
            "escaped descendants and cleanup",
            "hardware lifecycle",
            "runtime/effect authority",
            "physical timers",
            "dependency source builds",
        ],
    }
    Path(fixture["output"]).write_text(json.dumps(report, sort_keys=True, indent=2))
    print(json.dumps({"controls": len(outcomes), "actual_native_events": len(native_events), "scope": report["scope"]}))


if __name__ == "__main__":
    main()
