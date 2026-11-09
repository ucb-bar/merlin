"""Driver-owned native reservations and independent evidence beside Chia task refs."""

from __future__ import annotations

import json
import os
import tempfile
import time

from .chia_native import CleanupIncomplete, cleanup, setup


def local_options(resources: dict[str, float]) -> dict:
    """Explicit single-node deployment: never schedule onto an unmanaged peer."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_id = ray.get_runtime_context().get_node_id()
    nodes = [node for node in ray.nodes() if node.get("Alive") and node.get("NodeID") == node_id]
    required = {"CPU": 1, **resources}
    if len(nodes) != 1 or any(nodes[0].get("Resources", {}).get(key, 0) < count for key, count in required.items()):
        raise RuntimeError("managed native execution requires all task resources on the driver's own Ray node")
    return {
        "resources": dict(resources),
        "num_cpus": 1,
        "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
    }


class NativeTaskGroup:
    """Keep native cleanup distinct from task acknowledgement and upstream accounting.

    Enter inside chia_tasks and exit before its reference teardown/collector close.
    The caller also owns Session's outer context for pre-dispatch rollback.
    """

    def __init__(self, run, session):
        self.run, self.session = run, session
        self.path = run.run_dir / "chia/native.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owned = []
        self.document = {
            "schema": "merlin.native-task-group.v1",
            "source": session.source,
            "source_scope": "supervisor-and-guardian-only",
            "worker_imports": "not_frozen_by_this_adapter",
            "tasks": [],
            "cleanup_errors": [],
            "cleanup_complete": False,
        }
        with self.path.open("x") as stream:
            json.dump(self.document, stream, indent=2)

    def _write(self):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
                temporary = stream.name
                json.dump(self.document, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)

    def submit(self, task_owner, launch, /, *args, run_id):
        invitation = self.session.reserve()
        row = {"run_id": run_id, "invocation": invitation.invocation, "state": "reserved"}
        self._owned.append((invitation, row))
        self.document["tasks"].append(row)
        self._write()
        ref = task_owner.submit(
            launch,
            *args,
            _chia_setup=setup,
            _chia_setup_args=(invitation,),
            _chia_cleanup=cleanup,
            _chia_cleanup_args=(invitation,),
        )
        row.update(task_ref=ref.hex(), state="submitted")
        self._write()
        return ref

    def returned(self, ref, result):
        row = next(row for _, row in self._owned if row.get("task_ref") == ref.hex())
        row.update(state="returned", returncode=result.get("returncode") if isinstance(result, dict) else None)
        self._write()

    def __enter__(self):
        return self

    def __exit__(self, _kind, primary, _traceback):
        errors = self.document["cleanup_errors"]
        try:
            self.session.close()
        except BaseException as error:
            errors.append({"operation": "session_cleanup", "error": type(error).__name__})
        deadline = time.monotonic() + 1
        for invitation, row in self._owned:
            try:
                receipt = self.session.receipt(invitation, timeout=max(0.001, deadline - time.monotonic()))
                row["receipt"] = receipt
                if not receipt.get("cleanup_complete"):
                    raise CleanupIncomplete("native lifecycle evidence incomplete")
                if row["state"] == "returned":
                    native = receipt.get("guardian") or {}
                    if not native.get("native_started") or native.get("returncode") != row["returncode"]:
                        raise CleanupIncomplete("returned task has no matching native execution evidence")
            except BaseException as error:
                errors.append(
                    {"operation": "observe", "invocation": invitation.invocation, "error": type(error).__name__}
                )

        def mark_failed():
            try:
                self.run.mark_failed()
            except BaseException as error:
                errors.append({"operation": "mark_failed", "error": type(error).__name__})

        if primary is not None or errors:
            mark_failed()
        self.document["cleanup_complete"] = not errors
        try:
            self._write()
        except BaseException as error:
            errors.append({"operation": "persist", "error": type(error).__name__})
            self.document["cleanup_complete"] = False
            mark_failed()
        if errors:
            message = f"managed native evidence incomplete; inspect {self.path}"
            if primary is None:
                raise CleanupIncomplete(message)
            primary.add_note(message)
