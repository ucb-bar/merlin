"""Capture PyTorch sources through the selected model2MLIR worker and atomic cache transaction."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

from merlin.capture.artifacts import _MODEL_CAPTURE_ABI_VERSION, CapsuleArtifacts, M2MUnavailable, _float_reference_of
from merlin.capture.cachekey import capture_slot
from merlin.capture.loaders import build_loader_src
from merlin.capture.numerics import _DEF_ATOL, _DEF_RTOL
from merlin.capture.receipts import (
    _require_integer_contraction,
    _require_materialized_contraction,
    _require_pt2e_integerization_receipt,
    _static_pt2e_model,
)
from merlin.common.paths import module_source_path, repo_root
from merlin.targetgen import capture_cache
from merlin.targetgen._capture_diagnostics import stderr_cause as _stderr_cause


def _m2m_dir() -> Path:
    """The configured model2MLIR checkout, else a sibling of this repo.

    model2MLIR is an EXTERNAL repo, so the fallback is a CONVENTION (checked out next to merlin),
    not one machine's home directory -- callers already treat a non-existent path as "not
    available" and skip, and that behaviour is preserved.
    """
    from merlin.integrations.model2mlir import root

    return root(default=repo_root().parent / "model2MLIR")


def _m2m_python(*, checkout: Path | None = None) -> Path:
    """Shared interpreter selection with this edge's historical sibling default."""
    from merlin.integrations.model2mlir import capture_python_path

    return capture_python_path(checkout=checkout if checkout is not None else _m2m_dir())


class PytorchRefSource:
    """Capture a capsule from a PyTorch op via model2MLIR (subprocess, m2m venv)."""

    def __init__(self, m2m_dir: Path | None = None, python: Path | None = None, timeout: int = 900):
        self.m2m_dir = Path(m2m_dir) if m2m_dir else _m2m_dir()
        self.python = Path(python) if python else _m2m_python(checkout=self.m2m_dir)
        self.timeout = timeout

    def available(self) -> bool:
        return self._runnable(self.python)

    def _runnable(self, python: Path) -> bool:
        """Whether a capture can run under ``python`` -- the default interpreter, or the one a workload
        pins for itself. Same two conditions either way: an executable interpreter, next to an m2m
        checkout to import."""
        return (
            Path(python).is_file() and os.access(python, os.X_OK) and (self.m2m_dir / "m2m" / "__init__.py").is_file()
        )

    def capture(self, spec: dict, *, workdir: str | Path | None = None) -> CapsuleArtifacts:
        """Capture a generated op loader (matmul/attention/... rendered from ``spec``)."""
        if not self.available():
            raise M2MUnavailable(
                f"m2m venv python {self.python} or repo {self.m2m_dir} missing; set MERLIN_M2M_PYTHON / "
                f"MERLIN_M2M_DIR (torch is required and lives only in the m2m venv)"
            )
        dtype = spec.get("dtype", "fp32")
        loader_src = build_loader_src(spec)
        tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="capsule_m2m_"))
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "loader.py").write_text(loader_src, encoding="utf-8")
        return self._run(
            tmp / "loader.py", spec["op"], dtype, workdir=tmp, src=loader_src, scheme=spec.get("quant_scheme")
        )

    def capture_loader(
        self,
        loader_py: str | Path,
        dtype: str,
        *,
        workdir: str | Path | None = None,
        scheme: str | None = None,
        env: dict | None = None,
        python: str | Path | None = None,
        recipe: dict | None = None,
        already_quantized: bool = False,
        agreement_tolerance: tuple[float, float] | None = None,
        activation_contractions: bool = False,
        integer_nonlinear: bool = False,
    ) -> CapsuleArtifacts:
        """Capture an EXISTING loader file (a whole-model workload) rather than a generated op loader.
        Same worker path; ``op`` is ``model`` and ``pytorch_src`` is the loader's own source.

        ``scheme`` reaches the worker exactly as it does for an op capsule. Without it the whole-model
        path took the DEFAULT for its dtype, and the default for int8 is weight-only -- a float matmul
        over dequantized weights. A model capsule declaring `compile_dtype: int8` was therefore capturing
        a program with no integer contraction in it at all, which is the wrong program for an integer
        mesh and one no golden substitution can repair.

        ``env`` is the model's OWN declared loader environment (:func:`model_capture_env`) and ``python``
        the interpreter it pins (:func:`model_capture_python`). ``recipe`` is a ``quant_recipe_v1``
        mapping derived from the TARGET (:mod:`merlin.targetgen.quant_recipe`); with one, the model is
        quantized under it and ``scheme`` is not consulted. Both used to be ignored here, and a real
        network is not capturable without them: a loader that refuses to invent inputs unless told which
        stream to use raised, and a loader whose dependency lives only in its own venv raised
        ``ModuleNotFoundError`` -- and BOTH were recorded as "this model could not be built", which reads
        as a limit of the compiler rather than of how it was invoked.

        CAPTURE CAPABILITIES. ``activation_contractions`` also quantizes, in the scheme's own int8 dynamic
        form, the contractions the scheme cannot reach -- a matmul of two activations (attention's
        ``Q @ K^T`` and ``P @ V``) and a convolution that is a pure re-layout of a matmul (a patch
        embedding); see ``_activation_contractions.py``. The worker refuses it under any other
        quantization. ``integer_nonlinear`` (with ``activation_contractions`` only) also computes
        softmax, GELU and layer norm in integer arithmetic (``_integer_nonlinear.py``): a different
        numerical model, recorded as such."""
        loader_py = Path(loader_py)
        if integer_nonlinear and not activation_contractions:
            raise M2MUnavailable("integer nonlinear capture is defined on top of activation contractions only")
        if already_quantized and (recipe is not None or scheme):
            raise M2MUnavailable("already-materialized capture cannot also apply a recipe or a quantization scheme")
        # `available()` stays the predicate when nothing is pinned, so a caller that has established
        # availability some other way keeps saying so; a PINNED interpreter is checked on its own terms.
        interpreter = Path(python) if python else self.python
        if not (self._runnable(interpreter) if python else self.available()):
            raise M2MUnavailable(f"m2m venv python {interpreter} or repo {self.m2m_dir} missing")
        if not loader_py.is_file():
            raise M2MUnavailable(f"model loader not found: {loader_py}")
        tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="capsule_model_"))
        tmp.mkdir(parents=True, exist_ok=True)
        return self._run(
            loader_py,
            "model",
            dtype,
            workdir=tmp,
            src=loader_py.read_text(),
            scheme=scheme,
            env=env,
            python=interpreter,
            recipe=recipe,
            already_quantized=already_quantized,
            agreement_tolerance=agreement_tolerance,
            activation_contractions=activation_contractions,
            integer_nonlinear=integer_nonlinear,
        )

    def _launch_worker(self, cmd: list[str], *, env: dict, **_request) -> subprocess.CompletedProcess[str]:
        """Run the capture worker; a sealed capture source runs the same request in a sandbox instead.

        ``_request`` carries the structured capture request (loader, dtype, recipe path, tolerance,
        declared environment, interpreter) so a subclass need not parse ``cmd``.
        """
        return subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, env=env)

    def _cache_slot(
        self,
        op: str,
        dtype: str,
        src: str,
        scheme: str | None,
        env: dict | None = None,
        python: Path | None = None,
        recipe_sha256: str = "",
        already_quantized: bool = False,
        agreement_tolerance: tuple[float, float] | None = None,
        static_pt2e: bool = False,
        capabilities: tuple[str, ...] = (),
    ) -> Path | None:
        """Resolve the request cache key through the original overridable source boundary."""
        return capture_slot(
            self.m2m_dir,
            self.python,
            op,
            dtype,
            src,
            scheme,
            env,
            python,
            recipe_sha256,
            already_quantized,
            agreement_tolerance,
            static_pt2e,
            capabilities,
        )

    def _run(
        self,
        loader_py: Path,
        op: str,
        dtype: str,
        *,
        workdir: Path,
        src: str,
        scheme: str | None = None,
        env: dict | None = None,
        python: Path | None = None,
        recipe: dict | None = None,
        already_quantized: bool = False,
        agreement_tolerance: tuple[float, float] | None = None,
        activation_contractions: bool = False,
        integer_nonlinear: bool = False,
    ) -> CapsuleArtifacts:
        capabilities = tuple(
            name
            for name, asked in (
                ("activation_contractions", activation_contractions),
                ("integer_nonlinear", integer_nonlinear),
            )
            if asked
        )
        # WHICH FRONTEND MADE THIS CAPTURE, verified against its software pin and recorded on every
        # capture. A capture that computes softmax/GELU/layer norm in integer arithmetic is REFUSED off the
        # pin: its integer shifts and floor divisions only lower exactly through the decompositions the
        # pinned revision carries, and an off-pin checkout would import them as opaque calls or, worse,
        # with different bytes behind the same name.
        from merlin.integrations.model2mlir import capture_pin

        frontend_pin = capture_pin(getattr(self, "m2m_dir", None))
        if "integer_nonlinear" in capabilities and not frontend_pin["ok"]:
            raise M2MUnavailable(
                f"integer nonlinear capture needs model2MLIR at its software pin: {frontend_pin['citation']}; "
                f"drift: {frontend_pin['verification'].get('drift')}"
            )
        if agreement_tolerance is not None and (
            not isinstance(agreement_tolerance, (tuple, list))
            or len(agreement_tolerance) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in agreement_tolerance
            )
        ):
            raise M2MUnavailable("capture agreement tolerance must be a finite nonnegative (atol, rtol) pair")
        if agreement_tolerance is None and _static_pt2e_model(
            op, scheme=scheme, recipe=recipe, already_quantized=already_quantized
        ):
            agreement_tolerance = (_DEF_ATOL, _DEF_RTOL)
        static_pt2e = _static_pt2e_model(op, scheme=scheme, recipe=recipe, already_quantized=already_quantized)
        recipe_sha256 = ""
        if recipe is not None:
            from merlin.compiler.quantization import recipe as _recipe

            if recipe.get("status") != _recipe.DERIVED:
                raise M2MUnavailable(
                    f"the quantization recipe is {recipe.get('status')!r}, not derived, so there is "
                    f"nothing to capture under: {recipe.get('underivable')}"
                )
            recipe_sha256 = _recipe.digest(recipe)
        # A CACHED CAPTURE IS THE SAME CAPTURE. Reused only when the slot holds a complete, clean result
        # -- the same two conditions the fresh path checks below -- so a half-written slot from an
        # interrupted run re-captures instead of serving a truncated module.
        declared_env = {str(k): str(v) for k, v in (env or {}).items()}
        interpreter = Path(python) if python else self.python
        slot = self._cache_slot(
            op,
            dtype,
            src,
            scheme,
            declared_env,
            interpreter,
            recipe_sha256,
            already_quantized,
            agreement_tolerance,
            static_pt2e,
            capabilities,
        )
        with capture_cache.slot_lock(slot) as locked_slot:
            if slot is not None and locked_slot is None:
                # Never run unlocked in the shared directory. Retain this private
                # result because CapsuleArtifacts exposes its weights by pathname.
                from merlin.common.paths import build_dir

                root = build_dir() / "model-capture"
                root.mkdir(parents=True, exist_ok=True)
                workdir = Path(tempfile.mkdtemp(prefix="uncached-", dir=root))
            artifact, attempt = self._run_capture(
                loader_py,
                op,
                dtype,
                workdir=workdir,
                src=src,
                scheme=scheme,
                declared_env=declared_env,
                interpreter=interpreter,
                recipe=recipe,
                recipe_sha256=recipe_sha256,
                already_quantized=already_quantized,
                agreement_tolerance=agreement_tolerance,
                capabilities=capabilities,
                slot=locked_slot,
            )
            if locked_slot is not None:
                current = self._cache_slot(
                    op,
                    dtype,
                    src,
                    scheme,
                    declared_env,
                    interpreter,
                    recipe_sha256,
                    already_quantized,
                    agreement_tolerance,
                    static_pt2e,
                    capabilities,
                )
                if current != locked_slot:
                    raise M2MUnavailable("capture implementation changed during cache transaction; retry capture")
                if not capture_cache.observed_sources_match(artifact.meta):
                    raise M2MUnavailable("observed loader dependencies changed during cache transaction; retry capture")
                capture_cache.commit(locked_slot, attempt)
            if isinstance(artifact, CapsuleArtifacts):
                artifact.meta = {**(artifact.meta or {}), "model2mlir_pin": frontend_pin}
            return artifact

    def _run_capture(
        self,
        loader_py: Path,
        op: str,
        dtype: str,
        *,
        workdir: Path,
        src: str,
        scheme: str | None,
        declared_env: dict,
        interpreter: Path,
        recipe: dict | None,
        recipe_sha256: str,
        already_quantized: bool = False,
        agreement_tolerance: tuple[float, float] | None = None,
        capabilities: tuple[str, ...] = (),
        slot: Path | None,
    ) -> tuple[CapsuleArtifacts, Path]:
        """Lookup/capture/normalize/read while holding the caller's entire slot transaction.

        Source keys bind the direct Merlin implementation owners, not every upstream
        framework/importer dependency or arbitrary loader input. The caller rechecks
        the key before publishing or returning a shared-cache result.
        """
        worker = module_source_path("merlin.targetgen._m2m_capture_worker")
        cached_slot = capture_cache.committed_attempt(slot) if slot is not None else None
        if cached_slot is not None:
            try:
                cached = json.loads((cached_slot / "meta.json").read_text())
                # A slot written before the worker recorded input provenance cannot say what its inputs
                # were, and a capsule may not be built from a capture whose provenance is unrecoverable.
                # Re-capturing is the cheap, honest repair; serving it would launder an unknown.
                if (
                    cached.get("ok")
                    and cached.get("opaque", -1) == 0
                    and "loader_provenance_status" in cached
                    and cached.get("capture_abi_version") == _MODEL_CAPTURE_ABI_VERSION
                    and isinstance(cached.get("output_abi"), list)
                    and Path(cached.get("weights_manifest", "")).is_file()
                    and capture_cache.observed_sources_match(cached)
                    and capture_cache.frontend_trace_matches(cached, cached_slot)
                ):
                    cached_program = (cached_slot / "linalg.mlir").read_text(encoding="utf-8")
                    _require_integer_contraction(cached_program, scheme=scheme, recipe=recipe)
                    if _static_pt2e_model(op, scheme=scheme, recipe=recipe, already_quantized=already_quantized):
                        _require_pt2e_integerization_receipt(
                            cached_program, cached, agreement_tolerance=agreement_tolerance, capture_root=cached_slot
                        )
                    if already_quantized:
                        _require_materialized_contraction(cached_program, dtype=dtype)
                    return CapsuleArtifacts(
                        op=op,
                        dtype=dtype,
                        pytorch_src=src,
                        linalg_mlir=cached_program,
                        inputs=json.loads((cached_slot / "inputs.json").read_text()),
                        golden=json.loads((cached_slot / "golden.json").read_text()),
                        weights_path=cached.get("weights", str(cached_slot / "weights.safetensors")),
                        meta=cached,
                        float_reference=_float_reference_of(cached_slot),
                    ), cached_slot
            except Exception:  # noqa: BLE001 -- an unreadable slot is a miss, never a failure
                pass
        if slot is not None:
            slot.mkdir(parents=True, exist_ok=True)
            # A killed parent can leave its worker alive. Each attempt owns its
            # output bytes; only a parent can publish the shared index.
            workdir = Path(tempfile.mkdtemp(prefix="attempt-", dir=slot))
            (slot / "capture_complete.json").unlink(missing_ok=True)
        # THE MODEL'S OWN DECLARED LOADER ENVIRONMENT, applied under the ambient one. A loader states
        # what it needs next to itself (its workload's `capture.toml`, plus the curated fidelity knobs in
        # `baselines.bundle`); building the worker environment from `os.environ` alone silently dropped
        # that declaration, and every loader that refuses to guess then raised -- recorded as an unbuilt
        # model. The DECLARATION WINS over an ambient value, as it does on every other arm that replays
        # it: a capsule must capture the model its declaration names, not whatever the shell that ran
        # the generator happened to export, or two regenerations produce two different networks under
        # one name.
        env = dict(os.environ)
        env.update(declared_env)
        env["MERLIN_M2M_DIR"] = str(self.m2m_dir)
        # Live micro-model loaders live under the generated corpus. Importing
        # them must not add interpreter cache files to the immutable artifact.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmd = [
            str(interpreter),
            str(worker),
            "--loader",
            str(loader_py),
            "--dtype",
            dtype,
            "--out",
            str(workdir),
            "--m2m-dir",
            str(self.m2m_dir),
        ]
        if recipe is not None:
            recipe_path = workdir / "quant_recipe.json"
            recipe_path.write_text(json.dumps({**recipe, "recipe_sha256": recipe_sha256}, indent=1), encoding="utf-8")
            cmd += ["--recipe", str(recipe_path)]
        elif scheme:
            cmd += ["--scheme", str(scheme)]
        if already_quantized:
            cmd.append("--already-quantized")
        if "activation_contractions" in capabilities:
            cmd.append("--quantize-activation-contractions")
        if "integer_nonlinear" in capabilities:
            cmd.append("--integer-nonlinear")
        if agreement_tolerance is not None:
            cmd += ["--agreement-atol", str(agreement_tolerance[0]), "--agreement-rtol", str(agreement_tolerance[1])]
        # A rejected cache entry or reused caller directory may still hold the
        # previous attempt's verdict. Invalidate it before retrying so a crash
        # cannot inherit the previous attempt's success.
        meta_p = workdir / "meta.json"
        meta_p.unlink(missing_ok=True)
        proc = self._launch_worker(
            cmd,
            env=env,
            workdir=workdir,
            loader_py=loader_py,
            op=op,
            dtype=dtype,
            recipe_path=(workdir / "quant_recipe.json") if recipe is not None else None,
            scheme=scheme,
            already_quantized=already_quantized,
            agreement_tolerance=agreement_tolerance,
            declared_env=declared_env,
            interpreter=interpreter,
        )
        # ⚠️ A NON-ZERO RC IS NOT NECESSARILY A CRASH, and treating it as one threw away the
        # diagnosis. The worker returns 3 for "ran fine, but the program is not clean" -- and it
        # WRITES meta.json with `opaque` and `opaque_detail` before doing so. This branch fired first
        # on rc=3 and dumped the stderr tail, which for a torch export is a wall of warnings, while
        # the block just below already had the useful message. Measured on the lstmnetvit roster model
        # at W8A8: the real answer is opaque ops from un-inlined torchAO quantization calls, and what
        # got reported was an LSTM `_flat_weights` contiguity notice.
        #
        # So the stderr path is now for a worker that produced NO diagnosis at all; if meta.json
        # exists the worker reached its own verdict and that verdict is what to report.
        if not meta_p.exists():
            # ⚠️ REPORT THE CAUSE, NOT THE TAIL. This used to print `stderr[-1500:]`, and a capture
            # that emits warnings AFTER its traceback therefore reported the warnings. Measured on
            # the lstmnetvit roster model: the failure was attributed to a benign
            # "tensor attributes self.net.lstm._flat_weights[...] are not part of a single contiguous
            # chunk of memory" UserWarning, which is not an error at all and sent a reader looking at
            # the wrong thing. The two conditions are also different failures and were reported as
            # one: a non-zero rc means the worker died, a missing meta.json after rc=0 means it
            # exited cleanly without producing a capture.
            why = "worker exited non-zero" if proc.returncode != 0 else "worker exited 0 but wrote no meta.json"
            raise M2MUnavailable(
                f"m2m capture failed for op {op!r}/{dtype}: {why} (rc={proc.returncode})\n{_stderr_cause(proc.stderr)}"
            )
        meta = json.loads(meta_p.read_text())
        # The capture worker reports the RAW importer program.  Before enforcing zero opaque calls,
        # pass that program through the same target-neutral semantic normalization every whole-model
        # backend already needs.  This is deliberately here, at the shared capture boundary: putting
        # it in one backend makes corpus validity target-dependent, while teaching the torch-only
        # worker about Merlin's compiler silently forks the canonical lowering.
        #
        # The normalizer cross-checks the worker's census against a structural parse, records both
        # program digests and the raw census, and leaves any unknown call in the remaining census.
        # Therefore this cannot turn an unrecognized opaque op into a clean capsule.
        if meta.get("opaque"):
            from merlin.frontends.capture_normalization import (
                CaptureNormalizationError,
                normalize_capture_mlir,
            )

            linalg_p = workdir / "linalg.mlir"
            if linalg_p.is_file():
                # Preserve the exact frontend output behind its source trace.
                # Canonical normalization has a separate byte identity and is
                # not silently relabeled as the frontend's lowering receipt.
                raw_linalg = linalg_p.read_text(encoding="utf-8")
                raw_path = workdir / "frontend-source.mlir"
                raw_path.write_text(raw_linalg, encoding="utf-8")
                meta["frontend_raw_mlir_path"] = str(raw_path)
                try:
                    normalized, normalization = normalize_capture_mlir(
                        raw_linalg,
                        reported_opaque=meta.get("opaque_detail") or {},
                    )
                except CaptureNormalizationError as exc:
                    raise M2MUnavailable(f"m2m capture normalization failed for op {op!r}/{dtype}: {exc}") from exc
                linalg_p.write_text(normalized, encoding="utf-8")
                meta["capture_normalization"] = normalization
                meta["raw_linalg_ops"] = meta.get("linalg_ops")
                meta["linalg_ops"] = normalized.count("linalg.")
                remaining = normalization["remaining_opaque_detail"]
                meta["opaque_detail"] = remaining
                meta["opaque"] = sum(remaining.values())
                # Meta is written last.  An interruption before this point leaves the slot marked
                # non-clean and forces a fresh capture rather than serving half-normalized bytes.
                meta_p.write_text(json.dumps(meta), encoding="utf-8")
        if not meta.get("ok") or meta.get("opaque", -1) != 0:
            # Name WHICH ops are opaque: "opaque=44" says a capsule cannot use this program, and the
            # detail says what to go and inline. The scheme is included because the same model is
            # clean under one quantization and not another -- lstmnetvit captures cleanly at plain
            # int8 and carries opaque torchAO calls at W8A8, which is a fact about the scheme.
            detail = meta.get("opaque_detail") or {}
            top = ", ".join(f"{k}={v}" for k, v in sorted(detail.items(), key=lambda kv: -int(kv[1] or 0))[:6])
            raise M2MUnavailable(
                f"m2m capture produced a non-clean program for op {op!r}/{dtype} "
                f"(ok={meta.get('ok')}, opaque={meta.get('opaque')}, "
                f"scheme={meta.get('scheme')!r}, rc={proc.returncode}); a capsule input must be "
                f"0-opaque. Opaque ops: {top or '(none reported)'}"
            )
        program = (workdir / "linalg.mlir").read_text(encoding="utf-8")
        _require_integer_contraction(program, scheme=scheme, recipe=recipe)
        if _static_pt2e_model(op, scheme=scheme, recipe=recipe, already_quantized=already_quantized):
            _require_pt2e_integerization_receipt(
                program, meta, agreement_tolerance=agreement_tolerance, capture_root=workdir
            )
        if already_quantized:
            meta["materialized_contractions"] = _require_materialized_contraction(program, dtype=dtype)
            meta_p.write_text(json.dumps(meta), encoding="utf-8")
        return CapsuleArtifacts(
            op=op,
            dtype=dtype,
            pytorch_src=src,
            linalg_mlir=program,
            inputs=json.loads((workdir / "inputs.json").read_text()),
            golden=json.loads((workdir / "golden.json").read_text()),
            weights_path=meta.get("weights", str(workdir / "weights.safetensors")),
            meta=meta,
            float_reference=_float_reference_of(workdir),
        ), workdir
