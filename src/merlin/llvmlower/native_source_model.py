"""Native link transport for the existing checked source/provider transaction.

This route executes explicitly host-domain implementations. Imported witness
identities and artifact custody do not qualify a device, timer or provider body.
"""

import shutil
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.runtime import host_provider
from merlin.runtime.host_provider import (
    HostProviderContext,
    close_host_provider,
    prepare_host_provider,
    revalidate_host_provider,
)

from .compilation_recipe import CompilationRecipe
from .source_model import bind_source_lowerings, source_borrowed_calls, source_provider_builder


class NativeSourceModel:
    """One original native compilation; no provider discovery or runtime fallback."""

    def __init__(self, selections, *, targets):
        self.transform, self.builder = bind_source_lowerings(selections, None, None, None)
        if targets != ("host",) or any(s.execution_domain != "host" for s in selections):
            raise ValueError("native shared source lowering supports only host-domain selections and targets=('host',)")
        self.prepared = None
        self.provider = None
        self.link_inputs = ()
        self.implementation_pins = tuple(
            (p.resolve(), sha256_file(p))
            for p in (Path(__file__), Path(__file__).with_name("codegen.py"), Path(host_provider.__file__))
        )

    def lowering_layout(self, requested):
        from . import target_data_layout
        from .toolchain import clang

        compiler = Path(clang()).resolve(strict=True)
        digest = sha256_file(compiler)
        # Use the existing fresh query, rather than its path-only cache: this
        # transaction binds the current executable bytes before any callback.
        layout = target_data_layout._query(compiler, ("-O2", "-fPIC"))
        if sha256_file(compiler) != digest or (requested is not None and requested != layout):
            raise ValueError("native shared source lowering differs from the selected compiler data layout")
        self.implementation_pins = (
            *self.implementation_pins,
            (compiler, digest),
            (Path(target_data_layout.__file__).resolve(), sha256_file(target_data_layout.__file__)),
        )
        return layout

    def revalidate(self):
        source_provider_builder(self.builder)
        for path, digest in (*self.implementation_pins, *self.link_inputs):
            if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
                raise ValueError("original native model/runtime link input changed")
        if self.provider is not None and self.provider["status"] == "completed":
            revalidate_host_provider(self.provider)

    def prepare(self, source, work):
        from .prepared_model_transform import apply_prepared_model_transform

        path = work / "native-source-input.mlir"
        path.write_text(source, encoding="utf-8")
        self.prepared = apply_prepared_model_transform(path, work / "prepared_model_transform", self.transform)
        self.revalidate()
        return self.prepared.read_text(), source_borrowed_calls(self.builder, self.prepared)

    def begin_codegen(self, work):
        self.revalidate()
        self.compilation = CompilationRecipe(work, producer=Path(__file__))
        self.compilation.bind_preparation(
            "shared_source_lowering", work / "prepared_model_transform" / "prepared_model_transform.json"
        )

    def run(self, command, *, runner, inputs=(), outputs=(), stage="object"):
        if len(outputs) != 1:
            raise ValueError("native source compilation requires one explicit output per command")
        self.revalidate()
        result = self.compilation.run(
            command,
            runner=lambda argv: runner(argv, inputs=inputs, outputs=outputs, stage=stage),
            inputs=inputs,
            output=Path(outputs[0]),
        )
        self.revalidate()
        return result

    def objects(self, llvm, model_object, runtime_object, *, runtime_source, compiler, flags, runner):
        from merlin.runtime.host_provider import _symbols

        from .toolchain import llvm_nm

        self.inspector = llvm_nm().resolve(strict=True)
        self.link_inputs = tuple(
            (p.resolve(), sha256_file(p)) for p in (llvm, model_object, runtime_object, runtime_source, self.inspector)
        )
        # The original model/runtime may already require dynamically linked
        # host functions. A provider cannot add a new external dependency.
        model_defined, model_required = _symbols(model_object, self.inspector)
        runtime_defined, runtime_required = _symbols(runtime_object, self.inspector)
        self.inherited_imports = model_required | runtime_required
        self.inherited_imports -= model_defined.keys() | runtime_defined.keys()
        self.revalidate()

        def compile_provider(command, *, inputs, output):
            return self.run(command, runner=runner, inputs=inputs, outputs=(output,))

        context = HostProviderContext(
            self.prepared,
            llvm,
            model_object,
            llvm.parent / "host_provider",
            Path(shutil.which(str(compiler)) or compiler).resolve(),
            tuple(flags),
            compile_provider,
        )
        objects, self.provider = prepare_host_provider(
            source_provider_builder(self.builder), context, inspector=self.inspector
        )
        self.revalidate()
        # Required source writers must be satisfied by imported definitions;
        # they cannot masquerade as host dynamic imports at the final link.
        if self.provider is not None:
            definitions = {abi["symbol"] for row in self.provider["objects"] for abi in row["defined"]}
            self.inherited_imports -= definitions
        return objects

    def complete(self, objects, image, *, link_flags):
        self.revalidate()
        close_host_provider(
            self.provider,
            objects,
            image,
            inspector=self.inspector,
            link_flags=link_flags,
            native_dynamic_imports=tuple(sorted(self.inherited_imports)),
            native_link_inputs=(*self.implementation_pins, *self.link_inputs),
        )
        self.revalidate()
        self.compilation.completed(image)

    def record(self):
        self.revalidate()
        receipt = self.prepared.parent / "prepared_model_transform.json"
        return {
            "schema": "merlin.native_source_model.v1",
            "prepared_source": {"path": str(self.prepared), "sha256": sha256_file(self.prepared)},
            "preparation": {"path": str(receipt), "sha256": sha256_file(receipt)},
            "host_provider": self.provider,
            "compilation_recipe": {
                "path": str(self.compilation.path),
                "sha256": sha256_file(self.compilation.path),
            },
            "scope": "host-domain provider execution and link custody; no device or timing qualification",
        }
