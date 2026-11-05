"""Original-host DSE selection for one ordinary schedule compilation.

This companion is outside frozen SDK inventories. Its factory binds a live
original deletion law and verification code; serialized records create no
qualification and author requests cannot choose its callbacks or premises.
"""

from dataclasses import asdict
from pathlib import Path

from . import dead_write_check as C
from . import dead_writes as D
from .schedule_pipeline import PostStorageRewrite, ScheduleBudgets, compile_schedule


class _Deletion:
    def __init__(self, law, budgets, maximum, work):
        if type(law) is not C.WriteDeletionLaw or type(budgets) is not ScheduleBudgets:
            raise ValueError("ordinary write deletion needs original typed law and work budgets")
        if type(maximum) is not int or maximum < 0 or type(work) is not int or work <= 0:
            raise ValueError("ordinary write deletion needs finite explicit deletion and scan bounds")
        self.law, self.budgets, self.maximum, self.work = law, budgets, maximum, work
        self._original_law, self._original_budgets = law, budgets
        self._law_record = law.record
        self._original = (self._law_record(), asdict(budgets), maximum, work)
        self._reader = self._functions
        self._callbacks = self._reader()

    @staticmethod
    def _functions():
        return tuple(
            (value, value.__code__)
            for module in (C, D)
            for value in vars(module).values()
            if getattr(value, "__module__", None) == module.__name__ and hasattr(value, "__code__")
        ) + tuple(
            (value, value.__code__)
            for value in (
                _Deletion.__init__,
                _Deletion._functions,
                _Deletion.revalidate,
                _Deletion.record,
                _Deletion.options,
                _Deletion.propose,
                _Deletion.verify,
                D.eliminate_dead_writes,
                C.verify_dead_writes,
                C.WriteDeletionLaw.record,
                C.WriteDeletionLaw.__post_init__,
            )
        )

    def revalidate(self):
        current = self._reader()
        if len(current) != len(self._callbacks) or any(
            a is not b or ac is not bc for (a, ac), (b, bc) in zip(current, self._callbacks, strict=True)
        ):
            raise ValueError("original write-deletion implementation changed during compilation")
        if (
            self.law is not self._original_law
            or self.budgets is not self._original_budgets
            or (self._law_record(), asdict(self.budgets), self.maximum, self.work) != self._original
        ):
            raise ValueError("original write-deletion law or limits changed during compilation")

    def record(self):
        self.revalidate()
        return {
            "law": self._law_record(),
            "budgets": asdict(self.budgets),
            "max_deletions": self.maximum,
            "max_scan_steps": self.work,
            "scope": "original qualified private payload deletion; no new permission",
        }

    def options(self, windows, ownership):
        self.revalidate()
        return dict(
            deletion_law=self.law,
            external_windows=windows,
            private_ownership=ownership,
            max_calls=self.budgets.max_calls,
            max_objects=self.budgets.max_objects,
            max_deletions=self.maximum,
            max_scan_steps=self.work,
        )

    def propose(self, source, objects, banks, adapter, windows, ownership):
        result = D.eliminate_dead_writes(source, objects, banks, adapter, **self.options(windows, ownership))
        self.revalidate()
        return None if result is None else result.kernel

    def verify(self, source, actual, objects, banks, adapter, windows, ownership):
        result = C.verify_dead_writes(source, actual, objects, banks, adapter, **self.options(windows, ownership))
        self.revalidate()
        return result


def compile_schedule_with_dead_writes(
    *args,
    deletion_law=None,
    budgets=ScheduleBudgets(),
    max_write_deletions=128,
    max_write_deletion_scan_steps=100000,
    **kwargs,
):
    """Compile ONCE through the ordinary pre-emission pipeline.

    Missing law preserves the original default route. The original host owns
    the live law/verifier selection; records or callback names grant nothing.
    """
    if "post_storage" in kwargs:
        raise ValueError("write deletion cannot replace another original post-storage selection")
    if deletion_law is None:
        return compile_schedule(*args, budgets=budgets, **kwargs)
    owner = _Deletion(deletion_law, budgets, max_write_deletions, max_write_deletion_scan_steps)
    selection = PostStorageRewrite(
        "dead_writes",
        owner.propose,
        owner.verify,
        owner.revalidate,
        owner.record,
        (Path(__file__), Path(C.__file__), Path(D.__file__)),
    )
    return compile_schedule(*args, budgets=budgets, post_storage=selection, **kwargs)
