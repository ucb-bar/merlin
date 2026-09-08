"""Pure target-owned build formatting, loadable without the execution backend."""
from .format import CodegenError, Container, container_for, container_words
from .measurement import assemble_measurement_fragments
from .whole_program import render_whole_program


def build_source_paths():
    """Exact target-owned pure source closure; shared runtime files are pinned separately."""
    from pathlib import Path
    root = Path(__file__).resolve().parent
    return tuple(root / name for name in ('__init__.py', 'format.py', 'measurement.py', 'whole_program.py'))
