"""Keep optional original-host selections within their ordinary workflow."""

from . import broker_policy as BP
from .contracts import StageGateError


def validate(workflow_id, search_configuration, compiler_library_selection, archive_screen_selection):
    if compiler_library_selection is not None and workflow_id != BP.CORPUS_FEEDBACK_V1:
        raise StageGateError("additional compiler library requires the ordinary authoring workflow")
    if archive_screen_selection is not None and (
        workflow_id != BP.CORPUS_FEEDBACK_V1 or search_configuration is None
    ):
        raise StageGateError("archive screening requires the original ordinary queued authoring workflow")
