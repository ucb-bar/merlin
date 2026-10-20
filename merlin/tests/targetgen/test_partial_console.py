"""A stopped simulator run keeps the cycles its harness printed before the output frame."""

from merlin.targetgen import partial_console as PC


def test_metrics_read_complete_lines_only_and_need_no_done():
    console = "METRIC cycles 4242\nMETRIC cycle_window_x 1\nOUT Y0 3136 64 1 2 3 4 5\nMETRIC trunc 12"
    assert PC.metrics(console) == {"cycles": 4242, "cycle_window_x": 1}
    assert PC.metrics(b"METRIC cycles 7\nOUT_BIN_BEGIN v1 Y0 1 1 4 s 4\n\x00\x01") == {"cycles": 7}
    assert PC.metrics(None) == {} and PC.metrics("METRIC cycles notanumber\n") == {}


def test_recover_reads_the_transcript_a_stopped_run_left(tmp_path):
    assert PC.recover(tmp_path) is None
    (tmp_path / "oracle_console.log").write_text("METRIC cycles 9\nOUT Y0 1 2 1")
    assert PC.metrics(PC.recover(tmp_path)) == {"cycles": 9}
