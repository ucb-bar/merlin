"""``readable_json`` keeps objects indented, writes plain arrays on one line, and parses identically."""

from __future__ import annotations

import json

import pytest

from merlin.common.jsonio import readable_json


@pytest.mark.parametrize(
    "value",
    [
        {"tensor": [[1, 2], [3, 4]], "rows": [{"a": 1}, 2], "empty": [], "none": {}, 7: None, True: "t"},
        [1.5, "x", None, [True, False]],
        {"nested": [[{"deep": [1, 2]}]], "text": "café \"quoted\"\n"},
        "scalar",
    ],
)
def test_parses_to_what_json_dumps_parses_to(value):
    assert json.loads(readable_json(value)) == json.loads(json.dumps(value))


def test_numeric_arrays_take_one_line_and_objects_stay_indented():
    text = readable_json({"W": [[1, 2, 3], [4, 5, 6]], "meta": {"shape": [2, 3]}})
    assert '"W": [[1,2,3],[4,5,6]]' in text
    assert '\n  "meta": {\n    "shape": [2,3]\n  }' in text
    assert len(readable_json({"W": list(range(1000))})) < len(json.dumps({"W": list(range(1000))}, indent=2)) / 2


def test_unsupported_keys_are_refused_like_json_dumps():
    with pytest.raises(TypeError):
        readable_json({(1, 2): "tuple key"})
