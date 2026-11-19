"""Export inventory geometry is supplied by the caller's target provider."""
import json

import pytest

from merlin.compile import residency_inventory as inventory


@pytest.fixture
def exported(monkeypatch):
    monkeypatch.setattr(inventory, "load_safetensors_header", lambda path: ({
        "a": {"data_offsets": [0, 3]}, "b": {"data_offsets": [3, 7]},
    }, 0))


def test_inventory_uses_requested_alignment_and_exported_bytes(exported):
    requests = inventory.inventory("export", alignment=16)
    assert [(row.name, row.bytes, row.alignment) for row in requests] == [("a", 3, 16), ("b", 4, 16)]


def test_cli_uses_explicit_non_atlas_aperture_and_alignment(exported, tmp_path):
    output = tmp_path / "inventory.json"
    args = ["--weights", "export", "--capacity-base", "0x100", "--capacity-bytes", "20",
            "--dma-base", "0x100", "--dma-capacity", "64", "--alignment", "16", "--out", str(output)]
    assert inventory.main(args) == 0
    result = json.loads(output.read_bytes())
    assert result["plan"]["dma_aperture"] == {"base": 256, "capacity": 64}
    assert result["plan"]["required_bytes"] == 20
    assert [row["address"] for row in result["plan"]["entries"]] == [256, 272]
    args[args.index("20")] = "19"
    assert inventory.main(args) == 2
    assert json.loads(output.read_bytes())["overflow_bytes"] == 1


def test_cli_requires_target_geometry(exported, tmp_path):
    with pytest.raises(SystemExit) as error:
        inventory.main(["--weights", "export", "--capacity-base", "0", "--capacity-bytes", "64",
                        "--out", str(tmp_path / "out.json")])
    assert error.value.code == 2
