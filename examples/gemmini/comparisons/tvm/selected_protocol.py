"""Load the user's exact model protocol and approved run-local chain correction."""

import hashlib
import json
from pathlib import Path

import yaml

PROTOCOL_COMMIT = "8ec99308cb1182b9b896a6075300312266fadf6f"
PROTOCOL_SHA256 = "7e1a56a5656f74dfe67fc3c070fecf67df1dc3c0b1d5f1461359042d9de39540"
# Exact 8ec models.yaml with only the user-approved 128 -> 32 prefill correction.
MODELS_SHA256 = "e81125592bef2a24bd45846fab3bd0f5dc3edc19403ce2a43c6a3c3fb43c619b"


def identity(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_selected_protocol(path):
    path = Path(path).resolve(strict=True)
    if identity(path)["sha256"] != PROTOCOL_SHA256:
        raise ValueError("Require the exact 8ec99308 model protocol; checkout defaults are not accepted")
    protocol = yaml.safe_load(path.read_bytes())
    models = (path.parent / protocol["authority"]["numerical_contract"]).resolve(strict=True)
    if identity(models)["sha256"] != MODELS_SHA256:
        raise ValueError("Require 8ec numerics with only the approved 32+32 chain correction")
    numerical = yaml.safe_load(models.read_bytes())
    manifest = path.parent / protocol["authority"]["evaluation_input_manifest"]
    inputs = json.loads(manifest.read_bytes())
    recipes = {}
    for name, model in protocol["models"].items():
        if model["chain"] != numerical["models"][name]["chain"]:
            raise ValueError("Selected protocol and numerical chain disagree: " + name)
        recipe = (path.parent / model["quantization"]["recipe"]).resolve(strict=True)
        if identity(recipe)["sha256"] != model["quantization"]["file_sha256"]:
            raise ValueError("Selected recipe bytes differ: " + name)
        selected = model["external_evaluation_input"]
        if inputs["models"][name]["derived"][selected["derived_member"]] != selected["sha256"]:
            raise ValueError("Selected input manifest differs: " + name)
        recipes[name] = recipe
    return protocol, models, recipes


def check_selected_input(path, model):
    if identity(path)["sha256"] != model["external_evaluation_input"]["sha256"]:
        raise ValueError("Input bytes differ from the selected 8ec protocol")
