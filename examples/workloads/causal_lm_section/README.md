# Pinned TinyLlama source capture

`loader.py` accepts `CAUSAL_LM_REVISION` as an optional 40-character lowercase
Hugging Face commit SHA. When set, it passes that exact revision to
`from_pretrained` and rejects a loaded config with a different or missing
`_commit_hash`. With `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1`, it also
requires that exact snapshot in the local cache before loading. Without this
variable the existing model-name lookup is unchanged.

For the pretrained TinyLlama B1/S8 W8A8 capture, use:

```sh
export CAUSAL_LM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export CAUSAL_LM_REVISION=fe8a4ea1ffedaf415f4da2f062534de366a451e6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CAUSAL_LM_ENTRY=ids M2M_BATCH=1 M2M_SEQ=8
```

Resolve the snapshot in the **same Python environment and HF cache** as the
capture. Check its source files before recapturing; the model file check reads
about 2.2 GB:

```sh
snapshot="$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id=os.environ["CAUSAL_LM_MODEL"],
    revision=os.environ["CAUSAL_LM_REVISION"],
    local_files_only=True,
))
PY
)"
printf '%s  %s\n' \
  486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9 \
  "$snapshot/config.json" |
  sha256sum -c -
printf '%s  %s\n' \
  6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933 \
  "$snapshot/model.safetensors" |
  sha256sum -c -
```

Use the capture venv's Python in place of `python` above. The first two-layer
source uses `CAUSAL_LM_LAYERS=2` after loading the full pretrained checkpoint;
unset it for all 22 layers. Set `M2M_BATCH=8` only for a separate B8/S8
capture. This pin and these file checks establish the selected checkpoint
bytes. Captured input and TorchAO golden hashes, Torch CPU source facts, and
the capture receipt must still be checked on each new bundle.
