# Reproducibility

## Canonical Inputs

This repository's canonical verification inputs are generated in-repo rather than stored as an external corpus.

- `src/zpe_image_codec/fixtures.py` defines the bounded positive and reject-bucket image cases consumed by verification.
- `src/zpe_image_codec/perturbations.py` defines the perturbation pack applied to accepted cases during replay.
- `src/zpe_image_codec/verify.py` is the canonical verifier that assembles those inputs into the fresh falsification packet.

## Golden-Bundle Hash

This field will be populated by the `receipt-bundle.yml` workflow in Wave 3.

## Verification Command

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install '.[dev]'
zpe-image-verify --output validation/results/fresh_falsification_check.local.json
pytest -q
```

## Supported Runtimes

- CPython package runtime and CLI on `requires-python = ">=3.10"`.
- Fresh-clone verification path exercised through the installed Python package in a virtual environment.
- No sibling runtime checkout is required for the bounded verification surface shipped in this repository.
