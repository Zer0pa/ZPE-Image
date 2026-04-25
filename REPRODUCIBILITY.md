# Reproducibility

## Canonical Inputs

- `src/zpe_image_codec/fixtures.py` deterministically generates the named sparse-stroke, hole-bearing, mixed, and negative-natural verification cases used by the package.
- `proofs/artifacts/fresh_falsification_packet.json` is the promoted proof artifact for the current bounded claim surface.
- `validation/results/fresh_falsification_check.json` is the committed fresh verification replay for the promoted packet.

## Golden-Bundle Hash

This will be populated by the `receipt-bundle.yml` workflow in Wave 3.

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

- CPython 3.10+ via the root `zpe-image` package install.
- CLI verification through `zpe-image-verify` from a fresh clone without any sibling checkout.
