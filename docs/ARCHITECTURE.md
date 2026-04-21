# Architecture

## Runtime Surface

The public package lives entirely in `src/zpe_image_codec`. It now carries its own sparse route, topological route, enhanced fallback codec, narrower hole-bearing route, fixtures, perturbations, and verification runner.

## Standalone Install

The installed package no longer depends on any sibling checkout. Fresh-clone verification runs through `zpe-image-verify` after `pip install '.[dev]'`.

## Repo Roles

- `src/zpe_image_codec` contains the standalone codec runtime.
- `tests/` verifies the bounded route behavior and the README contract.
- `proofs/` contains the committed verification packet promoted by the README.
- `validation/` contains the latest fresh falsification run.
