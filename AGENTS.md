# AGENTS.md

## Cursor Cloud specific instructions

This repo is the **control plane** for Ascend-NPU LLM serving. The Cloud VM has
**no Ascend NPU / CANN / MindIE**, so real inference (EngineServer + vLLM-Ascend /
SGLang) cannot run here. The local dev loop is the **unit test suite**, which
mocks all hardware (NPU, `npu-smi`, etcd, K8s). A `.venv` (uv, Python 3.12) with
`requirements.txt` + pytest extras is already set up; `uv` is at `~/.local/bin`.

- **Generate protobufs before running anything:** `bash scripts/generate_proto.sh`
  (outputs are git-ignored). Imports/tests fail without it.
- Set `PYTHONPATH` to include the repo root and `motor/` when invoking pytest
  directly.
- Run tests with `bash tests/run_tests.sh --serial` (its default parallelism is
  too high for 4 CPUs). **Caveat:** `run_tests.sh` treats warnings as failure and
  exits non-zero even when every test passes — the tree currently has pre-existing
  `DeprecationWarning`s (pyOpenSSL X509). Use `.venv/bin/python -m pytest tests/ -q`
  to see the real result (expect ~1599 passed, 2 skipped).
- Lint via `pre-commit run --all-files`; hook repos are pulled from `gitcode.com`
  mirrors (needs network).
