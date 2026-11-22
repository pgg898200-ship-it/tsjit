# Release & Distribution Checklist (`tsjit` v0.1.0)

This repository is verified clean-room (zero personal identifiers, local filesystem paths, private hostnames/IPs, or proprietary references) and ready to publish under an independent open-source organization account.

## 1. Pre-Publish Verification Gate

Run from the repository root before tagging or uploading any artifact:

```bash
# 1. Clean-room anonymization audit (must return finding_count: 0)
python3 scripts/cleanroom_audit.py . --json

# 2. Lint and full test suite
ruff check --no-cache src tests benchmarks scripts
pytest -v -p no:cacheprovider tests/

# 3. Build sdist + wheel into an isolated output directory
python3 -m build --outdir /tmp/tsjit_release_dist

# 4. Verify zero-dependency isolated install + CLI + MCP smoke test
python3 -m venv /tmp/tsjit_release_venv
/tmp/tsjit_release_venv/bin/pip install --no-deps /tmp/tsjit_release_dist/tsjit-0.1.0-py3-none-any.whl
/tmp/tsjit_release_venv/bin/tsjit --version
/tmp/tsjit_release_venv/bin/tsjit info --json
/tmp/tsjit_release_venv/bin/tsjit verify --json
/tmp/tsjit_release_venv/bin/tsjit bench --steps 2000 --agents 4 --json
```

## 2. Publishing to GitHub, PyPI & Hugging Face (User-Authenticated)

Create or sign into a clean, independent organization account (e.g., `github.com/tsjit`, PyPI Trusted Publisher, and a matching Hugging Face org):

1. **GitHub:**
   ```bash
   git remote add origin https://github.com/<your-org>/tsjit.git
   git push -u origin main
   git tag -a v0.1.0 -m "tsjit v0.1.0"
   git push origin v0.1.0
   ```
   Update `[project.urls]` in `pyproject.toml` if your organization slug differs from `tsjit`.

2. **PyPI (via Trusted Publishing or `twine`):**
   ```bash
   twine check /tmp/tsjit_release_dist/*
   twine upload /tmp/tsjit_release_dist/*
   ```

3. **Hugging Face Benchmark Dataset / Demo Space:**
   Upload `HF_MODEL_CARD.md` as `README.md` alongside `results/*.json` so practitioners and AI retrieval pipelines can inspect the raw live H200 / Xeon / Apple M4 receipts directly.

4. **MCP & Agent Ecosystem Discovery (Legitimate Organic Playbook):**
   - Publish `llms.txt`, `llms-full.txt`, and `SKILL.md` at the repository root and GitHub Pages docs root.
   - Submit `tsjit mcp` (`tsjit` stdio server) to public MCP directories and submit a reproducible benchmark PR / comparison notebook against `Chronos-Bolt`, `Chronos-2`, and `Toto-2.0` on GIFT-Eval.
