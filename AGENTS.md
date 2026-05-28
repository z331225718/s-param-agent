# Repository Guidelines

## Project Structure & Module Organization

This is a Python S-parameter conversational agent built on Flask, scikit-rf, and Plotly.

- `scripts/app.py` runs the local Flask dashboard and API endpoints.
- `scripts/code_agent.py`, `llm_chat.py`, and `nl_parser.py` handle natural-language and LLM-driven workflows.
- `scripts/s_params.py` contains reusable S-parameter loading, processing, plotting, and export helpers.
- `scripts/api_graph.py`, `api_refs.py`, `api_index.json`, and `extract_apis.py` support API lookup.
- `scripts/templates/dashboard.html` is the Web UI.
- `scripts/demo_verify.py` is the end-to-end smoke verification script.
- `test_filter.s2p` is sample Touchstone data; generated outputs go under `scripts/demo_output/`.

## Build, Test, and Development Commands

```bash
pip install -r scripts/requirements.txt
```
Installs runtime dependencies.

```bash
cp scripts/config.example.json scripts/config.json
python scripts/app.py
```
Starts the dashboard at `http://localhost:5050`. Set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` instead of committing secrets.

```bash
python scripts/demo_verify.py
```
Generates sample data, exercises processing and plotting, and writes artifacts to `scripts/demo_output/`.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Prefer clear module-level helper functions and descriptive snake_case names. Keep route handlers in `app.py` thin where practical and place reusable RF logic in `s_params.py`. Avoid broad filesystem or subprocess access in agent-executed code; the code agent relies on allowlisted imports and AST validation.

No formatter or linter configuration is currently checked in. Keep edits consistent with nearby code and avoid unrelated style churn.

## Testing Guidelines

There is no formal unit test suite yet. Before submitting changes, run `python scripts/demo_verify.py` and confirm it completes without assertion failures. For Flask route changes, also start `python scripts/app.py` and verify `/health` plus the affected dashboard/API flow. Name future tests `test_*.py`; prefer generated Touchstone fixtures.

## Commit & Pull Request Guidelines

Recent history uses short prefixes such as `fix:` and `docs:` followed by a concise description, often in Chinese. Follow that style, for example `fix: handle empty agent request` or `docs: update dashboard usage`.

Pull requests should include a summary, verification commands, linked issues if any, and screenshots or generated HTML paths for UI/plot changes. Never include API keys, local `config.json`, or large generated reports unless intentionally reviewed.

## Security & Configuration Tips

Use `scripts/config.example.json` as the template only. Store real credentials in environment variables or an untracked `scripts/config.json`. Treat uploaded Touchstone files and generated code as untrusted inputs; preserve validation, timeouts, and import restrictions when modifying agent execution.
