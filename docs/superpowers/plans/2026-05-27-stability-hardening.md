# Stability Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the highest-impact runtime regressions found in the S-parameter agent: export format parsing, chart HTML export, CSV/Touchstone downloads, and obvious sandbox escapes.

**Architecture:** Add focused stdlib `unittest` regression tests under `tests/` so no new test dependency is needed. Keep changes narrow: parser behavior in `scripts/nl_parser.py`, file/stream export support in `scripts/s_params.py`, response handling in `scripts/app.py`, and AST validation/wrapper globals in `scripts/code_agent.py`.

**Tech Stack:** Python 3, Flask test client, unittest, scikit-rf, NumPy, Plotly.

---

### Task 1: Regression Tests

**Files:**
- Create: `tests/test_regressions.py`

- [ ] **Step 1: Write failing tests**

Create tests for four current failures: CSV export parsing, Flask HTML export response handling, in-memory export downloads, and sandbox escapes.

- [ ] **Step 2: Run tests to verify RED**

Run: `python3 -m unittest tests.test_regressions -v`
Expected: at least the parser/export/sandbox tests fail on current code.

### Task 2: Parser Export Format

**Files:**
- Modify: `scripts/nl_parser.py:201-211`

- [ ] **Step 1: Remove the unconditional HTML override**

Delete the final `op.export_format = "html"` line so earlier `csv`, `touchstone`, and `html` decisions are preserved.

- [ ] **Step 2: Run parser test**

Run: `python3 -m unittest tests.test_regressions.TestNaturalLanguageParser -v`
Expected: parser export tests pass.

### Task 3: Export Streams and HTML Response Handling

**Files:**
- Modify: `scripts/s_params.py:51-61`
- Modify: `scripts/s_params.py:64-91`
- Modify: `scripts/app.py:1052-1104`
- Modify: `scripts/app.py:1111-1131`
- Modify: `scripts/app.py:1291-1311`

- [ ] **Step 1: Support file-like CSV export**

Update `save_csv()` to detect `write` objects. For binary streams, write encoded CSV bytes; for text streams, write string CSV content.

- [ ] **Step 2: Support file-like Touchstone export**

For streams, write Touchstone output to a temporary path with `ntwk.write_touchstone()`, read bytes back, write to the stream, and clean temporary files.

- [ ] **Step 3: Fix chart HTML response handling**

Replace tuple indexing around `generate_chart()` with response/status normalization so both direct `Response` and `(Response, status)` paths work.

- [ ] **Step 4: Run export/API tests**

Run: `python3 -m unittest tests.test_regressions.TestExportsAndApi -v`
Expected: all export/API tests pass.

### Task 4: Sandbox Validation

**Files:**
- Modify: `scripts/code_agent.py:77-80`
- Modify: `scripts/code_agent.py:214-227`
- Modify: `scripts/code_agent.py:375-407`

- [ ] **Step 1: Block introspection escape builtins**

Add `getattr`, `setattr`, `delattr`, `globals`, `locals`, and `vars` to forbidden builtins.

- [ ] **Step 2: Remove `os` from generated-code wrapper globals**

Keep parent process `os` usage, but do not import `os` into the child wrapper unless required by trusted wrapper code.

- [ ] **Step 3: Run sandbox tests**

Run: `python3 -m unittest tests.test_regressions.TestCodeValidator -v`
Expected: direct and indirect `os.system` attempts are rejected.

### Task 5: Full Verification

**Files:**
- No production edits.

- [ ] **Step 1: Run syntax check**

Run: `python3 -m py_compile scripts/*.py`
Expected: no output and exit code 0.

- [ ] **Step 2: Run regression tests**

Run: `python3 -m unittest tests.test_regressions -v`
Expected: all tests pass.

- [ ] **Step 3: Run smoke verification**

Run: `python3 scripts/demo_verify.py`
Expected: script completes and writes artifacts to `scripts/demo_output/`.
