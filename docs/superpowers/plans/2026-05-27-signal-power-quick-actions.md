# Signal Power Quick Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backend-driven quick actions that distinguish signal and power Touchstone files and recommend RL, IL, NEXT, FEXT, or mag(Zxx) parameters.

**Architecture:** Put network classification, port-pair inference, and quick-action generation in a small Python helper module consumed by Flask upload/load/list responses. Extend chart generation with a `zmag` chart type for power impedance magnitude. Keep frontend logic thin: render backend `quick_actions`, apply selected params/chart type, and retain generic fallback buttons.

**Tech Stack:** Python 3, Flask, scikit-rf, NumPy, Plotly, stdlib unittest, browser JavaScript in `dashboard.html`.

---

### Task 1: Backend Analysis Tests

**Files:**
- Create: `scripts/network_inspector.py`
- Modify: `tests/test_regressions.py`

- [ ] **Step 1: Write failing tests**

Add tests that build synthetic networks with `port_names` and `z0`, then assert:
- low `z0` or power names produce `network_kind == "power"` and a `zmag` quick action.
- 50 ohm signal names produce `network_kind == "signal"` and `RL`, `IL`, `NEXT`, `FEXT`.
- name-paired ports produce IL params for the paired endpoints.
- when names are absent, matrix inference pairs ports by the first frequency point's closest-to-0 dB non-self transmission.

- [ ] **Step 2: Run RED**

Run: `python3 -m unittest tests.test_regressions.TestNetworkInspector -v`
Expected: import or assertion failures because `network_inspector.py` does not exist.

### Task 2: Network Inspector Implementation

**Files:**
- Create: `scripts/network_inspector.py`

- [ ] **Step 1: Implement metadata helpers**

Create functions:
- `get_port_names(ntwk) -> list[str]`
- `classify_network(ntwk) -> str`
- `detect_port_pairs(ntwk) -> list[dict]`
- `build_quick_actions(ntwk) -> list[dict]`
- `inspect_network(ntwk) -> dict`

- [ ] **Step 2: Implement name and matrix pairing**

Normalize names by removing endpoint tokens (`J1`, `J2`, `NEAR`, `FAR`, `IN`, `OUT`, `TX`, `RX`) and separators. For matrix inference, use `ntwk.s_db[0]`, ignore diagonal values, and choose the largest dB value per row.

- [ ] **Step 3: Run GREEN**

Run: `python3 -m unittest tests.test_regressions.TestNetworkInspector -v`
Expected: all inspector tests pass.

### Task 3: Flask Metadata and zmag Chart

**Files:**
- Modify: `scripts/app.py`
- Modify: `tests/test_regressions.py`

- [ ] **Step 1: Write failing endpoint/chart tests**

Add tests that register/upload-like networks and assert response metadata includes `network_kind`, `port_pairs`, and `quick_actions`; add `/api/chart` test for `type: "zmag"` with params like `Z11`.

- [ ] **Step 2: Add metadata to session entries and responses**

Import `network_inspector` in `app.py`. Add a helper that merges `inspect_network(ntwk)` into network entries and JSON responses in upload, batch upload, local load, glob load, cascade, deembed, and `/api/networks`.

- [ ] **Step 3: Add zmag parsing and trace/layout**

Extend `_parse_param()` to accept `Z11`; add `_make_zmag_trace()` using `np.abs(ntwk.z[:, m, n])`; add `chart_type == "zmag"` in `generate_chart()` and `_make_layout()`.

- [ ] **Step 4: Run endpoint/chart tests**

Run: `python3 -m unittest tests.test_regressions.TestNetworkMetadataApi -v`
Expected: all endpoint/chart tests pass.

### Task 4: Dashboard Quick Buttons

**Files:**
- Modify: `scripts/templates/dashboard.html`

- [ ] **Step 1: Render backend quick actions**

Update `renderParamTags()` so backend `quick_actions` render before generic fallback buttons when available.

- [ ] **Step 2: Apply action chart type**

Add `applyQuickAction(index)` that sets `state.selectedParams` from the action and sets `state.chartType` from action `chart_type`, defaulting to `db`.

- [ ] **Step 3: Use selected chart type for chart and export**

Change `generateChart()` and `exportHTML()` payload `type` from hardcoded `db` to `state.chartType || "db"`.

### Task 5: Full Verification

**Files:**
- No production edits.

- [ ] **Step 1: Run syntax and regression tests**

Run:
- `python3 -m py_compile scripts/*.py`
- `python3 -m unittest tests.test_regressions -v`

- [ ] **Step 2: Run smoke verification**

Run: `python3 scripts/demo_verify.py`
Expected: all checks pass.
