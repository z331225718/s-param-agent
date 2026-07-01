import io
import os
import sys
import tempfile
import unittest

import numpy as np
import skrf as rf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import app as app_module
import code_agent
import nl_parser
import s_params as sp


def make_network(name="fixture"):
    freq = rf.Frequency(1, 2, 5, unit="ghz")
    s = np.zeros((5, 2, 2), dtype=complex)
    s[:, 0, 0] = 0.1
    s[:, 1, 0] = 0.8
    s[:, 0, 1] = 0.05
    s[:, 1, 1] = 0.2
    return rf.Network(frequency=freq, s=s, z0=50, name=name)


def make_named_network(port_names, z0=50, transmissions=None, name="named"):
    nports = len(port_names)
    freq = rf.Frequency(1, 2, 3, unit="ghz")
    s = np.zeros((3, nports, nports), dtype=complex)
    for i in range(nports):
        s[:, i, i] = 0.05
    if transmissions:
        for m, n, value in transmissions:
            s[:, m, n] = value
    ntwk = rf.Network(frequency=freq, s=s, z0=z0, name=name)
    ntwk.port_names = list(port_names)
    return ntwk


def register_network(ntwk, session="test", name="fixture"):
    app_module.sessions.clear()
    app_module.sessions[session] = {
        "networks": {
            name: {
                "path": "",
                "_ntwk": ntwk,
                "nports": ntwk.nports,
                "f_min": float(ntwk.f[0]),
                "f_max": float(ntwk.f[-1]),
                "npoints": len(ntwk.f),
                "params": sp.list_params(ntwk),
            }
        }
    }


def register_networks(networks, session="test"):
    app_module.sessions.clear()
    app_module.sessions[session] = {"networks": {}}
    for name, ntwk in networks.items():
        app_module.sessions[session]["networks"][name] = {
            "path": "",
            "_ntwk": ntwk,
            "nports": ntwk.nports,
            "f_min": float(ntwk.f[0]),
            "f_max": float(ntwk.f[-1]),
            "npoints": len(ntwk.f),
            "params": sp.list_params(ntwk),
        }


class TestNaturalLanguageParser(unittest.TestCase):
    def test_export_csv_keeps_csv_format(self):
        ops = nl_parser.parse("导出 S21 为 CSV")

        self.assertEqual(1, len(ops))
        self.assertEqual("export", ops[0].action)
        self.assertEqual("csv", ops[0].export_format)
        self.assertEqual(["S21"], ops[0].params)

    def test_export_touchstone_keeps_touchstone_format(self):
        ops = nl_parser.parse("导出当前网络为 touchstone")

        self.assertEqual(1, len(ops))
        self.assertEqual("touchstone", ops[0].export_format)

    def test_compare_fills_two_available_networks(self):
        ops = nl_parser.parse("对比 A 和 B 的 S21", ["A", "B"])

        self.assertEqual(1, len(ops))
        self.assertEqual("compare", ops[0].action)
        self.assertEqual(["A", "B"], ops[0].compare_networks)
        self.assertEqual(["S21"], ops[0].params)

    def test_multi_port_params_are_preserved(self):
        ops = nl_parser.parse("画 S10_9 和 S2_1", ["fixture"])

        self.assertEqual(["S10_9", "S2_1"], ops[0].params)

    def test_plot_after_cascade_keeps_empty_target(self):
        ops = nl_parser.parse("级联 A 和 B，然后画 S21", ["A", "B"])

        self.assertEqual(["cascade", "plot"], [op.action for op in ops])
        self.assertEqual("A", ops[0].target)
        self.assertEqual("B", ops[0].cascade_with)
        self.assertFalse(ops[1].target)


class TestRfCoreRegressions(unittest.TestCase):
    def test_slice_freq_accepts_numeric_and_string_ranges(self):
        ntwk = make_network()

        numeric = sp.slice_freq(ntwk, 1.2e9, 1.6e9)
        string = sp.slice_freq(ntwk, "1.2-1.6ghz")

        self.assertEqual(list(numeric.f), list(string.f))
        self.assertEqual(2, len(string.f))

    def test_slice_freq_rejects_reversed_or_empty_ranges_and_keeps_original(self):
        ntwk = make_network()
        original_f = ntwk.f.copy()

        with self.assertRaises(ValueError):
            sp.slice_freq(ntwk, "2-1ghz")
        with self.assertRaises(ValueError):
            sp.slice_freq(ntwk, "5-6ghz")

        np.testing.assert_array_equal(original_f, ntwk.f)
        self.assertEqual(5, len(ntwk.f))

    def test_renormalize_returns_new_network_without_mutating_original(self):
        ntwk = make_network()

        result = sp.renormalize(ntwk, 75)

        self.assertIsInstance(result, rf.Network)
        self.assertIsNot(result, ntwk)
        self.assertTrue(np.allclose(result.z0, 75))
        self.assertTrue(np.allclose(ntwk.z0, 50))

    def test_parse_network_params_supports_large_ports_and_rejects_ambiguity(self):
        self.assertEqual(("S", 9, 8), sp.parse_network_param("S10_9", 12))
        self.assertEqual(("Z", 9, 9), sp.parse_network_param("Z10_10", 12))
        self.assertEqual(9, sp.parse_vswr_param("VSWR10", 12))
        self.assertEqual([0], sp._parse_vswr_params(make_network(), ["S11"]))

        with self.assertRaises(ValueError):
            sp.parse_network_param("S1010", 12)
        with self.assertRaises(ValueError):
            sp.parse_network_param("S3_1", 2)


class TestExportsAndApi(unittest.TestCase):
    def setUp(self):
        self.ntwk = make_network()

    def test_save_csv_accepts_binary_stream(self):
        out = io.BytesIO()

        result = sp.save_csv(self.ntwk, ["S21"], out)

        self.assertIs(result, out)
        text = out.getvalue().decode("utf-8")
        self.assertIn("Freq_GHz,S21_dB,S21_deg", text)

    def test_save_touchstone_accepts_binary_stream(self):
        out = io.BytesIO()

        result = sp.save_touchstone(self.ntwk, out)

        self.assertIs(result, out)
        text = out.getvalue().decode("utf-8", errors="ignore")
        self.assertIn("#", text)
        self.assertIn("S DB R", text.upper())

    def test_export_csv_endpoint_returns_download(self):
        register_network(self.ntwk)

        with app_module.app.test_client() as client:
            resp = client.post("/api/export/csv", json={
                "session": "test",
                "network": "fixture",
                "params": ["S21"],
            })

        self.assertEqual(200, resp.status_code)
        self.assertIn(b"Freq_GHz,S21_dB,S21_deg", resp.data)

    def test_chart_html_endpoint_handles_direct_response(self):
        register_network(self.ntwk)

        with app_module.app.test_client() as client:
            resp = client.post("/api/chart/html", json={
                "session": "test",
                "type": "db",
                "title": "Fixture",
                "networks": [{"name": "fixture", "params": ["S21"]}],
            })

        self.assertEqual(200, resp.status_code)
        payload = resp.get_json()
        self.assertIn("<html", payload["html"])
        self.assertEqual("Fixture", payload["title"])

    def test_chart_html_escapes_title_and_header(self):
        register_network(self.ntwk)
        malicious = "<script>alert(1)</script>"

        with app_module.app.test_client() as client:
            resp = client.post("/api/chart/html", json={
                "session": "test",
                "type": "db",
                "title": malicious,
                "networks": [{"name": "fixture", "params": ["S21"]}],
            })

        self.assertEqual(200, resp.status_code)
        html = resp.get_json()["html"]
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<h2><script>", html)
        self.assertNotIn("</script><script>", html)

    def test_chart_rejects_invalid_param_and_frequency_with_400(self):
        register_network(self.ntwk)

        with app_module.app.test_client() as client:
            bad_param = client.post("/api/chart", json={
                "session": "test",
                "type": "db",
                "networks": [{"name": "fixture", "params": ["S1010"]}],
            })
            bad_freq = client.post("/api/chart", json={
                "session": "test",
                "type": "db",
                "freq_range": ["5ghz", "6ghz"],
                "networks": [{"name": "fixture", "params": ["S21"]}],
            })

        self.assertEqual(400, bad_param.status_code)
        self.assertEqual(400, bad_freq.status_code)

    def test_compare_rejects_invalid_param_and_frequency_with_400(self):
        register_networks({"A": make_network("A"), "B": make_network("B")})

        with app_module.app.test_client() as client:
            bad_param = client.post("/api/compare", json={
                "session": "test",
                "networks": ["A", "B"],
                "params": ["S1010"],
            })
            bad_freq = client.post("/api/compare", json={
                "session": "test",
                "networks": ["A", "B"],
                "params": ["S21"],
                "freq_range": ["5ghz", "6ghz"],
            })

        self.assertEqual(400, bad_param.status_code)
        self.assertEqual(400, bad_freq.status_code)

    def test_chat_plot_after_cascade_uses_cascade_result(self):
        register_networks({"A": make_network("A"), "B": make_network("B")})

        with app_module.app.test_client() as client:
            resp = client.post("/api/chat", json={
                "session": "test",
                "text": "级联 A 和 B，然后画 S21",
            })

        self.assertEqual(200, resp.status_code)
        payload = resp.get_json()
        self.assertTrue(payload["handled"])
        self.assertEqual(["text", "chart"], [r["type"] for r in payload["results"]])
        self.assertIn("A+B", app_module.sessions["test"]["networks"])
        self.assertTrue(payload["results"][1]["title"].startswith("A+B"))


class TestPlotCompatibility(unittest.TestCase):
    def test_demo_plot_functions_are_compatible_with_installed_plotly(self):
        ntwk = make_network()

        figures = [
            sp.plot_s_db(ntwk, ["S21"], title="Compatibility dB"),
            sp.plot_s_deg(ntwk, ["S21"], title="Compatibility Phase"),
            sp.plot_s_smith(ntwk, ["S11"], title="Compatibility Smith"),
            sp.plot_vswr(ntwk, [0], title="Compatibility VSWR"),
            sp.plot_group_delay(ntwk, ["S21"], title="Compatibility GD"),
        ]

        self.assertTrue(all(fig.data for fig in figures))


class TestNetworkInspector(unittest.TestCase):
    def test_power_network_uses_zmag_quick_action(self):
        import network_inspector

        ntwk = make_named_network(["VDD", "GND"], z0=0.1)
        info = network_inspector.inspect_network(ntwk)

        self.assertEqual("power", info["network_kind"])
        self.assertEqual(["Z1_1", "Z2_2"], info["quick_actions"][0]["params"])
        self.assertEqual("zmag", info["quick_actions"][0]["chart_type"])

    def test_signal_names_generate_rl_il_next_fext(self):
        import network_inspector

        ntwk = make_named_network(
            ["J1_NET1", "J2_NET1", "J1_NET2", "J2_NET2"],
            z0=50,
            transmissions=[(1, 0, 0.8), (0, 1, 0.8), (3, 2, 0.75), (2, 3, 0.75)],
        )
        info = network_inspector.inspect_network(ntwk)
        actions = {a["id"]: a for a in info["quick_actions"]}

        self.assertEqual("signal", info["network_kind"])
        self.assertIn({"a": 0, "b": 1, "source": "name", "confidence": "high"}, info["port_pairs"])
        self.assertEqual(["S1_1", "S2_2", "S3_3", "S4_4"], actions["rl"]["params"])
        self.assertIn("S2_1", actions["il"]["params"])
        self.assertIn("S4_3", actions["il"]["params"])
        self.assertTrue(actions["next"]["params"])
        self.assertTrue(actions["fext"]["params"])

    def test_matrix_pairing_uses_closest_to_zero_db(self):
        import network_inspector

        ntwk = make_named_network(
            ["P1", "P2", "P3", "P4"],
            z0=50,
            transmissions=[
                (0, 1, 0.8), (1, 0, 0.8),
                (2, 3, 0.7), (3, 2, 0.7),
                (0, 2, 0.05), (2, 0, 0.05),
            ],
        )
        pairs = network_inspector.detect_port_pairs(ntwk)

        self.assertIn({"a": 0, "b": 1, "source": "matrix", "confidence": "high"}, pairs)
        self.assertIn({"a": 2, "b": 3, "source": "matrix", "confidence": "high"}, pairs)


class TestNetworkMetadataApi(unittest.TestCase):
    def test_list_networks_includes_inspection_metadata(self):
        ntwk = make_named_network(["VDD", "GND"], z0=0.1)
        register_network(ntwk)

        with app_module.app.test_client() as client:
            resp = client.get("/api/networks?session=test")

        self.assertEqual(200, resp.status_code)
        entry = resp.get_json()["networks"]["fixture"]
        self.assertEqual("power", entry["network_kind"])
        self.assertEqual("zmag", entry["quick_actions"][0]["chart_type"])

    def test_zmag_chart_returns_impedance_magnitude(self):
        ntwk = make_named_network(["VDD", "GND"], z0=0.1)
        register_network(ntwk)

        with app_module.app.test_client() as client:
            resp = client.post("/api/chart", json={
                "session": "test",
                "type": "zmag",
                "networks": [{"name": "fixture", "params": ["Z1_1"], "label": "fixture"}],
                "title": "PDN impedance",
            })

        self.assertEqual(200, resp.status_code)
        fig = resp.get_json()
        self.assertEqual("Magnitude |Z| (ohm)", fig["layout"]["yaxis"]["title"])
        self.assertEqual("fixture Z1_1", fig["data"][0]["name"])

    def test_large_port_file_registers_metadata_without_eager_load(self):
        app_module.sessions.clear()
        fd, path = tempfile.mkstemp(suffix=".s8p")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("# GHZ S RI R 50\n")
                f.write("1.0 0 0\n")
                f.write("2.0 0 0\n")

            info = app_module._register_path("large", path, name="big")

            self.assertFalse(info["loaded"])
            self.assertEqual(8, info["nports"])
            self.assertEqual(64, info["total_params"])
            self.assertIsNone(app_module.sessions["large"]["networks"]["big"]["_ntwk"])

            with app_module.app.test_client() as client:
                resp = client.get("/api/networks/big/status?session=large")

            self.assertEqual(200, resp.status_code)
            self.assertFalse(resp.get_json()["loaded"])
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_dashboard_does_not_use_inner_html_for_user_strings(self):
        dashboard = os.path.join(ROOT, "scripts", "templates", "dashboard.html")

        with open(dashboard, "r", encoding="utf-8") as f:
            text = f.read()

        self.assertNotIn("innerHTML", text)
        self.assertNotIn("insertAdjacentHTML", text)


class TestCodeValidator(unittest.TestCase):
    def test_getattr_os_system_is_rejected(self):
        ok, msg = code_agent.validate_code("getattr(os, 'system')('echo unsafe')\nfig = None")

        self.assertFalse(ok)
        self.assertIn("getattr", msg)

    def test_globals_os_system_is_rejected(self):
        ok, msg = code_agent.validate_code("globals()['os'].system('echo unsafe')\nfig = None")

        self.assertFalse(ok)
        self.assertIn("globals", msg)

    def test_direct_os_system_still_rejected(self):
        ok, msg = code_agent.validate_code("os.system('echo unsafe')\nfig = None")

        self.assertFalse(ok)
        self.assertIn("os.system", msg)

    def test_pathlib_import_is_rejected(self):
        ok, msg = code_agent.validate_code("from pathlib import Path\nfig = None")

        self.assertFalse(ok)
        self.assertIn("pathlib", msg)

    def test_path_file_methods_are_rejected(self):
        ok_read, msg_read = code_agent.validate_code("Path('x').read_text()\nfig = None")
        ok_write, msg_write = code_agent.validate_code("Path('x').write_text('x')\nfig = None")

        self.assertFalse(ok_read)
        self.assertIn("read_text", msg_read)
        self.assertFalse(ok_write)
        self.assertIn("write_text", msg_write)

    def test_network_constructors_are_rejected(self):
        cases = [
            "rf.Network('x.s2p')\nfig = None",
            "skrf.Network('x.s2p')\nfig = None",
            "from skrf import Network\nfig = None",
        ]
        for code in cases:
            with self.subTest(code=code):
                ok, msg = code_agent.validate_code(code)
                self.assertFalse(ok)
                self.assertIn("Network", msg)


class TestApiGraphDependency(unittest.TestCase):
    def test_api_graph_imports_with_declared_dependencies(self):
        import api_graph

        graph = api_graph.get_graph()
        self.assertGreaterEqual(len(graph.nodes), 1)


if __name__ == "__main__":
    unittest.main()
