import io
import os
import sys
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


class TestApiGraphDependency(unittest.TestCase):
    def test_api_graph_imports_with_declared_dependencies(self):
        import api_graph

        graph = api_graph.get_graph()
        self.assertGreaterEqual(len(graph.nodes), 1)


if __name__ == "__main__":
    unittest.main()
