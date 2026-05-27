#!/usr/bin/env python3
"""
S-Parameter Agent — 原生 GUI 版本
基于 tkinter，本地直接读取 .sNp 文件，绕过 HTTP 上传限制。
支持：批量加载 / 链式级联 / 多文件对比 / LLM Agent
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import webbrowser
import tempfile

# 确保能找到同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skrf as rf
import s_params as sp
import nl_parser
import code_agent


class SParamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("S-Parameter Agent")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        # ── 状态 ──
        self.networks = {}   # name → {"path", "_ntwk", "nports", ...}
        self.selected = []   # 当前选中的网络名列表
        self.selected_params = []
        self.chain = []
        self.show_diff = tk.BooleanVar(value=True)

        # ── 构建 UI ──
        self._build_ui()

        # 定期刷新加载状态
        self._poll_loading()

    # ═══════════════════════════════════════
    #  UI 构建
    # ═══════════════════════════════════════

    def _build_ui(self):
        # 主面板：左侧控制 / 右侧日志
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=420)
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        # ── 左侧控制面板 ──
        self._build_left_panel(left)
        # ── 右侧日志 ──
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        # 滚动容器
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        f = scroll_frame

        # ── 文件加载 ──
        grp = ttk.LabelFrame(f, text="📂 文件加载", padding=8)
        grp.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(grp, text="选择 .sNp 文件", command=self._load_files).pack(fill=tk.X, pady=2)
        ttk.Button(grp, text="加载整个目录", command=self._load_directory).pack(fill=tk.X, pady=2)
        row = ttk.Frame(grp)
        row.pack(fill=tk.X, pady=2)
        self.path_entry = ttk.Entry(row)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="路径加载", command=self._load_by_path).pack(side=tk.RIGHT, padx=(4, 0))

        # ── 已加载网络列表 ──
        grp = ttk.LabelFrame(f, text="📋 已加载网络", padding=8)
        grp.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.net_list = tk.Listbox(grp, selectmode=tk.EXTENDED, height=8,
                                   exportselection=False)
        self.net_list.pack(fill=tk.BOTH, expand=True)
        self.net_list.bind("<<ListboxSelect>>", self._on_net_select)
        row = ttk.Frame(grp)
        row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="移除选中", command=self._remove_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="清空全部", command=self._clear_all).pack(side=tk.LEFT, padx=2)

        # ── 参数选择 ──
        grp = ttk.LabelFrame(f, text="🎯 S 参数", padding=8)
        grp.pack(fill=tk.X, padx=6, pady=4)
        self.param_label = ttk.Label(grp, text="请先选择网络")
        self.param_label.pack(anchor=tk.W)
        self.param_list = tk.Listbox(grp, selectmode=tk.EXTENDED, height=4,
                                     exportselection=False)
        self.param_list.pack(fill=tk.X)
        row = ttk.Frame(grp)
        row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="全选", command=lambda: self._param_quick("all")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="反射", command=lambda: self._param_quick("refl")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="传输", command=lambda: self._param_quick("trans")).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(grp)
        row2.pack(fill=tk.X)
        self.param_entry = ttk.Entry(row2)
        self.param_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.param_entry.bind("<Return>", lambda e: self._add_custom_param())
        ttk.Button(row2, text="+", width=3, command=self._add_custom_param).pack(side=tk.RIGHT)

        # ── 图表类型 ──
        grp = ttk.LabelFrame(f, text="📊 图表类型", padding=8)
        grp.pack(fill=tk.X, padx=6, pady=4)
        self.chart_type = tk.StringVar(value="db")
        types = [("dB 幅度", "db"), ("相位", "deg"), ("Smith 圆图", "smith"),
                 ("VSWR", "vswr"), ("群时延", "groupdelay"), ("线性幅度", "mag")]
        for text, val in types:
            ttk.Radiobutton(grp, text=text, variable=self.chart_type, value=val).pack(anchor=tk.W)

        # ── 操作按钮 ──
        grp = ttk.Frame(f, padding=8)
        grp.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(grp, text="🔄 生成图表", command=self._generate_chart).pack(fill=tk.X, pady=2)
        ttk.Button(grp, text="📊 多文件对比", command=self._compare).pack(fill=tk.X, pady=2)
        ttk.Button(grp, text="🔗 链式级联", command=self._cascade_chain).pack(fill=tk.X, pady=2)
        ttk.Button(grp, text="💾 导出 CSV", command=self._export_csv).pack(fill=tk.X, pady=2)

        # ── 对比选项 ──
        grp = ttk.LabelFrame(f, text="📊 对比选项", padding=8)
        grp.pack(fill=tk.X, padx=6, pady=4)
        ttk.Checkbutton(grp, text="显示差异曲线 (Δ)", variable=self.show_diff).pack(anchor=tk.W)
        row = ttk.Frame(grp)
        row.pack(fill=tk.X)
        ttk.Label(row, text="参考网络:").pack(side=tk.LEFT)
        self.ref_var = tk.StringVar()
        self.ref_combo = ttk.Combobox(row, textvariable=self.ref_var, width=15)
        self.ref_combo.pack(side=tk.LEFT, padx=4)

        # ── LLM Agent ──
        grp = ttk.LabelFrame(f, text="🤖 LLM Agent (自然语言)", padding=8)
        grp.pack(fill=tk.X, padx=6, pady=4)
        self.agent_entry = ttk.Entry(grp)
        self.agent_entry.pack(fill=tk.X, pady=2)
        self.agent_entry.bind("<Return>", lambda e: self._agent_chat())
        ttk.Button(grp, text="发送", command=self._agent_chat).pack(fill=tk.X)

        # ── 状态栏 ──
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(f, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=2).pack(fill=tk.X, side=tk.BOTTOM)

    def _build_right_panel(self, parent):
        # 日志输出
        grp = ttk.LabelFrame(parent, text="📝 日志", padding=4)
        grp.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log_text = tk.Text(grp, wrap=tk.WORD, state=tk.DISABLED,
                                font=("Consolas", 10))
        scroll = ttk.Scrollbar(grp, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ═══════════════════════════════════════
    #  日志
    # ═══════════════════════════════════════

    def log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ═══════════════════════════════════════
    #  文件加载
    # ═══════════════════════════════════════

    def _load_files(self):
        paths = filedialog.askopenfilenames(
            title="选择 Touchstone 文件",
            filetypes=[("Touchstone", "*.s*p *.ts"), ("All files", "*.*")]
        )
        for p in paths:
            self._add_network(p)

    def _load_directory(self):
        directory = filedialog.askdirectory(title="选择目录")
        if not directory:
            return
        import glob as g
        for ext in ["*.s1p", "*.s2p", "*.s3p", "*.s4p", "*.s5p", "*.s6p",
                     "*.s7p", "*.s8p", "*.sNp", "*.ts"]:
            for p in g.glob(os.path.join(directory, "**", ext), recursive=True):
                self._add_network(p)

    def _load_by_path(self):
        path = self.path_entry.get().strip()
        if not path:
            return
        if os.path.isdir(path):
            self._load_directory()
            return
        if not os.path.exists(path):
            messagebox.showerror("错误", f"文件不存在: {path}")
            return
        self._add_network(path)
        self.path_entry.delete(0, tk.END)

    def _add_network(self, path):
        path = os.path.abspath(path)
        # 快速读头
        header = _read_header(path)
        if header is None:
            self.log(f"❌ 无法解析: {os.path.basename(path)}")
            return

        name = os.path.splitext(os.path.basename(path))[0]
        if name in self.networks:
            base = name
            idx = 2
            while f"{base}_{idx}" in self.networks:
                idx += 1
            name = f"{base}_{idx}"

        nports = header["nports"]
        all_params = [f"S{m+1}{n+1}" for m in range(nports) for n in range(nports)]

        self.networks[name] = {
            "path": path,
            "_ntwk": None,
            "_loading": False,
            "nports": nports,
            "f_min": header["f_min"],
            "f_max": header["f_max"],
            "npoints": header["npoints"],
            "params": all_params,
        }
        self.log(f"📂 {name}  {nports}端口  {header['npoints']}点  "
                 f"{header['f_min']/1e9:.4f}–{header['f_max']/1e9:.4f} GHz")
        self._refresh_net_list()

    def _get_network(self, name):
        """获取网络对象（延迟加载）。"""
        entry = self.networks.get(name)
        if not entry:
            return None
        if entry["_ntwk"] is not None:
            return entry["_ntwk"]
        # 触发加载
        if not entry["_loading"]:
            entry["_loading"] = True
            self.status(f"⏳ 正在加载 {name}...")
            t = threading.Thread(target=self._load_worker, args=(name,), daemon=True)
            t.start()
        return None  # 还没加载完

    def _load_worker(self, name):
        entry = self.networks[name]
        try:
            ntwk = rf.Network(entry["path"])
            entry["_ntwk"] = ntwk
            self.log(f"✅ {name} 已载入内存 ({ntwk.nports}端口, {len(ntwk.f)}点)")
        except Exception as e:
            self.log(f"❌ {name} 加载失败: {e}")
        finally:
            entry["_loading"] = False

    def _poll_loading(self):
        """定期检查加载状态，刷新 UI。"""
        updated = False
        for name, entry in self.networks.items():
            if entry["_loading"] and entry["_ntwk"] is not None:
                entry["_loading"] = False
                updated = True
        if updated:
            self._refresh_net_list()
        self.root.after(2000, self._poll_loading)

    # ═══════════════════════════════════════
    #  网络列表
    # ═══════════════════════════════════════

    def _refresh_net_list(self):
        self.net_list.delete(0, tk.END)
        for name, entry in self.networks.items():
            loaded = "✓" if entry["_ntwk"] is not None else "⏳"
            self.net_list.insert(tk.END,
                f"{loaded} {name}  [{entry['nports']}端口, {entry['npoints']}点]")
        # 更新参考网络下拉
        names = list(self.networks.keys())
        self.ref_combo["values"] = names
        if names and not self.ref_var.get():
            self.ref_var.set(names[0])

    def _on_net_select(self, event):
        sel = self.net_list.curselection()
        all_names = list(self.networks.keys())
        self.selected = [all_names[i] for i in sel]
        if self.selected:
            self._refresh_param_list()

    def _remove_selected(self):
        for name in list(self.selected):
            if name in self.networks:
                del self.networks[name]
        self.selected = []
        self.selected_params = []
        self._refresh_net_list()
        self._refresh_param_list()

    def _clear_all(self):
        self.networks.clear()
        self.selected = []
        self.selected_params = []
        self._refresh_net_list()
        self._refresh_param_list()

    # ═══════════════════════════════════════
    #  参数选择
    # ═══════════════════════════════════════

    def _refresh_param_list(self):
        self.param_list.delete(0, tk.END)
        if not self.selected:
            self.param_label.config(text="请先选择网络")
            return
        name = self.selected[0]
        entry = self.networks.get(name)
        if not entry:
            return
        self.param_label.config(
            text=f"{name}  {entry['nports']}端口 · {entry['nports']**2}个参数")
        for p in entry["params"]:
            self.param_list.insert(tk.END, p)
        # 恢复选中
        for i, p in enumerate(entry["params"]):
            if p in self.selected_params:
                self.param_list.selection_set(i)

    def _param_quick(self, kind):
        if not self.selected:
            return
        name = self.selected[0]
        entry = self.networks.get(name)
        if not entry:
            return
        nports = entry["nports"]
        new_params = []
        for m in range(1, nports + 1):
            for n in range(1, nports + 1):
                s = f"S{m}{n}"
                if kind == "all":
                    new_params.append(s)
                elif kind == "refl" and m == n:
                    new_params.append(s)
                elif kind == "trans" and m != n:
                    new_params.append(s)
        self.selected_params = list(dict.fromkeys(self.selected_params + new_params))
        self._refresh_param_list()

    def _add_custom_param(self):
        val = self.param_entry.get().strip().upper()
        self.param_entry.delete(0, tk.END)
        if not val:
            return
        for p in val.replace(",", " ").split():
            p = p.strip()
            if p and p not in self.selected_params:
                self.selected_params.append(p)
        self._refresh_param_list()

    # ═══════════════════════════════════════
    #  图表生成
    # ═══════════════════════════════════════

    def _generate_chart(self):
        if not self.selected:
            messagebox.showwarning("提示", "请先选择网络")
            return

        name = self.selected[0]
        ntwk = self._get_network(name)
        if ntwk is None:
            # 未加载 → 触发加载，等轮询
            self.log(f"⏳ {name} 正在后台加载，请稍后再试...")
            return

        params = self.selected_params if self.selected_params else None
        chart_type = self.chart_type.get()

        try:
            if chart_type == "db":
                fig = sp.plot_s_db(ntwk, params=sp._parse_params(ntwk, params),
                                   title=f"{name} S-Parameters")
            elif chart_type == "deg":
                fig = sp.plot_s_deg(ntwk, params=sp._parse_params(ntwk, params),
                                    title=f"{name} Phase")
            elif chart_type == "smith":
                fig = sp.plot_s_smith(ntwk, params=sp._parse_params(ntwk, params),
                                      title=f"{name} Smith Chart")
            elif chart_type == "vswr":
                ports = [i for i in range(ntwk.nports)]
                fig = sp.plot_vswr(ntwk, ports, title=f"{name} VSWR")
            elif chart_type == "groupdelay":
                fig = sp.plot_group_delay(ntwk, params=sp._parse_params(ntwk, params),
                                          title=f"{name} Group Delay")
            else:
                fig = sp.plot_s_db(ntwk, params=sp._parse_params(ntwk, params),
                                   title=f"{name} S-Parameters")

            self._show_fig(fig, f"{name} Chart")
            self.log(f"📊 {name} 图表已生成")
        except Exception as e:
            self.log(f"❌ 生成失败: {e}")

    def _show_fig(self, fig, title="Chart"):
        """保存为 HTML 并用浏览器打开。"""
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        fig.write_html(tmp.name, include_plotlyjs="cdn")
        tmp.close()
        webbrowser.open(f"file://{tmp.name}")
        self.log(f"🌐 图表已在浏览器中打开: {os.path.basename(tmp.name)}")

    # ═══════════════════════════════════════
    #  多文件对比
    # ═══════════════════════════════════════

    def _compare(self):
        names = self.selected if len(self.selected) >= 2 else list(self.networks.keys())[:10]
        if len(names) < 2:
            messagebox.showwarning("提示", "至少需要 2 个网络进行对比（Ctrl+点击多选）")
            return

        # 确保全部已加载
        for n in names:
            ntwk = self._get_network(n)
            if ntwk is None:
                self.log(f"⏳ {n} 正在后台加载，请稍后再试...")
                return

        networks = [self.networks[n]["_ntwk"] for n in names]
        ref_name = self.ref_var.get() or names[0]
        ref_idx = names.index(ref_name) if ref_name in names else 0

        # 默认 S21
        param = (1, 0)
        if self.selected_params:
            p = self.selected_params[0]
            if p.startswith("S") and len(p) == 3:
                param = (int(p[1]) - 1, int(p[2]) - 1)

        try:
            interpolated = sp.interpolate_to_common_freq(
                networks, npoints=max(len(n.f) for n in networks))
            fig = sp.plot_multi_db(interpolated, names=names, param=param,
                                   title=f"{' vs '.join(names)} Comparison",
                                   show_diff=self.show_diff.get(),
                                   reference_idx=ref_idx)
            self._show_fig(fig, "Comparison")
            self.log(f"📊 对比图已生成: {' vs '.join(names)}")
        except Exception as e:
            self.log(f"❌ 对比失败: {e}")

    # ═══════════════════════════════════════
    #  链式级联
    # ═══════════════════════════════════════

    def _cascade_chain(self):
        if len(self.selected) < 2:
            messagebox.showwarning("提示", "请 Ctrl+点击选择 2+ 个网络（按级联顺序）")
            return

        for n in self.selected:
            ntwk = self._get_network(n)
            if ntwk is None:
                self.log(f"⏳ {n} 正在后台加载，请稍后再试...")
                return

        networks = [self.networks[n]["_ntwk"] for n in self.selected]
        try:
            result = sp.cascade_chain(networks)
            result_name = "_".join(self.selected)
            tmp = tempfile.NamedTemporaryFile(suffix=f".s{result.nports}p", delete=False)
            result.write_touchstone(tmp.name)
            tmp.close()

            self.networks[result_name] = {
                "path": tmp.name,
                "_ntwk": result,
                "_loading": False,
                "nports": result.nports,
                "f_min": float(result.f[0]),
                "f_max": float(result.f[-1]),
                "npoints": len(result.f),
                "params": [f"S{m+1}{n+1}" for m in range(result.nports)
                           for n in range(result.nports)],
            }
            self._refresh_net_list()
            self.log(f"🔗 链式级联完成: {' → '.join(self.selected)} → {result_name} "
                     f"({result.nports}端口)")
        except Exception as e:
            self.log(f"❌ 级联失败: {e}")

    # ═══════════════════════════════════════
    #  导出 CSV
    # ═══════════════════════════════════════

    def _export_csv(self):
        if not self.selected:
            return
        name = self.selected[0]
        ntwk = self._get_network(name)
        if ntwk is None:
            self.log(f"⏳ {name} 正在加载...")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"{name}.csv",
        )
        if not path:
            return
        try:
            sp.save_csv(ntwk, self.selected_params or None, path)
            self.log(f"💾 已导出: {path}")
        except Exception as e:
            self.log(f"❌ 导出失败: {e}")

    # ═══════════════════════════════════════
    #  LLM Agent
    # ═══════════════════════════════════════

    def _agent_chat(self):
        text = self.agent_entry.get().strip()
        self.agent_entry.delete(0, tk.END)
        if not text:
            return
        if not code_agent.is_available():
            self.log("⚠️ LLM 未配置。请创建 config.json 并设置 api_key。")
            return

        self.log(f"🤖 你: {text}")

        # 收集当前网络
        nets_dict = {}
        for name, entry in self.networks.items():
            # 确保已加载
            ntwk = entry.get("_ntwk")
            if ntwk is None:
                ntwk = self._get_network(name)
            nets_dict[name] = {"path": entry["path"], "nports": entry["nports"]}

        def _run():
            try:
                result = code_agent.generate_code(text, networks=nets_dict)
                if "error" in result:
                    self.root.after(0, lambda: self.log(f"❌ {result['error']}"))
                    return
                code = result.get("code", "")
                exec_r = result.get("exec_result", {})
                if exec_r.get("ok") and exec_r.get("figure_json"):
                    fig_data = exec_r["figure_json"]["data"]
                    fig_layout = exec_r["figure_json"]["layout"]
                    import plotly.graph_objects as go
                    fig = go.Figure(data=fig_data, layout=fig_layout)
                    self.root.after(0, lambda: self._show_fig(fig, text[:60]))
                    self.root.after(0, lambda: self.log(f"✅ {result.get('reply', '完成')}"))
                elif exec_r.get("error"):
                    self.root.after(0, lambda: self.log(f"❌ 执行错误: {exec_r['error'][:200]}"))
                else:
                    self.root.after(0, lambda: self.log(f"✅ {result.get('reply', '完成')}"))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"❌ Agent 异常: {e}"))

        threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════
#  文件头快速解析（同 app.py）
# ═══════════════════════════════════════════

def _parse_float(s: str) -> float:
    """解析浮点数，兼容 Fortran D 指数记法 (1.0D+09)。"""
    return float(s.replace("D", "E").replace("d", "e"))


def _read_header(path: str) -> dict:
    import re as _re
    # 数据行：以可选空格、可选正负号、数字开头
    data_re = _re.compile(r"^\s*[-+]?\d")

    freq_unit = "ghz"
    nports = 0
    first_data_line = None
    # 记录 # 行之后才算数（过滤注释里的数字）
    past_option_line = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            if stripped.startswith("#"):
                past_option_line = True
                parts = stripped.split()
                freq_str = parts[1].lower() if len(parts) > 1 else "ghz"
                if "ghz" in freq_str: freq_unit = "ghz"
                elif "mhz" in freq_str: freq_unit = "mhz"
                elif "khz" in freq_str: freq_unit = "khz"
                elif "hz" in freq_str: freq_unit = "hz"
                continue
            if past_option_line and data_re.match(stripped):
                first_data_line = stripped
                break

    if first_data_line is None:
        return None

    cols = first_data_line.split()
    nvals = len(cols) - 1
    if nvals > 0:
        for cols_per_param in [2, 1]:
            n2 = nvals // cols_per_param
            n = int(n2 ** 0.5)
            if n * n == n2 and n > 0:
                nports = n
                break

    if nports == 0:
        return None

    freq_mul = {"ghz": 1e9, "mhz": 1e6, "khz": 1e3, "hz": 1.0}.get(freq_unit, 1e9)
    try:
        f_min = _parse_float(cols[0]) * freq_mul
    except Exception:
        f_min = 0.0

    npoints = 0
    last_freq = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if data_re.match(line):
                npoints += 1
                try:
                    last_freq = _parse_float(line.split()[0])
                except Exception:
                    pass

    f_max = (last_freq * freq_mul) if last_freq is not None else f_min

    return {
        "nports": nports,
        "f_min": float(f_min),
        "f_max": float(f_max),
        "npoints": npoints,
        "freq_unit": freq_unit,
    }


# ═══════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = SParamGUI(root)
    root.mainloop()
