#!/usr/bin/env python3
"""
S-Parameter Agent — 原生 GUI 版本 (matplotlib 引擎)
基于 tkinter + matplotlib，图表直接嵌入界面。
本地直接读取 .sNp 文件，绕过 HTTP 上传限制。
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skrf as rf
import s_params as sp
import nl_parser
import code_agent

# ── matplotlib ──
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# matplotlib 全局字体（monospace 在所有平台可用，无方块问题）
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['Consolas', 'Courier New', 'DejaVu Sans Mono', 'monospace']
plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════
#  matplotlib 画图函数
# ═══════════════════════════════════════════

_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def _mpl_plot_db(ax, ntwk, params=None):
    """matplotlib: dB 幅度 vs 频率"""
    if params is None:
        params = [(m, n) for m in range(ntwk.nports) for n in range(ntwk.nports)]
    freq_ghz = ntwk.f / 1e9
    for i, (m, n) in enumerate(params):
        ax.plot(freq_ghz, ntwk.s_db[:, m, n],
                color=_COLORS[i % len(_COLORS)], linewidth=1.2,
                label=f"S{m+1}{n+1} ({ntwk.name})")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which='major', color='#ccc', linewidth=0.5)
    ax.grid(True, which='minor', color='#eee', linewidth=0.3)
    ax.legend(loc='upper right', fontsize=7, ncol=2)


def _mpl_plot_deg(ax, ntwk, params=None):
    """matplotlib: 相位 vs 频率"""
    if params is None:
        params = [(m, n) for m in range(ntwk.nports) for n in range(ntwk.nports)]
    freq_ghz = ntwk.f / 1e9
    for i, (m, n) in enumerate(params):
        ax.plot(freq_ghz, ntwk.s_deg[:, m, n],
                color=_COLORS[i % len(_COLORS)], linewidth=1.2,
                label=f"S{m+1}{n+1} ({ntwk.name})")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Phase (deg)")
    ax.grid(True, which='major', color='#ccc', linewidth=0.5)
    ax.grid(True, which='minor', color='#eee', linewidth=0.3)
    ax.legend(loc='upper right', fontsize=7, ncol=2)


def _mpl_plot_smith(ax, ntwk, params=None):
    """matplotlib: Smith 圆图"""
    # 画 Smith 参考圆
    theta = np.linspace(0, 2 * np.pi, 360)
    for r in np.linspace(0.2, 1.0, 5):
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color='lightgray', linewidth=0.5)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', linewidth=0.8)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    if params is None:
        params = [(m, n) for m in range(ntwk.nports) for n in range(ntwk.nports)]
    for i, (m, n) in enumerate(params):
        gamma = ntwk.s[:, m, n]
        ax.plot(np.real(gamma), np.imag(gamma),
                color=_COLORS[i % len(_COLORS)], linewidth=1.2,
                label=f"S{m+1}{n+1} ({ntwk.name})")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel("Real (Γ)")
    ax.set_ylabel("Imag (Γ)")
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=7, ncol=2)


def _mpl_plot_vswr(ax, ntwk, ports=None):
    """matplotlib: VSWR vs 频率"""
    if ports is None:
        ports = list(range(ntwk.nports))
    freq_ghz = ntwk.f / 1e9
    for i, p in enumerate(ports):
        ax.plot(freq_ghz, ntwk.s_vswr[:, p, p],
                color=_COLORS[i % len(_COLORS)], linewidth=1.2,
                label=f"VSWR{p+1} ({ntwk.name})")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("VSWR")
    ax.grid(True, which='major', color='#ccc', linewidth=0.5)
    ax.grid(True, which='minor', color='#eee', linewidth=0.3)
    ax.legend(loc='upper right', fontsize=7, ncol=2)


def _mpl_plot_groupdelay(ax, ntwk, params=None):
    """matplotlib: 群时延 vs 频率"""
    if params is None:
        params = [(m, n) for m in range(ntwk.nports) for n in range(ntwk.nports)]
    freq_ghz = ntwk.f / 1e9
    for i, (m, n) in enumerate(params):
        phase_rad = np.unwrap(np.angle(ntwk.s[:, m, n]))
        dphi_df = np.gradient(phase_rad, ntwk.f)
        gd_ns = -dphi_df / (2 * np.pi) * 1e9
        ax.plot(freq_ghz, gd_ns,
                color=_COLORS[i % len(_COLORS)], linewidth=1.2,
                label=f"GD S{m+1}{n+1} ({ntwk.name})")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Group Delay (ns)")
    ax.grid(True, which='major', color='#ccc', linewidth=0.5)
    ax.grid(True, which='minor', color='#eee', linewidth=0.3)
    ax.legend(loc='upper right', fontsize=7, ncol=2)


def _mpl_compare(ax, networks, names, param=(1, 0), show_diff=True, ref_idx=0):
    """matplotlib: 多文件 dB 对比 + 差异曲线"""
    freq_ghz = networks[0].f / 1e9
    for i, ntwk in enumerate(networks):
        ax.plot(freq_ghz, ntwk.s_db[:, param[0], param[1]],
                color=_COLORS[i % len(_COLORS)], linewidth=1.2, label=names[i])

    if show_diff and len(networks) >= 2:
        ax2 = ax.twinx()
        ref = networks[ref_idx]
        ref_db = ref.s_db[:, param[0], param[1]]
        for i, ntwk in enumerate(networks):
            if i == ref_idx:
                continue
            diff = ntwk.s_db[:, param[0], param[1]] - ref_db
            ax2.plot(freq_ghz, diff, color=_COLORS[i % len(_COLORS)],
                     linewidth=0.8, linestyle='--', alpha=0.7,
                     label=f"Δ ({names[i]}−{names[ref_idx]})")
        ax2.set_ylabel("Δ (dB)", fontsize=8)
        ax2.legend(loc='lower right', fontsize=6)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (GHz)")
    pstr = f"S{param[0]+1}{param[1]+1}"
    ax.set_ylabel(f"{pstr} (dB)")
    ax.grid(True, which='major', color='#ccc', linewidth=0.5)
    ax.grid(True, which='minor', color='#eee', linewidth=0.3)
    ax.legend(loc='upper right', fontsize=7, ncol=2)


# ═══════════════════════════════════════════
#  GUI App
# ═══════════════════════════════════════════

class SParamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("S-Parameter Agent")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 650)
        self.root.configure(bg='#f0f0f0')

        self.networks = {}
        self.selected = []
        self.selected_params = []
        self.show_diff = tk.BooleanVar(value=True)
        self.current_fig = None
        self.chart_canvas = None
        self.toolbar = None
        self._loading_threads = []  # 追踪后台加载线程
        self._cancel_event = threading.Event()

        self._build_ui()
        # 窗口关闭时取消所有后台线程
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ════════════════════════════
    #  UI 布局
    # ════════════════════════════

    def _build_ui(self):
        # ── 顶部工具栏 ──
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X, padx=4, pady=(4, 0))

        ttk.Button(toolbar_frame, text="📂 加载文件", command=self._load_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="📁 加载目录", command=self._load_directory).pack(side=tk.LEFT, padx=2)
        self.path_entry = ttk.Entry(toolbar_frame, width=30)
        self.path_entry.pack(side=tk.LEFT, padx=4)
        self.path_entry.bind("<Return>", lambda e: self._load_by_path())
        ttk.Button(toolbar_frame, text="路径加载", command=self._load_by_path).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(toolbar_frame, text="图表:").pack(side=tk.LEFT, padx=2)
        self.chart_type = tk.StringVar(value="db")
        for text, val in [("dB", "db"), ("相位", "deg"), ("Smith", "smith"),
                          ("VSWR", "vswr"), ("群时延", "gd")]:
            ttk.Radiobutton(toolbar_frame, text=text, variable=self.chart_type,
                            value=val).pack(side=tk.LEFT, padx=1)

        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(toolbar_frame, text="🔄 绘图", command=self._generate_chart).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="📊 对比", command=self._compare).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="🔗 级联", command=self._cascade_chain).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="💾 CSV", command=self._export_csv).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(toolbar_frame, text="Agent:").pack(side=tk.LEFT, padx=2)
        self.agent_entry = ttk.Entry(toolbar_frame, width=25)
        self.agent_entry.pack(side=tk.LEFT, padx=2)
        self.agent_entry.bind("<Return>", lambda e: self._agent_chat())
        ttk.Button(toolbar_frame, text="发送", command=self._agent_chat).pack(side=tk.LEFT, padx=2)

        # ── 主区域：左侧网络列表 + 右侧图表 ──
        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 左侧面板
        left = ttk.Frame(main_pw, width=320)
        right = ttk.Frame(main_pw)
        main_pw.add(left, weight=0)
        main_pw.add(right, weight=1)

        self._build_left(left)
        self._build_right(right)

        # ── 底部状态栏 ──
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=4, pady=(0, 2))

        self.status_var = tk.StringVar(value="就绪 — 点击「加载文件」选择 .sNp 文件")
        ttk.Label(status_frame, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W, padding=2).pack(fill=tk.X, side=tk.LEFT)

        self.log_text = tk.Text(status_frame, height=3, wrap=tk.WORD,
                                font=("Consolas", 9), bg='#1e1e1e', fg='#c0c0c0',
                                state=tk.DISABLED)
        self.log_text.pack(fill=tk.X, side=tk.LEFT, expand=True)

    def _build_left(self, parent):
        f = ttk.Frame(parent)
        f.pack(fill=tk.BOTH, expand=True)

        # 网络列表
        grp = ttk.LabelFrame(f, text="📋 已加载网络", padding=4)
        grp.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.net_list = tk.Listbox(grp, selectmode=tk.EXTENDED, height=10,
                                   exportselection=False)
        self.net_list.pack(fill=tk.BOTH, expand=True)
        self.net_list.bind("<<ListboxSelect>>", self._on_net_select)
        row = ttk.Frame(grp)
        row.pack(fill=tk.X, pady=1)
        ttk.Button(row, text="移除", command=self._remove_selected, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(row, text="清空", command=self._clear_all, width=6).pack(side=tk.LEFT, padx=1)

        # 参数选择
        grp = ttk.LabelFrame(f, text="🎯 S 参数", padding=4)
        grp.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.param_label = ttk.Label(grp, text="请先选择网络")
        self.param_label.pack(anchor=tk.W)
        self.param_list = tk.Listbox(grp, selectmode=tk.EXTENDED, height=5,
                                     exportselection=False)
        self.param_list.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(grp)
        row.pack(fill=tk.X, pady=1)
        for txt, kind in [("全选", "all"), ("反射", "refl"), ("传输", "trans")]:
            ttk.Button(row, text=txt, width=5,
                       command=lambda k=kind: self._param_quick(k)).pack(side=tk.LEFT, padx=1)
        row2 = ttk.Frame(grp)
        row2.pack(fill=tk.X)
        self.param_entry = ttk.Entry(row2)
        self.param_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.param_entry.bind("<Return>", lambda e: self._add_custom_param())
        ttk.Button(row2, text="+", width=3, command=self._add_custom_param).pack(side=tk.RIGHT)

        # 对比选项
        grp = ttk.LabelFrame(f, text="📊 对比选项", padding=4)
        grp.pack(fill=tk.X, padx=2, pady=2)
        ttk.Checkbutton(grp, text="显示差异 (Δ)", variable=self.show_diff).pack(anchor=tk.W)
        row = ttk.Frame(grp)
        row.pack(fill=tk.X)
        ttk.Label(row, text="参考:").pack(side=tk.LEFT)
        self.ref_var = tk.StringVar()
        self.ref_combo = ttk.Combobox(row, textvariable=self.ref_var, width=12)
        self.ref_combo.pack(side=tk.LEFT, padx=2)

    def _build_right(self, parent):
        # 图表区域
        self.chart_frame = ttk.Frame(parent)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.chart_canvas, self.chart_frame)
        self.toolbar.update()

        self.ax.text(0.5, 0.5, "Load .sNp file and select parameters\nClick [Plot] to generate chart",
                     transform=self.ax.transAxes, ha='center', va='center',
                     fontsize=12, color='#aaa', family='monospace')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.chart_canvas.draw()

    def _on_close(self):
        """窗口关闭：取消后台线程并退出。"""
        self._cancel_event.set()
        self.root.destroy()

    # ════════════════════════════════════
    #  日志
    # ════════════════════════════════════

    def log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        # 只保留最后 50 行
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 50:
            self.log_text.delete('1.0', f'{lines - 50}.0')

    def status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ════════════════════════════════════
    #  文件加载
    # ════════════════════════════════════

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
        name = os.path.splitext(os.path.basename(path))[0]
        self.status(f"⏳ 正在解析 {os.path.basename(path)}...")
        # 后台线程加载，避免 UI 卡顿
        threading.Thread(target=self._load_worker, args=(path, name), daemon=True).start()

    def _load_worker(self, path, name):
        try:
            ntwk = rf.Network(path)
        except Exception as e:
            if not self._cancel_event.is_set():
                self.root.after(0, lambda: self.log(f"❌ 无法解析: {os.path.basename(path)} — {e}"))
                self.root.after(0, lambda: self.status("就绪"))
            return

        if self._cancel_event.is_set():
            return

        if name in self.networks:
            base, idx = name, 2
            while f"{base}_{idx}" in self.networks:
                idx += 1
            name = f"{base}_{idx}"

        nports = ntwk.nports
        all_params = [f"S{m+1}{n+1}" for m in range(nports) for n in range(nports)]

        entry = {
            "path": path, "_ntwk": ntwk,
            "nports": nports,
            "f_min": float(ntwk.f[0]), "f_max": float(ntwk.f[-1]),
            "npoints": len(ntwk.f), "params": all_params,
        }
        # 切回主线程更新 UI
        def _done():
            if self._cancel_event.is_set():
                return
            self.networks[name] = entry
            self.log(f"📂 {name}  {nports}端口  {len(ntwk.f)}点  "
                     f"{ntwk.f[0]/1e9:.4f}–{ntwk.f[-1]/1e9:.4f} GHz")
            self._refresh_net_list()
            self.status("就绪")
        self.root.after(0, _done)

    def _get_network(self, name):
        entry = self.networks.get(name)
        return entry["_ntwk"] if entry else None

    # ════════════════════════════════════
    #  网络列表
    # ════════════════════════════════════

    def _refresh_net_list(self):
        self.net_list.delete(0, tk.END)
        for name, entry in self.networks.items():
            self.net_list.insert(tk.END,
                f"{name}  [{entry['nports']}端口, {entry['npoints']}点]")
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

    # ════════════════════════════════════
    #  参数选择
    # ════════════════════════════════════

    def _refresh_param_list(self):
        self.param_list.delete(0, tk.END)
        if not self.selected:
            self.param_label.config(text="请先选择网络")
            return
        name = self.selected[0]
        entry = self.networks.get(name)
        if not entry:
            return
        nports = entry['nports']
        self.param_label.config(text=f"{name}  {nports}端口 · {nports**2}参数")
        # 大端口截断显示，通过搜索定位
        params = entry["params"]
        max_show = 500
        show_params = params if len(params) <= max_show else params[:max_show]
        for p in show_params:
            self.param_list.insert(tk.END, p)
        if len(params) > max_show:
            self.param_list.insert(tk.END, f"... 还有 {len(params) - max_show} 个参数，用搜索框定位")
        for i, p in enumerate(show_params):
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
        existing = set(self.selected_params)
        if kind == "all":
            existing.update(f"S{m}{n}" for m in range(1, nports + 1) for n in range(1, nports + 1))
        elif kind == "refl":
            existing.update(f"S{m}{m}" for m in range(1, nports + 1))
        elif kind == "trans":
            existing.update(f"S{m}{n}" for m in range(1, nports + 1) for n in range(1, nports + 1) if m != n)
        self.selected_params = list(existing)
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

    # ════════════════════════════════════
    #  图表生成
    # ════════════════════════════════════

    def _show_image(self, pil_image, title="Agent Chart"):
        """在图表区显示 PIL Image（Agent 生成结果）。"""
        from PIL import ImageTk
        cw = self.chart_frame.winfo_width()
        ch = self.chart_frame.winfo_height()
        if cw < 10: cw = 800
        if ch < 10: ch = 500
        iw, ih = pil_image.size
        scale = min(cw / iw, ch / ih, 1.0)
        if scale < 1.0:
            pil_image = pil_image.resize((int(iw * scale), int(ih * scale)))
        self._current_img_tk = ImageTk.PhotoImage(pil_image)
        self.chart_canvas.delete("all")
        self.chart_canvas.create_image(cw // 2, ch // 2, image=self._current_img_tk, anchor=tk.CENTER)
        self.status(title)

    def _clear_chart(self):
        # 清除数据但保留轴框架
        for artist in list(self.ax.lines) + list(self.ax.collections) + list(self.ax.texts):
            artist.remove()
        if self.ax.get_legend():
            self.ax.get_legend().remove()
        self.ax.relim()
        self.ax.autoscale()

    def _draw_chart(self, title=""):
        if title:
            self.ax.set_title(title, fontsize=11, fontweight='bold', family='monospace')
        self.fig.subplots_adjust(left=0.10, right=0.95, top=0.93, bottom=0.12)
        self.chart_canvas.draw()

    def _generate_chart(self):
        if not self.selected:
            messagebox.showwarning("提示", "请先选择网络")
            return

        name = self.selected[0]
        ntwk = self._get_network(name)
        if ntwk is None:
            self.log(f"⏳ {name} 未加载")
            return

        self.status(f"⏳ 正在画 {name}...")
        self.root.update_idletasks()

        params = self.selected_params if self.selected_params else None
        chart_type = self.chart_type.get()

        self._clear_chart()
        try:
            if chart_type == "db":
                _mpl_plot_db(self.ax, ntwk,
                             sp._parse_params(ntwk, params) if params else None)
                title = f"{name} — S-Parameter Magnitude"
            elif chart_type == "deg":
                _mpl_plot_deg(self.ax, ntwk,
                              sp._parse_params(ntwk, params) if params else None)
                title = f"{name} — Phase"
            elif chart_type == "smith":
                _mpl_plot_smith(self.ax, ntwk,
                                sp._parse_params(ntwk, params) if params else None)
                title = f"{name} — Smith Chart"
            elif chart_type == "vswr":
                ports = list(range(ntwk.nports))
                _mpl_plot_vswr(self.ax, ntwk, ports)
                title = f"{name} — VSWR"
            elif chart_type == "gd":
                _mpl_plot_groupdelay(self.ax, ntwk,
                                     sp._parse_params(ntwk, params) if params else None)
                title = f"{name} — Group Delay"
            else:
                _mpl_plot_db(self.ax, ntwk)
                title = f"{name} — S-Parameters"

            self._draw_chart(title)
            self.log(f"📊 {name} 图表已生成")
        except Exception as e:
            self.log(f"❌ 生成失败: {e}")
        finally:
            self.status("就绪")

    # ════════════════════════════════════
    #  多文件对比
    # ════════════════════════════════════

    def _compare(self):
        names = self.selected if len(self.selected) >= 2 else list(self.networks.keys())[:10]
        if len(names) < 2:
            messagebox.showwarning("提示", "至少需要 2 个网络")
            return

        networks = [self.networks[n]["_ntwk"] for n in names
                    if n in self.networks and self.networks[n].get("_ntwk")]
        if len(networks) < 2:
            self.log("需要至少 2 个已加载的网络")
            return

        # 插值到共同频率
        try:
            interpolated = sp.interpolate_to_common_freq(
                networks, npoints=max(len(n.f) for n in networks))
        except Exception:
            interpolated = networks

        ref_name = self.ref_var.get() or names[0]
        ref_idx = names.index(ref_name) if ref_name in names else 0

        param = (1, 0)
        if self.selected_params:
            p = self.selected_params[0]
            if p.startswith("S") and len(p) == 3:
                param = (int(p[1]) - 1, int(p[2]) - 1)

        self._clear_chart()
        _mpl_compare(self.ax, interpolated, names, param,
                     show_diff=self.show_diff.get(), ref_idx=ref_idx)
        self._draw_chart(f"{' vs '.join(names)} — S{param[0]+1}{param[1]+1}")
        self.log(f"📊 对比: {' vs '.join(names)}")

    # ════════════════════════════════════
    #  链式级联
    # ════════════════════════════════════

    def _cascade_chain(self):
        if len(self.selected) < 2:
            messagebox.showwarning("提示", "请 Ctrl+点击选择 2+ 个网络")
            return

        networks = [self.networks[n]["_ntwk"] for n in self.selected
                    if n in self.networks]
        if len(networks) < 2:
            self.log("需要至少 2 个已加载的网络")
            return

        try:
            result = sp.cascade_chain(networks)
            result_name = "_".join(self.selected)
            tmp = tempfile.NamedTemporaryFile(suffix=f".s{result.nports}p", delete=False)
            result.write_touchstone(tmp.name)
            tmp.close()

            self.networks[result_name] = {
                "path": tmp.name, "_ntwk": result,
                "nports": result.nports,
                "f_min": float(result.f[0]), "f_max": float(result.f[-1]),
                "npoints": len(result.f),
                "params": [f"S{m+1}{n+1}" for m in range(result.nports)
                           for n in range(result.nports)],
            }
            self._refresh_net_list()
            self.log(f"🔗 级联: {' → '.join(self.selected)} → {result_name} ({result.nports}端口)")
        except Exception as e:
            self.log(f"❌ 级联失败: {e}")

    # ════════════════════════════════════
    #  导出
    # ════════════════════════════════════

    def _export_csv(self):
        if not self.selected:
            return
        name = self.selected[0]
        ntwk = self._get_network(name)
        if ntwk is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile=f"{name}.csv")
        if not path:
            return
        try:
            sp.save_csv(ntwk, self.selected_params or None, path)
            self.log(f"💾 已导出: {path}")
        except Exception as e:
            self.log(f"❌ 导出失败: {e}")

    # ════════════════════════════════════
    #  LLM Agent
    # ════════════════════════════════════

    def _agent_chat(self):
        text = self.agent_entry.get().strip()
        self.agent_entry.delete(0, tk.END)
        if not text:
            return
        if not code_agent.is_available():
            self.log("⚠️ LLM 未配置。创建 config.json 并设置 api_key")
            return

        self.log(f"🤖 你: {text}")
        nets_dict = {n: {"path": e["path"], "nports": e["nports"]}
                     for n, e in self.networks.items()}

        def _run():
            try:
                result = code_agent.generate_code(text, networks=nets_dict)
                if "error" in result:
                    self.root.after(0, lambda: self.log(f"❌ {result['error']}"))
                    return
                exec_r = result.get("exec_result", {})
                if exec_r.get("ok") and exec_r.get("figure_png_b64"):
                    # Agent 返回 matplotlib PNG → 嵌入界面
                    import base64, io as _io2
                    from PIL import Image, ImageTk
                    img_data = base64.b64decode(exec_r["figure_png_b64"])
                    img = Image.open(_io2.BytesIO(img_data))
                    self.root.after(0, lambda: self._show_image(img, text[:60]))
                    self.root.after(0, lambda: self.log("✅ Agent 图表已生成"))
                self.root.after(0, lambda: self.log(
                    f"✅ {result.get('reply', '完成')}"))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"❌ Agent 异常: {e}"))

        threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = SParamGUI(root)
    root.mainloop()
