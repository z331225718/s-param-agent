#!/usr/bin/env python3
"""
S-Parameter 核心工具库
基于 scikit-rf + plotly，提供读/写/处理/交互式画图/导出的全部能力。
所有画图函数返回 plotly Figure 或自包含 HTML 文件。
"""

import io
import os
import base64
from typing import Optional, Union, List, Tuple

import numpy as np
import skrf as rf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── 1. 文件 I/O ───────────────────────────────────────────────────────

def load_ntwk(path: str) -> rf.Network:
    """
    加载 Touchstone 文件 (.s1p / .s2p / .sNp)。
    自动识别端口数和格式。
    """
    ntwk = rf.Network(path)
    ntwk.name = os.path.splitext(os.path.basename(path))[0]
    return ntwk


def load_ntwk_from_arrays(
    freq: np.ndarray,
    s_data: np.ndarray,
    z0: float = 50.0,
    name: str = "Network",
) -> rf.Network:
    """
    从 numpy 数组构造 Network 对象。

    Args:
        freq: 频率数组 (Hz)
        s_data: S 参数复数矩阵，shape = (nfreqs, nports, nports)
        z0: 参考阻抗
        name: 网络名称
    """
    freq_rf = rf.Frequency.from_f(freq, unit="hz")
    ntwk = rf.Network(frequency=freq_rf, s=s_data, z0=z0, name=name)
    return ntwk


def save_touchstone(ntwk: rf.Network, path: str, fmt: str = "db") -> str:
    """
    将 Network 写回 Touchstone 文件。

    Args:
        ntwk: 网络对象
        path: 输出路径
        fmt: 'db', 'ma' (magnitude/angle), 'ri' (real/imaginary)
    """
    ntwk.write_touchstone(path, form=fmt)
    return path


def save_csv(
    ntwk: rf.Network,
    params=None,
    path: str = "output.csv",
    include_deg: bool = True,
) -> str:
    """
    导出指定 S 参数为 CSV。

    Args:
        ntwk: 网络对象
        params: 参数列表，支持 "S11" 字符串、(0,0) 元组 或 ["S11","S21"]。默认导出全部 S 参数
        path: CSV 输出路径
        include_deg: 是否包含相位列
    """
    params = _parse_params(ntwk, params)
    freq_ghz = ntwk.f / 1e9
    columns = ["Freq_GHz"]
    data = [freq_ghz]

    for m, n in params:
        s_mn = ntwk.s[:, m, n]
        columns.append(f"S{m+1}{n+1}_dB")
        data.append(ntwk.s_db[:, m, n])
        if include_deg:
            columns.append(f"S{m+1}{n+1}_deg")
            data.append(ntwk.s_deg[:, m, n])

    rows = np.column_stack(data)
    header = ",".join(columns)
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    return path


# 别名，兼容旧调用
export_csv = save_csv


# ─── 2. 信息查看 ───────────────────────────────────────────────────────

def info(ntwk: rf.Network) -> str:
    """返回网络的多行文本概览。"""
    lines = [
        f"名称:     {ntwk.name}",
        f"端口数:   {ntwk.nports}",
        f"频率范围: {ntwk.f[0]/1e9:.4f} – {ntwk.f[-1]/1e9:.4f} GHz",
        f"频率点数: {len(ntwk.f)}",
        f"参考阻抗: {ntwk.z0[0, 0]:.1f} Ω",
    ]
    params = []
    for m in range(ntwk.nports):
        for n in range(ntwk.nports):
            params.append(f"S{m+1}{n+1}")
    lines.append(f"参数:     {', '.join(params)}")
    return "\n".join(lines)


def summary(ntwk: rf.Network) -> dict:
    """返回网络的字典概览，便于程序化使用。"""
    return {
        "name": ntwk.name,
        "nports": ntwk.nports,
        "f_min_ghz": ntwk.f[0] / 1e9,
        "f_max_ghz": ntwk.f[-1] / 1e9,
        "npoints": len(ntwk.f),
        "z0": ntwk.z0[0, 0],
    }


def list_params(ntwk: rf.Network) -> List[str]:
    """返回所有 S 参数的名称列表，如 ['S11', 'S12', 'S21', 'S22']。"""
    return [f"S{m+1}{n+1}" for m in range(ntwk.nports) for n in range(ntwk.nports)]


# ─── 3. 数据提取 ───────────────────────────────────────────────────────

def get_s(ntwk: rf.Network, m: int, n: int) -> np.ndarray:
    """提取 S_{m+1}{n+1} 复数数组。m,n 从 0 开始。"""
    return ntwk.s[:, m, n]


def get_s_db(ntwk: rf.Network, m: int, n: int) -> np.ndarray:
    """提取 S_{m+1}{n+1} dB 值。"""
    return ntwk.s_db[:, m, n]


def get_s_deg(ntwk: rf.Network, m: int, n: int) -> np.ndarray:
    """提取 S_{m+1}{n+1} 相位 (度)。"""
    return ntwk.s_deg[:, m, n]


def get_vswr(ntwk: rf.Network, port: int) -> np.ndarray:
    """提取指定端口的 VSWR。port 从 0 开始。"""
    return ntwk.s_vswr[:, port, port]


def get_z(ntwk: rf.Network, port: int) -> np.ndarray:
    """提取指定端口的阻抗 (复数)。port 从 0 开始。"""
    return ntwk.z[:, port, port]


def get_group_delay(ntwk: rf.Network, m: int, n: int) -> np.ndarray:
    """
    计算群时延 (ns)。
    对 S_{m+1}{n+1} 的相位求数值微分。
    """
    phase_rad = np.unwrap(np.angle(ntwk.s[:, m, n]))
    freq = ntwk.f
    # 中心差分
    dphi_df = np.gradient(phase_rad, freq)
    gd = -dphi_df / (2 * np.pi)  # 秒
    return gd * 1e9  # ns


# ─── 4. 处理 / 变换 ────────────────────────────────────────────────────

def cascade(ntwk_a: rf.Network, ntwk_b: rf.Network) -> rf.Network:
    """级联两个网络 (ntwk_a → ntwk_b)。要求频率对齐。"""
    return ntwk_a ** ntwk_b


def deembed(ntwk_dut: rf.Network, fixture: rf.Network) -> rf.Network:
    """
    去嵌：从 DUT+夹具的测量结果中移除夹具效应。
    要求 fixture 是可逆的双端口网络。
    """
    # 典型用法：ntwk_measured = fixture_input ** ntwk_dut ** fixture_output
    # 这里提供简单形式：去除串联的 fixture
    fixture_inv = fixture.inv
    deembedded = fixture_inv ** ntwk_dut
    return deembedded


def slice_freq(
    ntwk: rf.Network,
    start: Union[float, str],
    stop: Optional[Union[float, str]] = None,
) -> rf.Network:
    """
    截取频率子集。

    支持三种调用方式：
        slice_freq(ntwk, '2-4ghz')            # 范围字符串
        slice_freq(ntwk, 2e9, 4e9)            # 两个数值 (Hz)
        slice_freq(ntwk, '2GHz', '4GHz')      # 两个字符串
    """
    if stop is None and isinstance(start, str) and "-" in start:
        # 范围字符串解析：'2-4ghz', '1GHz-6GHz'
        parts = start.split("-", 1)
        start = _parse_freq_str(parts[0])
        stop = _parse_freq_str(parts[1])
    else:
        if isinstance(start, str):
            start = _parse_freq_str(start)
        if isinstance(stop, str):
            stop = _parse_freq_str(stop)
    return ntwk[f"{start}-{stop}"]


def interpolate_to(ntwk: rf.Network, freqs: np.ndarray) -> rf.Network:
    """重采样到指定频率点。"""
    freq_obj = rf.Frequency.from_f(freqs, unit="hz")
    return ntwk.interpolate(freq_obj)


def renormalize(ntwk: rf.Network, z0: float) -> rf.Network:
    """重归一化到不同参考阻抗。"""
    return ntwk.renormalize(z0)


def cascade_chain(networks: List[rf.Network]) -> rf.Network:
    """
    链式级联多个网络 (A → B → C → ...)。
    自动验证端口匹配：前一网络输出端口数必须等于后一网络输入端口数。

    Args:
        networks: 按顺序排列的网络列表 [LNA, BPF, AMP]
    Returns:
        级联后的复合网络
    Raises:
        ValueError: 端口不匹配时抛出
    """
    if len(networks) < 2:
        raise ValueError(f"级联需要至少 2 个网络，提供了 {len(networks)} 个")

    result = networks[0]
    for i, nxt in enumerate(networks[1:], 1):
        # scikit-rf ** 操作符要求：前一网络的输出端口数 == 后一网络的输入端口数
        # 对于二端口级联：ntwk_a (port1=out) ** ntwk_b (port0=in)
        try:
            result = result ** nxt
        except Exception as e:
            raise ValueError(
                f"级联失败: {networks[i-1].name} (端口{networks[i-1].nports}) → "
                f"{nxt.name} (端口{nxt.nports}): {e}"
            )
    result.name = " → ".join(n.name for n in networks)
    return result


def load_networks_batch(paths: List[str]) -> List[rf.Network]:
    """
    批量加载多个 Touchstone 文件。

    Args:
        paths: 文件路径列表
    Returns:
        成功加载的 Network 对象列表（跳过失败的文件并打印警告）
    """
    networks = []
    for p in paths:
        try:
            ntwk = load_ntwk(p)
            networks.append(ntwk)
        except Exception as e:
            import warnings
            warnings.warn(f"跳过无法加载的文件 {p}: {e}")
    return networks


def find_common_freq_range(networks: List[rf.Network]) -> Tuple[float, float]:
    """
    找到多个网络频率范围的交集。

    Returns:
        (f_min, f_max) 共有的频率范围 (Hz)
    """
    f_min = max(n.f[0] for n in networks)
    f_max = min(n.f[-1] for n in networks)
    if f_min >= f_max:
        raise ValueError(
            f"网络之间没有共同的频率范围。"
            f"范围: {[(n.f[0]/1e9, n.f[-1]/1e9) for n in networks]}"
        )
    return (f_min, f_max)


def interpolate_to_common_freq(
    networks: List[rf.Network],
    npoints: int = None,
    freq_array: np.ndarray = None,
) -> List[rf.Network]:
    """
    将多个网络插值到共同的频率网格，便于逐点对比。

    Args:
        networks: 网络列表
        npoints: 目标频率点数（使用共同范围的等间距网格）
        freq_array: 自定义频率数组 (Hz)，优先级高于 npoints
    Returns:
        插值后的网络列表（新对象，不修改原网络）
    """
    if freq_array is not None:
        target_freq = freq_array
    elif npoints is not None:
        f_min, f_max = find_common_freq_range(networks)
        target_freq = np.linspace(f_min, f_max, npoints)
    else:
        # 使用第一个网络的频率作为参考
        target_freq = networks[0].f

    freq_obj = rf.Frequency.from_f(target_freq, unit="hz")
    return [n.interpolate(freq_obj) for n in networks]


def compute_diff_stats(
    networks: List[rf.Network],
    param: Tuple[int, int] = (1, 0),
    reference_idx: int = 0,
) -> dict:
    """
    计算多个网络之间同一 S 参数的差异统计。

    Args:
        networks: 已插值到同一频率网格的网络列表
        param: S 参数索引 (m, n)
        reference_idx: 作为参考基准的网络索引（默认第一个）
    Returns:
        {
            "param": "S21",
            "reference": "LNA",
            "comparisons": [
                {"name": "BPF", "mean_diff_db": 1.2, "max_diff_db": 3.5,
                 "min_diff_db": 0.1, "rms_diff_db": 1.8}
            ],
            "common_freq_ghz": [f1, f2, ...],
            "diff_traces": [(name, diff_db_array), ...]
        }
    """
    m, n = param
    param_str = f"S{m+1}{n+1}"
    ref = networks[reference_idx]
    ref_db = ref.s_db[:, m, n]
    freq_ghz = ref.f / _FREQ_UNIT

    comparisons = []
    diff_traces = []
    for i, ntwk in enumerate(networks):
        if i == reference_idx:
            continue
        db = ntwk.s_db[:, m, n]
        diff = db - ref_db
        comparisons.append({
            "name": ntwk.name,
            "mean_diff_db": float(np.mean(diff)),
            "max_diff_db": float(np.max(np.abs(diff))),
            "min_diff_db": float(np.min(diff)),
            "rms_diff_db": float(np.sqrt(np.mean(diff ** 2))),
        })
        diff_traces.append((ntwk.name, diff))

    return {
        "param": param_str,
        "reference": ref.name,
        "comparisons": comparisons,
        "common_freq_ghz": freq_ghz.tolist(),
        "diff_traces": diff_traces,
    }


def _parse_freq_str(s: str) -> float:
    """解析 '2GHz', '2.4g', '500mhz', '500M' 为 Hz 数值。"""
    s = s.strip().lower().replace(" ", "")
    if s.endswith("ghz") or s.endswith("g"):
        return float(s.rstrip("ghz").rstrip("g")) * 1e9
    elif s.endswith("mhz") or s.endswith("m"):
        return float(s.rstrip("mhz").rstrip("m")) * 1e6
    elif s.endswith("khz") or s.endswith("k"):
        return float(s.rstrip("khz").rstrip("k")) * 1e3
    elif s.endswith("hz"):
        return float(s.rstrip("hz"))
    else:
        # 默认按 GHz 处理纯数字
        return float(s) * 1e9


# ─── 5. 交互式画图 (Plotly) ────────────────────────────────────────────

_DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

_FREQ_UNIT = 1e9  # 默认 X 轴用 GHz


def _param_label(m: int, n: int, ntwk_name: str = "") -> str:
    """生成图例标签，如 'S21 (filter)'"""
    s = f"S{m+1}{n+1}"
    if ntwk_name:
        s += f" ({ntwk_name})"
    return s


def _parse_params(ntwk: rf.Network, params) -> List[Tuple[int, int]]:
    """
    标准化 params 参数。支持：
        - "S11" / "S21" 字符串
        - (0,0) / (1,0) 元组
        - ["S11", "S21"] 列表
        - None → 返回全参数列表
    返回 [(m, n), ...] 列表。
    """
    if params is None:
        return [(m, n) for m in range(ntwk.nports) for n in range(ntwk.nports)]

    result = []
    items = params if isinstance(params, list) else [params]
    for p in items:
        if isinstance(p, str):
            p = p.strip().upper()
            if p.startswith("S") and len(p) == 3:
                m = int(p[1]) - 1
                n = int(p[2]) - 1
                result.append((m, n))
            # VSWR 不在这里处理——由专门的 vswr 函数处理
        elif isinstance(p, (tuple, list)) and len(p) == 2:
            result.append((int(p[0]), int(p[1])))
    return result


def _parse_vswr_params(ntwk: rf.Network, ports) -> List[int]:
    """标准化 VSWR 端口参数，支持 'VSWR1', 0, [0, 1] 等格式。"""
    if ports is None:
        return list(range(ntwk.nports))
    items = ports if isinstance(ports, list) else [ports]
    result = []
    for p in items:
        if isinstance(p, str):
            p = p.strip().upper()
            if p.startswith("VSWR"):
                result.append(int(p[4:]) - 1)
            else:
                result.append(int(p) - 1 if int(p) > 0 else int(p))
        elif isinstance(p, (int, float)):
            result.append(int(p))
    return result


def _build_hover(freq_ghz, y_data, y_label: str) -> list:
    """构造 hover 模板列表。"""
    return [
        f"Freq: {f:.4f} GHz<br>{y_label}: {v:.3f}"
        for f, v in zip(freq_ghz, y_data)
    ]


def plot_s_db(
    ntwk: rf.Network,
    params: List[Tuple[int, int]] = None,
    title: str = "S-Parameter Magnitude",
    figsize: Tuple[int, int] = (1000, 550),
    show: bool = False,
    save_to: str = None,
) -> go.Figure:
    """
    dB 幅度 vs 频率 （交互式）。

    Args:
        ntwk: 网络对象
        params: 要画的参数 [(0,0), (1,0)]；默认画全部
        title: 图表标题
        figsize: (宽, 高) px
        show: 是否在浏览器中打开
        save_to: 保存为 HTML 的路径（自包含）
    Returns:
        plotly Figure 对象
    """
    params = _parse_params(ntwk, params)

    fig = go.Figure()
    freq_ghz = ntwk.f / _FREQ_UNIT

    for i, (m, n) in enumerate(params):
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        db = ntwk.s_db[:, m, n]
        label = _param_label(m, n, ntwk.name)
        fig.add_trace(go.Scatter(
            x=freq_ghz,
            y=db,
            mode="lines",
            name=label,
            line=dict(color=color, width=1.8),
            hovertemplate="%{customdata}",
            customdata=_build_hover(freq_ghz, db, "dB"),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Magnitude (dB)",
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(
            type="log",
            tickformat=".4g",
            showgrid=True,
            gridcolor="#e0e0e0",
            gridwidth=0.5,
            minor=dict(
                showgrid=True,
                gridcolor="#f0f0f0",
                gridwidth=0.3,
                tickmode="auto",
                tickcount=9,
            ),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#e0e0e0",
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor="#999",
            zerolinewidth=1,
        ),
    )

    if save_to:
        fig.write_html(save_to, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def plot_s_db_dual(
    ntwk: rf.Network,
    left_params=None,
    right_params=None,
    title: str = "S-Parameter Dual Y-Axis",
    figsize: Tuple[int, int] = (1050, 580),
    show: bool = False,
    save_to: str = None,
) -> go.Figure:
    """
    双Y轴 dB 幅度图。左边放 reflection 类参数，右边放 transmission 类参数。

    Args:
        ntwk: 网络对象
        left_params: 左轴参数，默认 S11 等反射参数
        right_params: 右轴参数，默认 S21 等传输参数
        title: 图表标题
        save_to: 保存为 HTML 的路径
    """
    # 默认分配：所有参数，m==n 的放左边（反射），m!=n 的放右边（传输）
    if left_params is None and right_params is None:
        all_params = _parse_params(ntwk, None)
        left_params = [(m, n) for (m, n) in all_params if m == n]
        right_params = [(m, n) for (m, n) in all_params if m != n]
    else:
        left_params = _parse_params(ntwk, left_params) if left_params else []
        right_params = _parse_params(ntwk, right_params) if right_params else []

    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    freq_ghz = ntwk.f / _FREQ_UNIT

    # 左轴 (蓝色系)
    left_colors = ["#1f77b4", "#17becf", "#4da6ff"]
    for i, (m, n) in enumerate(left_params):
        color = left_colors[i % len(left_colors)]
        db = ntwk.s_db[:, m, n]
        label = _param_label(m, n, ntwk.name) + " (左轴)"
        fig.add_trace(go.Scatter(
            x=freq_ghz, y=db, mode="lines", name=label,
            line=dict(color=color, width=2.2),
            hovertemplate=f"<b>{label}</b><br>%{{x:.4f}} GHz<br>%{{y:.3f}} dB<extra></extra>",
        ), secondary_y=False)

    # 右轴 (红/橙色系)
    right_colors = ["#d62728", "#ff7f0e", "#e377c2"]
    for i, (m, n) in enumerate(right_params):
        color = right_colors[i % len(right_colors)]
        db = ntwk.s_db[:, m, n]
        label = _param_label(m, n, ntwk.name) + " (右轴)"
        fig.add_trace(go.Scatter(
            x=freq_ghz, y=db, mode="lines", name=label,
            line=dict(color=color, width=2.2, dash="solid"),
            hovertemplate=f"<b>{label}</b><br>%{{x:.4f}} GHz<br>%{{y:.3f}} dB<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        width=figsize[0], height=figsize[1],
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="Frequency (GHz)", type="log", tickformat=".4g",
                     showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5,
                     minor=dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=0.3))
    fig.update_yaxes(title_text="Magnitude (dB) — 左轴", secondary_y=False,
                     showgrid=True, gridcolor="#dde", zerolinecolor="#445")
    fig.update_yaxes(title_text="Magnitude (dB) — 右轴", secondary_y=True,
                     showgrid=True, gridcolor="#edd", zerolinecolor="#544")

    if save_to:
        fig.write_html(save_to, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def plot_s_deg(
    ntwk: rf.Network,
    params: List[Tuple[int, int]] = None,
    title: str = "S-Parameter Phase",
    figsize: Tuple[int, int] = (1000, 550),
    show: bool = False,
    save_to: str = None,
) -> go.Figure:
    """相位 vs 频率 (度)。"""
    params = _parse_params(ntwk, params)

    fig = go.Figure()
    freq_ghz = ntwk.f / _FREQ_UNIT

    for i, (m, n) in enumerate(params):
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        deg = ntwk.s_deg[:, m, n]
        label = _param_label(m, n, ntwk.name)
        fig.add_trace(go.Scatter(
            x=freq_ghz,
            y=deg,
            mode="lines",
            name=label,
            line=dict(color=color, width=1.8),
            hovertemplate="%{customdata}",
            customdata=_build_hover(freq_ghz, deg, "deg"),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Phase (deg)",
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(type="log", tickformat=".4g", showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5,
                   minor=dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=0.3)),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5, zeroline=True, zerolinecolor="#999", zerolinewidth=1),
    )

    if save_to:
        fig.write_html(save_to, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def plot_s_smith(
    ntwk: rf.Network,
    params: List[Tuple[int, int]] = None,
    title: str = "Smith Chart",
    figsize: Tuple[int, int] = (700, 700),
    show: bool = False,
    save_to: str = None,
) -> go.Figure:
    """
    Smith 圆图 (交互式)。
    用反射系数 Γ 的实部/虚部画在单位圆内。
    """
    params = _parse_params(ntwk, params)

    fig = go.Figure()
    freq_ghz = ntwk.f / _FREQ_UNIT

    # 画 Smith 圆图参考线
    _add_smith_grid(fig)

    for i, (m, n) in enumerate(params):
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        gamma = ntwk.s[:, m, n]
        re, im = np.real(gamma), np.imag(gamma)
        label = _param_label(m, n, ntwk.name)

        # hover 文本包含频率和 Γ
        hover_texts = [
            f"Freq: {f:.4f} GHz<br>Γ: {r:.4f} + j{imv:.4f}<br>|Γ|: {np.abs(g):.4f}"
            for f, r, imv, g in zip(freq_ghz, re, im, gamma)
        ]

        fig.add_trace(go.Scatter(
            x=re,
            y=im,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=1.8),
            marker=dict(size=4),
            text=hover_texts,
            hovertemplate="%{text}",
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis=dict(
            title="Real (Γ)",
            range=[-1.1, 1.1],
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
        ),
        yaxis=dict(
            title="Imag (Γ)",
            range=[-1.1, 1.1],
        ),
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )

    if save_to:
        fig.write_html(save_to, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def _add_smith_grid(fig: go.Figure, n_circles: int = 6):
    """在 Smith 图上添加参考圆（等电阻圆、等电抗弧）。"""
    theta = np.linspace(0, 2 * np.pi, 360)
    # 等 |Γ| 圆
    for r in np.linspace(0.2, 1.0, 5):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color="lightgray", width=0.6, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
    # 单位圆
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode="lines",
        line=dict(color="gray", width=1.2),
        showlegend=False, hoverinfo="skip", name="Unit Circle",
    ))
    # 水平轴
    fig.add_trace(go.Scatter(
        x=[-1, 1], y=[0, 0], mode="lines",
        line=dict(color="gray", width=0.8, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    # 垂直轴
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-1, 1], mode="lines",
        line=dict(color="gray", width=0.8, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))


def plot_vswr(
    ntwk: rf.Network,
    ports: List[int] = None,
    title: str = "VSWR",
    figsize: Tuple[int, int] = (1000, 550),
    show: bool = False,
    save_to: str = None,
) -> go.Figure:
    """VSWR vs 频率。"""
    ports = _parse_vswr_params(ntwk, ports)

    fig = go.Figure()
    freq_ghz = ntwk.f / _FREQ_UNIT

    for i, port in enumerate(ports):
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        vswr = ntwk.s_vswr[:, port, port]
        label = f"VSWR{port+1} ({ntwk.name})"
        fig.add_trace(go.Scatter(
            x=freq_ghz,
            y=vswr,
            mode="lines",
            name=label,
            line=dict(color=color, width=1.8),
            hovertemplate="%{customdata}",
            customdata=_build_hover(freq_ghz, vswr, "VSWR"),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Frequency (GHz)",
        yaxis_title="VSWR",
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(type="log", tickformat=".4g", showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5,
                   minor=dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=0.3)),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5),
    )

    if save_to:
        fig.write_html(save_to, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def plot_group_delay(
    ntwk: rf.Network,
    params: List[Tuple[int, int]] = None,
    title: str = "Group Delay",
    figsize: Tuple[int, int] = (1000, 550),
    show: bool = False,
    save_to: str = None,
) -> go.Figure:
    """群时延 vs 频率 (ns)。"""
    params = _parse_params(ntwk, params)

    fig = go.Figure()
    freq_ghz = ntwk.f / _FREQ_UNIT

    for i, (m, n) in enumerate(params):
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        gd = get_group_delay(ntwk, m, n)
        label = f"GD S{m+1}{n+1} ({ntwk.name})"
        fig.add_trace(go.Scatter(
            x=freq_ghz,
            y=gd,
            mode="lines",
            name=label,
            line=dict(color=color, width=1.8),
            hovertemplate="%{customdata}",
            customdata=_build_hover(freq_ghz, gd, "ns"),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Group Delay (ns)",
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(type="log", tickformat=".4g", showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5,
                   minor=dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=0.3)),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5, zeroline=True, zerolinecolor="#999", zerolinewidth=1),
    )

    if save_to:
        fig.write_html(save_to, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


# ─── 6. 多文件对比画图 ──────────────────────────────────────────────────

def plot_multi_db(
    networks: List[rf.Network],
    names: List[str] = None,
    param: Tuple[int, int] = (1, 0),
    title: str = "Multi-File S-Parameter Comparison (dB)",
    figsize: Tuple[int, int] = (1100, 600),
    save_to: str = None,
    show: bool = False,
    show_diff: bool = False,
    reference_idx: int = 0,
) -> go.Figure:
    """
    多个网络文件的同一参数 dB 对比。

    Args:
        networks: 网络列表（若 show_diff=True，需已插值到同一频率网格）
        names: 图例名称列表
        param: S 参数索引 (m, n)
        show_diff: 是否显示相对于参考网络的差异曲线
        reference_idx: 差异计算的参考网络索引
    """
    if names is None:
        names = [n.name for n in networks]

    fig = go.Figure()

    for i, ntwk in enumerate(networks):
        freq_ghz = ntwk.f / _FREQ_UNIT
        db = ntwk.s_db[:, param[0], param[1]]
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        fig.add_trace(go.Scatter(
            x=freq_ghz,
            y=db,
            mode="lines",
            name=names[i],
            line=dict(color=color, width=2),
            hovertemplate="%{customdata}",
            customdata=_build_hover(freq_ghz, db, "dB"),
        ))

    # 差异曲线
    if show_diff and len(networks) >= 2:
        ref = networks[reference_idx]
        ref_db = ref.s_db[:, param[0], param[1]]
        ref_freq = ref.f / _FREQ_UNIT
        diff_colors = ["#e377c2", "#bcbd22", "#8c564b", "#7f7f7f"]
        for i, ntwk in enumerate(networks):
            if i == reference_idx:
                continue
            db = ntwk.s_db[:, param[0], param[1]]
            diff = db - ref_db
            dcolor = diff_colors[(i - 1) % len(diff_colors)]
            diff_name = f"Δ ({names[i]} − {names[reference_idx]})"
            fig.add_trace(go.Scatter(
                x=ref_freq,
                y=diff,
                mode="lines",
                name=diff_name,
                line=dict(color=dcolor, width=1.2, dash="dot"),
                yaxis="y2",
                hovertemplate="%{customdata}",
                customdata=_build_hover(ref_freq, diff, "Δ dB"),
            ))

        # 统计标注
        stats = compute_diff_stats(networks, param, reference_idx)
        annot_lines = [f"<b>差异统计 (vs {stats['reference']})</b>"]
        for c in stats["comparisons"]:
            annot_lines.append(
                f"{c['name']}: mean={c['mean_diff_db']:+.2f} dB, "
                f"max|Δ|={c['max_diff_db']:.2f} dB, RMS={c['rms_diff_db']:.2f} dB"
            )
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text="<br>".join(annot_lines),
            showarrow=False,
            bgcolor="rgba(26,26,46,0.9)",
            font=dict(color="#c0c0c0", size=11, family="monospace"),
            align="left",
            bordercolor="#444",
            borderwidth=1,
            borderpad=8,
        )

        fig.update_layout(
            yaxis2=dict(
                title="Δ (dB)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

    param_str = f"S{param[0]+1}{param[1]+1}"
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Frequency (GHz)",
        yaxis_title=f"{param_str} (dB)",
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        xaxis=dict(type="log", tickformat=".4g", showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5,
                   minor=dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=0.3)),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5, zeroline=True, zerolinecolor="#999", zerolinewidth=1),
    )

    if save_to:
        fig.write_html(save_to, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def plot_multi_smith(
    networks: List[rf.Network],
    names: List[str] = None,
    param: Tuple[int, int] = (0, 0),
    title: str = "Multi-File Smith Chart Comparison",
    figsize: Tuple[int, int] = (750, 750),
    save_to: str = None,
    show: bool = False,
) -> go.Figure:
    """多个文件同一参数的 Smith 圆图对比。"""
    if names is None:
        names = [n.name for n in networks]

    fig = go.Figure()
    _add_smith_grid(fig)

    for i, ntwk in enumerate(networks):
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        gamma = ntwk.s[:, param[0], param[1]]
        re, im = np.real(gamma), np.imag(gamma)
        freq_ghz = ntwk.f / _FREQ_UNIT

        hover_texts = [
            f"{names[i]}<br>Freq: {f:.4f} GHz<br>Γ: {r:.4f}+j{imv:.4f}<br>|Γ|: {np.abs(g):.4f}"
            for f, r, imv, g in zip(freq_ghz, re, im, gamma)
        ]

        fig.add_trace(go.Scatter(
            x=re,
            y=im,
            mode="lines+markers",
            name=names[i],
            line=dict(color=color, width=2),
            marker=dict(size=3),
            text=hover_texts,
            hovertemplate="%{text}",
        ))

    param_str = f"S{param[0]+1}{param[1]+1}"
    fig.update_layout(
        title=dict(text=f"{title} – {param_str}", x=0.5, font=dict(size=18)),
        xaxis=dict(range=[-1.1, 1.1], scaleanchor="y", scaleratio=1, constrain="domain"),
        yaxis=dict(range=[-1.1, 1.1]),
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )

    if save_to:
        fig.write_html(save_to, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def plot_compare(
    networks: List[rf.Network],
    names: List[str] = None,
    params: List[Tuple[int, int]] = None,
    title: str = "Multi-File Multi-Parameter Comparison",
    figsize: Tuple[int, int] = (1200, 700),
    show_diff: bool = True,
    reference_idx: int = 0,
    save_to: str = None,
    show: bool = False,
) -> go.Figure:
    """
    全功能对比视图：多文件 × 多参数，带差异统计。
    自动插值到共同频率网格。

    Args:
        networks: 网络列表（可不同频率网格，自动插值）
        names: 图例名称
        params: 要对比的参数列表，默认取所有网络共有的第一个传输参数
        show_diff: 显示差异曲线
        reference_idx: 差异参考基准
    """
    if len(networks) < 2:
        raise ValueError(f"对比至少需要 2 个网络，提供了 {len(networks)} 个")

    if names is None:
        names = [n.name for n in networks]

    # 自动插值到共同频率
    try:
        interpolated = interpolate_to_common_freq(networks, npoints=max(len(n.f) for n in networks))
    except ValueError:
        # 无共同范围 → 使用各自频率直接画
        interpolated = networks
        show_diff = False

    # 默认参数：取第一个传输参数 S21
    if params is None:
        for m in range(interpolated[0].nports):
            for n_val in range(interpolated[0].nports):
                if m != n_val:
                    params = [(m, n_val)]
                    break
            if params:
                break
        if params is None:
            params = [(0, 0)]

    from plotly.subplots import make_subplots

    n_params = len(params)
    subplot_titles = [f"S{p[0]+1}{p[1]+1} (dB)" for p in params]
    if show_diff:
        subplot_titles += [f"Δ S{p[0]+1}{p[1]+1} (dB)" for p in params]

    n_cols = min(n_params, 3)
    n_rows = (n_params + n_cols - 1) // n_cols
    if show_diff:
        n_rows *= 2

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
    )

    ref_interp = interpolated[reference_idx]

    for pi, (m, n_val) in enumerate(params):
        row_main = (pi // n_cols) * (2 if show_diff else 1) + 1
        col = pi % n_cols + 1

        # 主图：dB 幅度
        for i, ntwk in enumerate(interpolated):
            freq_ghz = ntwk.f / _FREQ_UNIT
            db = ntwk.s_db[:, m, n_val]
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            fig.add_trace(go.Scatter(
                x=freq_ghz, y=db, mode="lines", name=names[i],
                line=dict(color=color, width=1.8),
                hovertemplate=f"<b>{names[i]}</b><br>%{{x:.4f}} GHz<br>%{{y:.3f}} dB<extra></extra>",
                legendgroup=names[i],
                showlegend=(pi == 0),
            ), row=row_main, col=col)

        # 差异图
        if show_diff:
            diff_row = row_main + 1
            ref_db = ref_interp.s_db[:, m, n_val]
            ref_freq = ref_interp.f / _FREQ_UNIT
            for i, ntwk in enumerate(interpolated):
                if i == reference_idx:
                    continue
                db = ntwk.s_db[:, m, n_val]
                diff = db - ref_db
                dcolor = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
                fig.add_trace(go.Scatter(
                    x=ref_freq, y=diff, mode="lines",
                    name=f"Δ ({names[i]} − {names[reference_idx]})",
                    line=dict(color=dcolor, width=1.2, dash="dot"),
                    hovertemplate=f"<b>Δ ({names[i]})</b><br>%{{x:.4f}} GHz<br>%{{y:.3f}} dB<extra></extra>",
                    legendgroup=f"diff_{names[i]}",
                    showlegend=(pi == 0),
                ), row=diff_row, col=col)

    for row in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Frequency (GHz)", row=row, col=1,
                         type="log", tickformat=".4g",
                         showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5,
                         minor=dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=0.3))
    fig.update_yaxes(title_text="S-Parameter (dB)",
                     showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5,
                     zeroline=True, zerolinecolor="#999", zerolinewidth=1)
    if show_diff:
        fig.update_yaxes(title_text="Δ (dB)", row=n_rows, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        width=figsize[0],
        height=figsize[1] * (n_rows / 2),
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )

    if save_to:
        fig.write_html(save_to, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


# ─── 7. 综合报告页 ─────────────────────────────────────────────────────

def generate_report(
    ntwk: rf.Network,
    output_path: str = "s_param_report.html",
    title: str = "S-Parameter Report",
) -> str:
    """
    生成一个综合 HTML 报告页，包含：
    - 基本信息表
    - dB 幅度图（全部 S 参数）
    - 相位图
    - Smith 圆图 (仅反射参数)
    - VSWR 图

    所有图表均为交互式 Plotly，嵌入单个 HTML 文件。
    """
    # 生成各子图
    fig_db = plot_s_db(ntwk, show=False)
    fig_deg = plot_s_deg(ntwk, show=False)

    refl_params = [(p, p) for p in range(ntwk.nports)]
    fig_smith = plot_s_smith(ntwk, params=refl_params, show=False)
    fig_vswr = plot_vswr(ntwk, show=False)

    # 基本信息
    info_text = info(ntwk).replace("\n", "<br>")

    # 组合成 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: #f5f5f5;
  }}
  h1 {{ text-align: center; color: #333; }}
  .info-card {{
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}
  .info-card h2 {{ margin-top: 0; color: #1a73e8; }}
  .info-card p {{ line-height: 1.8; font-family: 'SF Mono', 'Consolas', monospace; font-size: 14px; color: #555; }}
  .chart-container {{
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin: 30px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}
  .chart-container h2 {{
    margin: 0 0 15px 0;
    color: #1a73e8;
    font-size: 18px;
    border-bottom: 2px solid #e8f0fe;
    padding-bottom: 10px;
  }}
</style>
</head>
<body>
<h1>{title}</h1>

<div class="info-card">
  <h2>📋 Network Info</h2>
  <p>{info_text}</p>
</div>

<div class="chart-container">
  <h2>📈 Magnitude (dB)</h2>
  <div id="chart_db"></div>
</div>

<div class="chart-container">
  <h2>📐 Phase (deg)</h2>
  <div id="chart_deg"></div>
</div>

<div class="chart-container">
  <h2>🎯 Smith Chart</h2>
  <div id="chart_smith"></div>
</div>

<div class="chart-container">
  <h2>📊 VSWR</h2>
  <div id="chart_vswr"></div>
</div>

<script>
  var db_spec = {fig_db.to_json()};
  var deg_spec = {fig_deg.to_json()};
  var smith_spec = {fig_smith.to_json()};
  var vswr_spec = {fig_vswr.to_json()};
  Plotly.newPlot('chart_db', db_spec.data, db_spec.layout, {{responsive: true}});
  Plotly.newPlot('chart_deg', deg_spec.data, deg_spec.layout, {{responsive: true}});
  Plotly.newPlot('chart_smith', smith_spec.data, smith_spec.layout, {{responsive: true}});
  Plotly.newPlot('chart_vswr', vswr_spec.data, vswr_spec.layout, {{responsive: true}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ─── 8. 便捷函数 ───────────────────────────────────────────────────────

def fig_to_html(fig: go.Figure, path: str) -> str:
    """将 Plotly Figure 保存为自包含 HTML，返回路径。"""
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def fig_to_base64_png(fig: go.Figure) -> str:
    """将 Plotly Figure 渲染为 base64 PNG 字符串（用于内嵌）。"""
    img_bytes = fig.to_image(format="png", width=1200, height=700, scale=1.5)
    return base64.b64encode(img_bytes).decode("utf-8")
