#!/usr/bin/env python3
"""
自验证脚本：生成示例 S 参数数据，跑通 读→处理→画图→导出 全流程。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import skrf as rf
import s_params as sp

OUT = os.path.join(os.path.dirname(__file__), "demo_output")
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  S-Parameter Agent — 自验证")
print("=" * 60)

# ── 1. 生成示例双端口网络 ──
print("\n[1/6] 生成示例 2 端口 S 参数...")
freq = rf.Frequency(1, 5, 101, unit="ghz")
# 模拟一个带通滤波器：S21 在 3GHz 附近有峰值，S11 匹配良好
f_ghz = freq.f / 1e9

# S11: -15dB 回波，中心 3GHz
s11_mag = 10 ** (-15 / 20) * (1 + 0.3 * np.sin((f_ghz - 1) * np.pi / 2))
s11_phase = -2 * np.pi * (f_ghz - 1) / 4
s11 = s11_mag * np.exp(1j * s11_phase)

# S21: -1dB 插损，中心 3GHz
s21_mag = 10 ** (-1 / 20) * np.exp(-((f_ghz - 3) / 0.8) ** 2)
s21_phase = -2 * np.pi * f_ghz * 0.5
s21 = s21_mag * np.exp(1j * s21_phase)

s = np.zeros((len(freq), 2, 2), dtype=complex)
s[:, 0, 0] = s11
s[:, 1, 0] = s21
s[:, 0, 1] = s21  # 互易
s[:, 1, 1] = s11  # 对称

ntwk = rf.Network(frequency=freq, s=s, z0=50, name="DemoFilter")
tmp_s2p = os.path.join(OUT, "demo_filter.s2p")
ntwk.write_touchstone(tmp_s2p)
print(f"   已写入: {tmp_s2p}")

# ── 2. 加载与概览 ──
print("\n[2/6] 加载并查看信息...")
ntwk_loaded = sp.load_ntwk(tmp_s2p)
sp.info(ntwk_loaded)

d = sp.summary(ntwk_loaded)
assert d["nports"] == 2
assert d["npoints"] == 101
print("   summary() 通过 ✓")

# ── 3. 数据提取 ──
print("\n[3/6] 测试数据提取...")
s21_db = sp.get_s_db(ntwk_loaded, 1, 0)
assert len(s21_db) == 101
print(f"   S21 dB 范围: {s21_db.min():.2f} ~ {s21_db.max():.2f}")

vswr1 = sp.get_vswr(ntwk_loaded, 0)
print(f"   VSWR1 范围: {vswr1.min():.2f} ~ {vswr1.max():.2f}")

gd = sp.get_group_delay(ntwk_loaded, 1, 0)
print(f"   群时延范围: {gd.min():.4f} ~ {gd.max():.4f} ns")

z1 = sp.get_z(ntwk_loaded, 0)
print(f"   Z1 实部范围: {z1.real.min():.1f} ~ {z1.real.max():.1f} Ω")

# ── 4. 处理 ──
print("\n[4/6] 测试处理操作...")
sliced = sp.slice_freq(ntwk_loaded, "2-4ghz")
print(f"   截取 2-4GHz: {len(sliced.f)} 点 (原 {len(ntwk_loaded.f)} 点)")

result = sp.cascade(ntwk_loaded, ntwk_loaded)
print(f"   级联: {result.nports} 端口 ✓")

z50 = sp.renormalize(ntwk_loaded, 75)
print(f"   重归一化 75Ω ✓")

# ── 5. 画交互式图表 ──
print("\n[5/6] 生成交互式 HTML 图表...")

# dB 图
sp.plot_s_db(ntwk_loaded, ["S11", "S21"],
             title="Demo Filter — S-Parameters (dB)",
             save_to=os.path.join(OUT, "demo_db.html"))
print(f"   {OUT}/demo_db.html ✓")

# 相位图
sp.plot_s_deg(ntwk_loaded, ["S21"],
              title="Demo Filter — S21 Phase",
              save_to=os.path.join(OUT, "demo_phase.html"))
print(f"   {OUT}/demo_phase.html ✓")

# Smith 圆图
sp.plot_s_smith(ntwk_loaded, ["S11"],
                title="Demo Filter — S11 Smith Chart",
                save_to=os.path.join(OUT, "demo_smith.html"))
print(f"   {OUT}/demo_smith.html ✓")

# VSWR
sp.plot_vswr(ntwk_loaded, [0, 1],
             title="Demo Filter — VSWR",
             save_to=os.path.join(OUT, "demo_vswr.html"))
print(f"   {OUT}/demo_vswr.html ✓")

# ── 6. 导出 ──
print("\n[6/6] 测试导出...")
sp.save_touchstone(ntwk_loaded, os.path.join(OUT, "exported.s2p"))
print(f"   Touchstone: {OUT}/exported.s2p ✓")

sp.export_csv(ntwk_loaded, ["S11", "S21"], os.path.join(OUT, "exported.csv"))
print(f"   CSV: {OUT}/exported.csv ✓")

# 综合报告
sp.generate_report(ntwk_loaded, os.path.join(OUT, "demo_report.html"))
print(f"   综合报告: {OUT}/demo_report.html ✓")

print("\n" + "=" * 60)
print("  ✅ 全部验证通过！")
print(f"  输出目录: {OUT}")
print("  浏览器打开任意 .html 文件查看交互图表")
print("=" * 60)
