"""研究量具必须跨河段接合计长，且不能把总宽河长度当连续长度。"""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("river_width_probe", Path(__file__).parents[1] / "docs/probes/river_width.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_connected_widths_include_joins_and_diagonals():
    # 两条支流 0->4->8、1->5->8 在 8 汇合，随后 8->12；每个点可以各属一个 segment。
    down = np.full(16, -1)
    down[[0, 4, 1, 5, 8]] = [4, 8, 5, 8, 12]
    area = np.zeros(16)
    area[[0, 1, 4, 5, 8, 12]] = [1, 1, 2, 2, 5, 6]
    width = np.full(16, 20.0)
    cells = [12, 5, 0, 8, 4, 1]
    stats = probe.connected_widths(cells, down, area, width, 4, 1.0)
    assert stats["length_km"] == round(4 + np.sqrt(2), 3)
    assert stats["continuous_ge_m_max_km"]["15"] == round(2 + np.sqrt(2), 3)
    assert stats["total_ge_m_km"]["15"] == stats["length_km"]
    # 汇合处窄下来会切断连续宽河，不可越过它累计两个宽段。
    width[8] = 10.0
    interrupted = probe.connected_widths(cells, down, area, width, 4, 1.0)
    assert interrupted["continuous_ge_m_max_km"]["15"] == 1.0
