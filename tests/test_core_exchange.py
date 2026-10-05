"""Opt-in world-setting model: units, source isolation and missing-input errors.

These tests verify computation and invariants, not the physical validity of the
chosen exchange time, mountain footprint, rainfall ceiling or channel geometry.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
core = pytest.importorskip('skyisle_gen._core')
from skyisle_gen.config import load_config
from skyisle_gen.engine import planet_config
from skyisle_gen.island.engine import flat_config


@pytest.fixture(scope='module')
def case():
    cfg = load_config(sets=['s03.islands.n_islands=1600'])
    pc = core.make_config(planet_config(cfg))
    P, W, I, C = core.planet_run(pc, 7, 4, 4, False)
    a = core.islands_arrays(I)
    node = int(np.flatnonzero((a['main_area_km2'] > 200) & (a['main_area_km2'] < 500))[0])
    inp = core.node_inputs(I, C, node, 7, pc)
    assert len(inp['water_column_mm']) == 4
    return inp, core.planet_view(P, I, C, pc), cfg['island']


def run(case, column=None, tau=0., gain=None):
    inp, planet, cfg = case
    inp, cfg = copy.deepcopy(inp), copy.deepcopy(cfg)
    inp['water_column_mm'] = column
    cfg['water']['core_exchange_days'] = tau
    if gain is not None:
        cfg['water']['core_gain'] = gain
    return core.generate(inp, planet, flat_config(cfg), steps=2, res_m=300., threads=2)


def test_column_units_and_source_isolation(case):
    # Small explicit test columns keep the inherited rain ceiling inactive.
    a = run(case, [0.01]*4, 4.)['hydro']
    b = run(case, [0.02]*4, 4.)['hydro']
    c = run(case, [0.02]*4, 8., gain=100.)['hydro']
    assert a['condense_mm'].max() > 0
    np.testing.assert_allclose(b['condense_mm'], 2*a['condense_mm'], rtol=2e-7)
    np.testing.assert_array_equal(c['condense_mm'], a['condense_mm'])
    for name in ['rain_mm', 'river', 'stream']:
        np.testing.assert_array_equal(a[name], b[name])
    # More core water belongs to recharge; it is not a second rain input.
    wet = a['condense_mm'] > 0
    np.testing.assert_allclose((b['runoff_mm']-a['runoff_mm'])[wet],
                               (b['condense_mm']-a['condense_mm'])[wet], atol=3e-4)


def test_zero_water_ceiling_and_legacy(case):
    zero = run(case, [0.]*4, 4.)['hydro']
    assert not np.any(zero['condense_mm'])
    cap = run(case, [1.e6]*4, 4.)['hydro']
    assert np.all(cap['condense_mm'] <= cap['rain_mm'])
    old_a = run(case)['hydro']
    old_b = run(case, [1.e6]*4)['hydro']
    for name in ['condense_mm', 'rain_mm', 'runoff_mm', 'river', 'stream']:
        np.testing.assert_array_equal(old_a[name], old_b[name])


@pytest.mark.parametrize('column,tau', [(None, 4.), ([-1.]*4, 4.), ([float('nan')]*4, 4.), ([1.]*4, -1.)])
def test_missing_or_invalid_inputs_fail(case, column, tau):
    with pytest.raises(ValueError):
        run(case, column, tau)
