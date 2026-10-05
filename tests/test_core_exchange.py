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


def run(case, column=None, tau=0., gain=None, footprint=0., mountain=False, activity=None, steps=2):
    inp, planet, cfg = case
    inp, cfg = copy.deepcopy(inp), copy.deepcopy(cfg)
    inp['water_column_mm'] = column
    cfg['water']['core_exchange_days'] = tau
    cfg['water']['core_footprint_scale'] = footprint
    cfg['water']['core_mountain_domain'] = mountain
    if activity is not None:
        cfg['water']['core_activity_per_km_day'] = activity
    if gain is not None:
        cfg['water']['core_gain'] = gain
    return core.generate(inp, planet, flat_config(cfg), steps=steps, res_m=300., threads=2)


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


def test_footprint_has_finite_extent_and_source_ledger(case):
    a = run(case, [10.]*4, 4., footprint=1.)
    h, t = a['hydro'], a['terrain']
    ids = t['island_id']
    ri, cj = np.indices(ids.shape)
    x = t['origin_x']+(cj+.5)*t['res_km']
    y = t['origin_y']-(ri+.5)*t['res_km']
    within = np.zeros(ids.shape, dtype=bool)
    expected_volume = 0.
    for s in h['core_water_sources']:
        within |= (ids == s['island']) & ((x-s['x_km'])**2+(y-s['y_km'])**2 < s['radius_km']**2)
        expected_volume += s['condense_m3s']
    assert np.any(h['condense_mm'] > 0)
    assert not np.any(h['condense_mm'][~within])
    volume = h['condense_mm'].astype(float).sum()*t['res_km']**2*1000/h['year_s']
    assert volume == pytest.approx(expected_volume, rel=1e-7)


def test_footprint_has_no_rain_cap_or_legacy_gain(case):
    # Large test column proves removal of the rain ceiling; not a proposal for
    # production humidity or core capability.
    a = run(case, [100.]*4, 4., footprint=2.)['hydro']
    b = run(case, [200.]*4, 8., gain=100., footprint=2.)['hydro']
    assert np.any(a['condense_mm'] > a['rain_mm'])
    np.testing.assert_array_equal(a['condense_mm'], b['condense_mm'])
    np.testing.assert_array_equal(a['rain_mm'], b['rain_mm'])
    zero = run(case, [0.]*4, 4., footprint=2.)['hydro']
    assert not np.any(zero['condense_mm'])


@pytest.mark.parametrize('tau,scale', [(0., 1.), (4., -1.), (4., float('nan'))])
def test_invalid_footprint_inputs(case, tau, scale):
    with pytest.raises(ValueError):
        run(case, [1.]*4, tau, footprint=scale)


def test_mountain_domain_capacity_and_source_ledger(case):
    result = run(case, [10.]*4, mountain=True, activity=.5)
    h, t = result['hydro'], result['terrain']
    land = t['island_id'] >= 0
    sources = h['core_water_sources']
    assert np.all(h['condense_mm'][land] > 0)
    assert not np.any(h['condense_mm'][~land])
    assert sum(s['domain_area_km2'] for s in sources) == pytest.approx(land.sum()*t['res_km']**2)
    for s in sources:
        assert s['capacity_m3s'] == pytest.approx(10*s['mountain_volume_km3']*.5*1000/86400)
        assert s['condense_m3s'] == pytest.approx(s['capacity_m3s'], rel=1e-12)
    actual = h['condense_mm'].astype(float).sum()*t['res_km']**2*1000/h['year_s']
    assert actual == pytest.approx(sum(s['capacity_m3s'] for s in sources), rel=1e-7)
    # Ownership export must retain the terrain stage's exact membership.
    for name in ['core_member', 'core_neighbor', 'core_member_weight']:
        np.testing.assert_array_equal(t[name], h[name])


def test_mountain_ability_is_linear_and_independent_of_old_gain(case):
    a = run(case, [10.]*4, mountain=True, activity=.5)
    b = run(case, [10.]*4, mountain=True, activity=1.)
    c = run(case, [20.]*4, gain=100, mountain=True, activity=.5)
    np.testing.assert_allclose(b['hydro']['condense_mm'], 2*a['hydro']['condense_mm'], rtol=2e-7)
    np.testing.assert_array_equal(b['hydro']['condense_mm'], c['hydro']['condense_mm'])
    # Tracking ownership cannot alter terrain RNG or rainfall.
    old = run(case)
    original = run(case, steps=1)['terrain']
    tracked = run(case, mountain=True, steps=1)['terrain']
    np.testing.assert_array_equal(original['height'], tracked['height'])
    np.testing.assert_array_equal(old['hydro']['rain_mm'], a['hydro']['rain_mm'])
    zero = run(case, [10.]*4, mountain=True, activity=0.)
    assert not np.any(zero['hydro']['condense_mm'])
    high = run(case, [1e4]*4, mountain=True, activity=.5)
    assert np.any(high['hydro']['condense_mm'] > high['hydro']['rain_mm'])


@pytest.mark.parametrize('column,activity,tau,footprint', [
    (None, .5, 0, 0), ([1.]*4, None, 0, 0), ([1.]*4, -1, 0, 0),
    ([1.]*4, float('nan'), 0, 0), ([1.]*4, .5, 4, 0), ([1.]*4, .5, 4, 2)])
def test_invalid_mountain_inputs(case, column, activity, tau, footprint):
    with pytest.raises(ValueError):
        run(case, column, tau, footprint=footprint, mountain=True, activity=activity)


def test_multicore_mountain_shares_do_not_duplicate_load(case):
    inp, planet, cfg = case
    cfg = copy.deepcopy(cfg)
    cfg['terrain'].update(multicore_frac=1., multicore_min_km2=1.,
        multicore_kernel_full=1e-10, multicore_three_frac=1., age_young=0.)
    # Choose an existing convergent input; do not fake plate or core locations.
    pc = core.make_config(planet_config(load_config(sets=['s03.islands.n_islands=1600'])))
    _, _, islands, climate = core.planet_run(pc, 7, 4, 4, False)
    arrays = core.islands_arrays(islands)
    candidates = np.flatnonzero((arrays['main_area_km2'] > 200) & (arrays['main_area_km2'] < 500))
    selected = None
    for node in candidates:
        candidate = core.node_inputs(islands, climate, int(node), 7, pc)
        terrain = run((candidate, planet, cfg), [10.]*4, mountain=True, activity=.5, steps=1)['terrain']
        if len(terrain['islands'][0]['cores']) > 1:
            selected = candidate
            break
    assert selected is not None, 'Fixture must exercise actual multicore blending'
    r = run((selected, planet, cfg), [10.]*4, mountain=True, activity=.5)
    t, h = terrain, r['hydro']
    for island in t['islands']:
        pick = t['island_id'] == island['id']
        sources = [s for s in h['core_water_sources'] if s['island'] == island['id']]
        if not sources:
            continue
        # condensation precedes channel incision, using pre-hydro terrain/rim.
        volume = np.maximum(t['height'][pick]-round(island['rim'], 1), 0).sum()*t['res_km']**2/1000
        assert sum(s['mountain_volume_km3'] for s in sources) == pytest.approx(volume, rel=1e-10)
        assert sum(s['domain_area_km2'] for s in sources) == pytest.approx(pick.sum()*t['res_km']**2)
        for s in sources:
            w = np.where(h['core_member'] == s['core_index'], h['core_member_weight'], 0.)
            w += np.where(h['core_neighbor'] == s['core_index'], 1-h['core_member_weight'], 0.)
            assert s['domain_area_km2'] == pytest.approx(w[pick].sum()*t['res_km']**2)
            assert s['condense_m3s'] == pytest.approx(s['capacity_m3s'], rel=1e-12)
    assert np.any((h['core_member_weight'] > .5) & (h['core_member_weight'] < 1))
    # The separate terrain/hydro entry point must carry the exact same domains.
    state = dict(inp=copy.deepcopy(selected), height=t['height'], island_id=t['island_id'],
        cliff=t['cliff'], res_km=t['res_km'], origin_x=t['origin_x'], origin_y=t['origin_y'],
        strat_top=t['strat_top'], skel_top=t['skel_top'], islands=[])
    state['inp']['water_column_mm'] = [10.]*4
    for name in ['core_member', 'core_neighbor', 'core_member_weight']:
        state[name] = t[name]
    for island in t['islands']:
        state['islands'].append(dict(rim_m=round(island['rim'], 1), keel_m=round(island['keel'], 1),
            peak_m=round(island['peak'], 1), age=round(island['age'], 3), young=False,
            gc=island['gc'], cores=island['cores'], strat=island['strat']))
    cfg['water'].update(core_mountain_domain=True, core_activity_per_km_day=.5,
        core_exchange_days=0., core_footprint_scale=0.)
    staged = core.build_hydro(state, planet, flat_config(cfg), threads=2)
    np.testing.assert_array_equal(staged['condense_mm'], h['condense_mm'])
    assert staged['core_water_sources'] == h['core_water_sources']
