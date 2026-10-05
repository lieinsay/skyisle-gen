"""Partition, emergence and confluence tests for one-count river water budgets."""
import numpy as np
import pytest

from test_aquifer_balance import core


def routed(rain=1e-7, core_water=0., fraction=.4, head=0., k=0., top=1., recv=None, **kw):
    shape = (1, 3)
    grid = lambda x: np.broadcast_to(np.asarray(x, dtype=float), shape).copy()
    return core.step_spring_routing(grid(top), grid(0.), grid(k), grid(rain),
        grid(core_water), grid(fraction), np.ones(shape, dtype=bool),
        np.array([1, 2, -1] if recv is None else recv, dtype=np.int64),
        grid(head), grid(.1), 100., 86400., **kw)


def closed(r):
    a = r['aquifer']
    supply = r['rain_input_m3s'] + r['core_input_m3s']
    exits = r['surface_outlet_m3s'] + np.sum(a['coast_m3s'])
    assert (supply-exits)*86400 == pytest.approx(a['total_storage_change_m3'], abs=2e-6)
    assert r['surface_outlet_m3s'] == pytest.approx(r['local_surface_m3s'].sum())


def test_rain_partition_is_not_added_to_river_twice():
    r = routed()
    closed(r)
    np.testing.assert_allclose(r['river_m3s'], [[.0006, .0012, .0018]])
    assert r['aquifer']['recharge_m3s'] == pytest.approx(.0012)
    assert not np.any(r['aquifer']['surface_m3s'])


def test_core_stays_underground_until_emergence():
    r = routed(rain=0., core_water=1e-7)
    closed(r)
    assert not np.any(r['river_m3s'])
    assert r['aquifer']['total_storage_change_m3'] > 0
    seep = routed(rain=0., core_water=1e-7, head=1.)
    closed(seep)
    np.testing.assert_allclose(seep['river_m3s'], [[.001, .002, .003]])


def test_coastal_groundwater_is_not_inland_river_supply():
    r = routed(rain=0., core_water=1e-8, k=1e-4, head=.5, top=10.)
    closed(r)
    assert np.sum(r['aquifer']['coast_m3s']) > 0
    assert not np.any(r['river_m3s'])


def test_confluence_preserves_donor_sum_and_dry_event_can_end():
    # Two tributaries meet at cell 1. No river/perennial mask is supplied.
    wet = routed(rain=[[1e-7, 0., 2e-7]], fraction=0., recv=[1, -1, 1])
    closed(wet)
    q = wet['river_m3s'][0]
    assert q[1] == pytest.approx(q[0]+q[2])
    dry = routed(rain=0., fraction=0., recv=[1, -1, 1])
    closed(dry)
    assert not np.any(dry['river_m3s'])


@pytest.mark.parametrize('kw', [dict(fraction=1.1), dict(rain=-1.),
    dict(core_water=-1.), dict(recv=[1, 0, -1]), dict(recv=[1, 3, -1]),
    dict(recv=[2, 2, -1])])
def test_invalid_source_or_routing_is_rejected(kw):
    with pytest.raises(ValueError):
        routed(**kw)


def test_unconverged_aquifer_cannot_be_routed_as_valid_spring():
    with pytest.raises(RuntimeError, match='did not converge'):
        routed(k=1e-3, head=.5, top=10., max_iterations=1)
