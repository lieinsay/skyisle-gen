"""Independent mass/analytic checks; material values here define test fixtures.

No fixture calibrates world conductivity or a desired channel width.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
core = pytest.importorskip('skyisle_gen._core')


def solve(top, recharge=1.e-8, conductivity=1.e-5, bottom=0., cell_m=100., **kw):
    top = np.asarray(top, dtype=np.float64)
    return core.solve_aquifer_balance(top, np.full_like(top, bottom),
        np.full_like(top, conductivity), np.full_like(top, recharge),
        np.ones(top.shape, dtype=bool), cell_m, **kw)


def closed(r):
    assert r['converged']
    assert r['discharge_m3s'] == pytest.approx(r['recharge_m3s'], rel=1e-7, abs=1e-12)
    assert np.abs(r['residual_m3s']).sum() <= max(1e-12, r['recharge_m3s']*1e-7)
    assert (r['surface_m3s'] >= 0).all() and (r['coast_m3s'] >= 0).all()


def test_isolated_cell_analytic_free_face_and_seepage():
    # Four half-cell coastal faces: Q = 4 K h^2. R A = 1e-4 m3/s.
    r = solve([[10.]])
    closed(r)
    assert r['head_m'][0, 0] == pytest.approx(np.sqrt(1e-4/(4e-5)), rel=1e-10)
    assert r['surface_m3s'][0, 0] == 0
    wet = solve([[1.]])
    closed(wet)
    assert wet['head_m'][0, 0] == 1
    assert wet['surface_m3s'][0, 0] == pytest.approx(6e-5)
    assert wet['coast_m3s'][0, 0] == pytest.approx(4e-5)


def test_zero_recharge_and_impermeable_surface():
    dry = solve(np.full((7, 7), 10.), recharge=0.)
    closed(dry)
    assert not np.any(dry['surface_m3s']+dry['coast_m3s'])
    sealed = solve(np.full((7, 7), 10.), conductivity=0.)
    closed(sealed)
    np.testing.assert_allclose(sealed['surface_m3s'], 1e-4)
    assert not np.any(sealed['coast_m3s'])


def test_symmetric_mound_and_recharge_response():
    top = np.full((9, 9), 50.)
    a, b = solve(top), solve(top, recharge=4e-8)
    closed(a)
    closed(b)
    assert a['head_m'][4, 4] > a['head_m'][0, 0]
    np.testing.assert_allclose(a['head_m'], a['head_m'][::-1, ::-1], atol=2e-6)
    # Flat-bed Dupuit equation scales quadratically with saturated thickness.
    np.testing.assert_allclose(b['head_m'], 2*a['head_m'], rtol=2e-7)
    assert not np.any(a['surface_m3s'])


def test_valley_appears_as_seep_without_preassigned_river():
    top = np.full((9, 9), 50.)
    top[4, 4] = 1.
    r = solve(top)
    closed(r)
    assert r['surface_m3s'][4, 4] > 0
    assert np.count_nonzero(r['surface_m3s']) == 1


def test_failed_convergence_is_not_normalized_to_pass():
    r = solve(np.full((15, 15), 50.), max_iterations=1)
    assert not r['converged']
    assert np.abs(r['residual_m3s']).sum() > r['recharge_m3s']*1e-7


def test_vertical_datum_does_not_change_discharge():
    a, b = solve(np.full((5, 5), 10.)), solve(np.full((5, 5), 1010.), bottom=1000.)
    closed(a)
    closed(b)
    np.testing.assert_allclose(a['head_m']+1000, b['head_m'], atol=1e-8)
    np.testing.assert_allclose(a['coast_m3s'], b['coast_m3s'], rtol=1e-8)


@pytest.mark.parametrize('kw', [dict(conductivity=-1.), dict(recharge=-1.), dict(bottom=20.)])
def test_invalid_inputs(kw):
    with pytest.raises(ValueError):
        solve([[10.]], **kw)
