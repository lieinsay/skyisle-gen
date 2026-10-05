"""Transient analytic/volume checks, with explicit synthetic material fixtures."""
import numpy as np
import pytest

from test_aquifer_balance import core, solve


def step(top, head, recharge=0., conductivity=1e-5, specific_yield=.1,
         bottom=0., dt_s=86400., **kw):
    top = np.asarray(top, dtype=float)
    r = core.step_aquifer_balance(top, np.full_like(top, bottom),
        np.full_like(top, conductivity), np.full_like(top, recharge),
        np.ones(top.shape, dtype=bool), np.broadcast_to(head, top.shape).copy(),
        np.full_like(top, specific_yield), 100., dt_s, **kw)
    if r['converged']:
        tol = max(1e-12, max(r['recharge_m3s'], r['discharge_m3s'])*1e-7)
        assert np.abs(r['residual_m3s']).sum() <= tol
        assert abs((r['recharge_m3s']-r['discharge_m3s'])*dt_s
                   - r['total_storage_change_m3']) <= tol*dt_s*1.001
        assert np.all(r['head_m'] >= bottom) and np.all(r['head_m'] <= top)
    return r


def test_isolated_recession_matches_backward_euler_root():
    # 4 K h^2 + Sy A (h-h0)/dt = 0, four free coastal faces.
    h0, storage_rate, a = 5., .1*10000/86400, 4e-5
    h = 2*storage_rate*h0/(storage_rate+np.sqrt(storage_rate**2+4*a*storage_rate*h0))
    r = step([[20.]], h0)
    assert r['converged']
    assert r['head_m'][0, 0] == pytest.approx(h, rel=1e-10)
    assert r['discharge_m3s'] > 0 and r['recharge_m3s'] == 0
    assert r['total_storage_change_m3'] < 0


def test_recharge_fills_storage_before_surface_overflow():
    r = step([[1.]], 0., recharge=1e-7, conductivity=0., dt_s=100000.)
    assert r['head_m'][0, 0] == pytest.approx(.1)
    assert r['surface_m3s'][0, 0] == 0
    r = step([[1.]], .9, recharge=1e-6, conductivity=0., dt_s=100000.)
    assert r['head_m'][0, 0] == 1.
    assert r['total_storage_change_m3'] == pytest.approx(100.)
    assert r['surface_m3s'][0, 0] == pytest.approx(.009)


def test_dry_zero_head_does_not_create_water():
    r = step(np.full((5, 5), 10.), 0.)
    assert r['converged']
    assert r['discharge_m3s'] == r['total_storage_change_m3'] == 0


def test_annual_mean_seepage_does_not_imply_perennial_spring():
    # Annual mean above seepage threshold, but a wet pulse followed by a dry
    # period loses surface emergence. Classifying from annual mean would fail.
    mean = solve([[1.]], recharge=1e-8)
    assert mean['surface_m3s'][0, 0] > 0
    wet = step([[1.]], 0., recharge=1e-6, specific_yield=.01)
    assert wet['surface_m3s'][0, 0] > 0
    dry = step([[1.]], wet['head_m'], specific_yield=.01)
    assert dry['surface_m3s'][0, 0] == 0
    assert dry['head_m'][0, 0] < 1


def test_spatial_pulse_budget_and_timestep_refinement():
    top = np.full((5, 5), 10.)
    top[2, 2] = .5
    initial = solve(top)['head_m']

    def run(dt):
        head = initial.copy()
        inputs = outputs = 0.
        for day in range(12):
            for _ in range(round(86400/dt)):
                r = step(top, head, recharge=3e-8 if day < 4 else 0., dt_s=dt)
                assert r['converged']
                inputs += r['recharge_m3s']*dt
                outputs += r['discharge_m3s']*dt
                head = r['head_m']
        storage = .1*10000*np.sum(head-initial)
        assert abs(inputs-outputs-storage) < 1e-6*max(inputs, outputs)
        return head

    coarse, half, quarter = run(86400), run(43200), run(21600)
    assert np.linalg.norm(half-quarter) < np.linalg.norm(coarse-half)


def test_transient_datum_invariance():
    a = step([[10.]], 3., recharge=1e-8)
    b = step([[1010.]], 1003., recharge=1e-8, bottom=1000.)
    np.testing.assert_allclose(a['head_m']+1000, b['head_m'], atol=1e-8)
    assert a['total_storage_change_m3'] == pytest.approx(b['total_storage_change_m3'], rel=1e-8)


@pytest.mark.parametrize('kw', [dict(specific_yield=0.), dict(specific_yield=1.1),
    dict(dt_s=0.), dict(head=-1.), dict(head=20.)])
def test_invalid_storage_inputs(kw):
    args = dict(top=[[10.]], head=1.)
    args.update(kw)
    with pytest.raises(ValueError):
        step(**args)
