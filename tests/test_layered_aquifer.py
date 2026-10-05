"""Independent balances, analytic cases and perched/connected layer behavior."""
import numpy as np
import pytest

from test_aquifer_balance import core


STRATA = dict(t_cap=1., t_sed=1., t_gab=1., bed_lime=1., bed_marl=1., bed_phase=0., scale=1.)


def model(shape=(1, 1), surface=2., base=0., structural=2., strata=STRATA,
          kh=(0., 0., 1e-4, 0., 0.), kv=(1e-4, 1e-4, 1e-4, 1e-4, 0.), shift=0.):
    grid = lambda v: np.broadcast_to(v, shape).copy().astype(float)
    return core.make_layered_aquifer(grid(surface)+shift, grid(structural)+shift, grid(base)+shift,
        np.ones(shape, dtype=bool), strata, list(kh), list(kv), [.1]*5, [1e-5]*5, 10.)


def closed(r, dt=None):
    assert r['converged']
    change = r['total_storage_change_m3']/dt if dt else 0.
    assert r['input_m3s'] == pytest.approx(r['output_m3s']+change, abs=2e-10)
    assert r['absolute_residual_m3s'] < max(1e-12, max(r['input_m3s'], r['output_m3s'])*1.001e-7)


def test_single_unconfined_cell_matches_radial_face_balance():
    a = model(strata=None)
    r = a.solve(np.array([[1e-7]]))
    closed(r)
    # Four half-cell coastal faces: q = 4*K*h^2.
    assert r['head_m'][0] == pytest.approx(np.sqrt(1e-5/(4e-4)), rel=1e-6)


def test_dewatered_lower_layer_does_not_pull_with_unsaturated_head_drop():
    a = model()
    r = a.solve(np.array([[1e-7]]))
    closed(r)
    lower, upper = r['head_m']
    assert lower == pytest.approx(np.sqrt(1e-5/(4e-4)), rel=1e-6)
    # C = area/(half-thickness/K above + half-thickness/K below) = .01.
    assert upper == pytest.approx(1+1e-5/.01, abs=2e-9)
    assert r['surface_m3s'][0, 0] == 0


def test_low_vertical_conductivity_retains_water_and_forms_surface_spring():
    a = model(kv=(1e-4, 1e-4, 1e-10, 1e-4, 0.))
    r = a.solve(np.array([[1e-7]]))
    closed(r)
    leak = 100/(.5/1e-4+.5/1e-10)  # upper head=2, perched reference=1
    assert r['head_m'][1] == 2
    assert r['surface_m3s'][0, 0] == pytest.approx(1e-5-leak, rel=1e-6)
    assert r['coast_m3s'][0, 0] == pytest.approx(leak, rel=1e-5)


def test_no_input_and_no_water_cannot_make_springs_or_fill_layers():
    a = model(kh=(0.,)*5, kv=(0.,)*5)
    dry = a.solve(np.zeros((1, 1)))
    closed(dry)
    np.testing.assert_array_equal(dry['head_m'], a.node_bottoms)
    after = a.step(np.zeros((1, 1)), dry['head_m'], 86400.)
    closed(after, 86400.)
    assert after['output_m3s'] == after['total_storage_change_m3'] == 0


def test_transient_budget_includes_confined_storage_and_upward_exchange():
    a = model()
    head = np.array([1.8, 1.7])  # lower layer artesian, upper one unconfined
    r = a.step(np.zeros((1, 1)), head, 100.)
    closed(r, 100.)
    assert r['total_storage_change_m3'] < 0
    assert r['coast_m3s'][0, 0] > 0
    assert r['head_m'][0] < head[0]


def test_horizontal_layer_contacts_conserve_water_on_uneven_surfaces():
    top = np.array([[4., 4., 4.], [4., 1.5, 4.], [4., 4., 4.]])
    a = model(shape=top.shape, surface=top, structural=4.,
              strata=dict(STRATA, t_cap=1., t_sed=2., t_gab=2.),
              kh=(1e-4, 1e-9, 1e-5, 1e-5, 0.), kv=(1e-5, 1e-9, 1e-6, 1e-6, 0.))
    rain = np.zeros(top.shape)
    rain[1, 1] = 1e-7  # interior recharge can reach the coast only via internal faces
    r = a.solve(rain)
    closed(r)
    assert r['coast_m3s'].sum() > 0
    assert r['coast_m3s'][1, 1] == 0
    for _ in range(3):
        r = a.step(np.zeros(top.shape), r['head_m'], 86400.)
        closed(r, 86400.)
    assert np.all(r['head_m'] >= np.array(a.node_bottoms))


def test_datum_shift_does_not_change_flow_or_stored_volume():
    a, b = model(), model(shift=1500.)
    r, s = [m.solve(np.array([[1e-7]])) for m in (a, b)]
    closed(r); closed(s)
    np.testing.assert_allclose(s['head_m']-1500., r['head_m'], atol=2e-8)
    np.testing.assert_allclose(s['coast_m3s'], r['coast_m3s'], atol=2e-12)


def test_unconverged_result_is_explicit_and_invalid_state_is_rejected():
    a = model()
    r = a.solve(np.array([[1e-7]]), max_iterations=1)
    assert not r['converged']
    with pytest.raises(ValueError):
        a.step(np.zeros((1, 1)), np.array([-1., 0.]), 86400.)
    with pytest.raises(ValueError):
        a.solve(np.array([[-1e-7]]))


def test_resumed_equilibrium_does_not_add_transient_storage_or_change_answer():
    a = model()
    rain = np.array([[1e-7]])
    expected = a.solve(rain)
    partial = a.solve(rain, max_iterations=1)
    resumed = a.solve(rain, initial_head=partial['head_m'])
    closed(resumed)
    np.testing.assert_allclose(resumed['head_m'], expected['head_m'], atol=2e-8)
    assert resumed['total_storage_change_m3'] == 0
    with pytest.raises(ValueError):
        a.solve(rain, initial_head=np.array([-1.]))


def test_exposed_slope_drains_to_lower_land_without_predefined_channel():
    # Non-overlapping rock columns: one of the high cell's four open faces
    # discharges onto the lower cell. The other three discharge off-island.
    a = model(shape=(1, 2), surface=[[2., 1.]], base=[[1., 0.]], strata=None)
    r = a.solve(np.array([[1e-7, 0.]]))
    closed(r)
    np.testing.assert_allclose(r['surface_m3s'], [[0., 2.5e-6]], atol=2e-12)
    np.testing.assert_allclose(r['coast_m3s'], [[7.5e-6, 0.]], atol=2e-12)
    assert r['head_m'][1] == 0  # slope discharge is not injected into this aquifer twice
    routed = core.route_surface_water(r['surface_m3s'], np.ones((1, 2), bool),
        np.array([1, -1], dtype=np.int64))
    assert routed['outlet_m3s'] == pytest.approx(2.5e-6)


def test_finite_shallow_horizon_preserves_rock_contacts_and_deep_resistance():
    grid = lambda v: np.array([[v]], dtype=float)
    deep, shallow = [1e-6]*5, [1e-4]*5
    a = core.make_layered_aquifer(grid(3), grid(3), grid(0), np.ones((1, 1), bool),
        None, deep, deep, [.1]*5, [1e-5]*5, 10., shallow_depth_m=1.,
        shallow_horizontal_ms=shallow, shallow_vertical_ms=shallow)
    assert a.node_bottoms == [0., 2.]
    assert a.node_tops == [2., 3.]
    r = a.solve(grid(1e-7))
    closed(r)
    lower, upper = r['head_m']
    assert lower < 2. < upper < 3.
    # Lower cell's four half-cell coast outlets equal the leakage it receives.
    leak = 100/(1/1e-6+.5/1e-4)*(upper-2.)
    assert 4e-6*lower**2 == pytest.approx(leak, abs=2e-12)
    assert 4e-4*(upper-2.)**2+leak == pytest.approx(1e-5, abs=2e-12)
    with pytest.raises(ValueError):
        core.make_layered_aquifer(grid(3), grid(3), grid(0), np.ones((1, 1), bool),
            None, deep, deep, [.1]*5, [1e-5]*5, 10., shallow_depth_m=1.)
