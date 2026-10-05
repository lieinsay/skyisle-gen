"""Exact surviving strata, independent of any hydrologic parameter choice."""
import numpy as np
import pytest

from test_aquifer_balance import core


def profile(surface=120., top=100., base=0., **kw):
    s = dict(t_cap=10., t_sed=60., t_gab=30., bed_lime=15., bed_marl=10.,
             bed_phase=0., scale=1.)
    s.update(kw)
    return np.asarray(core.strat_column(surface, top, base, s))


def test_exact_interfaces_include_eroded_cap_and_skeleton_truncation():
    p = profile()
    np.testing.assert_array_equal(p, [[0, 10, 4], [10, 40, 3], [40, 50, 2],
                                     [50, 65, 1], [65, 75, 2], [75, 120, 1]])
    np.testing.assert_array_equal(profile(surface=72, base=42),
                                  [[42, 50, 2], [50, 65, 1], [65, 72, 2]])
    assert profile(surface=42, base=42).size == 0


def test_phase_changes_contacts_without_gaps_or_added_rock():
    p = profile(surface=90, base=40, bed_phase=8.)
    np.testing.assert_array_equal(p, [[40, 48, 1], [48, 58, 2], [58, 73, 1],
                                     [73, 83, 2], [83, 90, 1]])
    assert np.diff(p[:, :2], axis=1).sum() == 50
    np.testing.assert_array_equal(p[:-1, 1], p[1:, 0])


def test_affine_elevation_fit_scales_every_contact():
    p = profile()
    q = profile(surface=340, top=300, base=100, scale=2.)
    np.testing.assert_array_equal(q[:, :2], 100+2*p[:, :2])
    np.testing.assert_array_equal(q[:, 2], p[:, 2])


@pytest.mark.parametrize('kw', [dict(scale=0), dict(bed_lime=0), dict(bed_marl=-1),
                                dict(t_sed=5), dict(bed_phase=np.nan)])
def test_invalid_geology_is_rejected(kw):
    with pytest.raises(ValueError):
        profile(**kw)
