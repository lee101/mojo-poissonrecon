import numpy as np
import pytest

import mojo_poissonrecon as mpr
from mojo_poissonrecon import _reference
from mojo_poissonrecon._lib import addr, f64, i64, lib
from mojo_poissonrecon.core import _restrict


def test_octree_matches_upstream_child_index_translation():
    rng = np.random.default_rng(3)
    points = rng.random((200, 3))
    depth = 7
    expected = _reference.octree_codes(points, depth)
    tree = mpr.build_octree(points, depth)
    for d in range(depth + 1):
        assert np.array_equal(tree.morton[tree.depth == d], np.unique(expected[:, d]))
    assert np.array_equal(tree.morton[tree.point_leaf], expected[:, -1])
    assert np.array_equal(
        tree.morton[tree.parent[tree.depth > 0]], tree.morton[tree.depth > 0] >> 3
    )


def test_degree_one_density_screen_and_rhs_parity():
    points = np.array(
        [[0.15, 0.25, 0.35], [0.8, 0.7, 0.6], [1.0, 0.5, 0.0]], dtype=np.float64
    )
    normals = np.array([[2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 4.0]])
    solution = mpr.solve_screened_poisson(
        points,
        normals,
        depth=2,
        cycles=1,
        confidence=True,
        samples_per_node=1.0,
    )
    density, vx, vy, vz, screen = _reference.splat(
        points, normals, 5, True, samples_per_node=1.0
    )
    rhs = _reference.assemble(vx, vy, vz, screen, 4.0)
    assert np.allclose(solution.density, density, rtol=0, atol=2e-15)
    assert np.allclose(solution.screen, screen, rtol=0, atol=2e-15)
    assert np.allclose(solution.rhs, rhs, rtol=0, atol=2e-13)


def test_confidence_uses_normal_length_as_upstream_weight():
    points = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    plain = mpr.solve_screened_poisson(
        points, normals, depth=2, cycles=1, confidence=False, samples_per_node=1.0
    )
    weighted = mpr.solve_screened_poisson(
        points, normals, depth=2, cycles=1, confidence=True, samples_per_node=1.0
    )
    assert plain.density.sum() == 4.0
    assert weighted.density.sum() == 6.0
    assert plain.screen.sum() == 2.0
    assert weighted.screen.sum() == 3.0


def test_full_weighting_restriction_parity():
    rng = np.random.default_rng(11)
    fine = rng.normal(size=(9, 9, 9))
    assert np.allclose(_restrict(fine), _reference.restrict(fine), rtol=0, atol=2e-16)


def test_trilinear_prolongation_and_iso_value():
    coarse = np.arange(125, dtype=np.float64).reshape(5, 5, 5)
    fine = np.zeros((9, 9, 9), dtype=np.float64)
    lib().mpr_prolong_add_f64(addr(coarse), addr(fine), 9)
    assert np.array_equal(fine[::2, ::2, ::2], coarse)
    assert fine[1, 1, 1] == pytest.approx(coarse[:2, :2, :2].mean())

    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    normals = np.ones_like(points)
    iso = lib().mpr_iso_value_f64(
        addr(fine), addr(points), addr(normals), len(points), 9, 0
    )
    assert iso == pytest.approx(np.mean([fine[0, 0, 0], fine[4, 4, 4], fine[8, 8, 8]]))
    normals[1] *= 2.0
    weighted = lib().mpr_iso_value_f64(
        addr(fine), addr(points), addr(normals), len(points), 9, 1
    )
    weights = np.linalg.norm(normals, axis=1)
    expected = np.average(
        [fine[0, 0, 0], fine[4, 4, 4], fine[8, 8, 8]], weights=weights
    )
    assert weighted == pytest.approx(expected)


def test_ffi_conversions_reject_narrowing_and_bad_buffers():
    with pytest.raises(TypeError, match="complex"):
        f64(np.array([1 + 2j]))
    with pytest.raises(TypeError, match="real numbers"):
        f64(["1.0"])
    with pytest.raises(ValueError, match="exactly"):
        f64(np.array([2**53 + 1], dtype=np.int64))
    with pytest.raises(ValueError, match="finite integers"):
        i64(np.array([1.5]))
    with pytest.raises(TypeError, match="dtype"):
        addr(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.ones((3, 3), dtype=np.float64)[:, ::2])


def _stencil_sum_degree(values):
    neighbor_sum = np.zeros_like(values)
    degree = np.zeros(values.shape, dtype=np.int64)
    neighbor_sum[:, :, 1:] += values[:, :, :-1]
    degree[:, :, 1:] += 1
    neighbor_sum[:, :, :-1] += values[:, :, 1:]
    degree[:, :, :-1] += 1
    neighbor_sum[:, 1:, :] += values[:, :-1, :]
    degree[:, 1:, :] += 1
    neighbor_sum[:, :-1, :] += values[:, 1:, :]
    degree[:, :-1, :] += 1
    neighbor_sum[1:, :, :] += values[:-1, :, :]
    degree[1:, :, :] += 1
    neighbor_sum[:-1, :, :] += values[1:, :, :]
    degree[:-1, :, :] += 1
    return neighbor_sum, degree


@pytest.mark.parametrize("n", [9, 65], ids=["simd-tail-serial", "parallel-threshold"])
def test_residual_simd_tail_and_parallel_threshold(n):
    rng = np.random.default_rng(n)
    solution = rng.normal(size=(n, n, n))
    rhs = rng.normal(size=(n, n, n))
    screen = rng.random((n, n, n))
    actual = np.empty_like(solution)
    point_weight = 4.0
    neighbor_sum, degree = _stencil_sum_degree(solution)
    h_inv2 = float((n - 1) ** 2)
    expected = rhs - (
        (degree * h_inv2 + point_weight * screen * h_inv2) * solution
        - neighbor_sum * h_inv2
    )
    actual_norm = lib().mpr_residual_f64(
        addr(solution), addr(rhs), addr(screen), addr(actual), n, point_weight
    )
    assert np.allclose(actual, expected, rtol=2e-15, atol=3e-11)
    assert actual_norm == pytest.approx(np.linalg.norm(expected), rel=2e-15)


def test_parallel_red_black_relaxation_parity():
    n = 65
    rng = np.random.default_rng(71)
    solution = rng.normal(size=(n, n, n))
    expected = solution.copy()
    rhs = rng.normal(size=(n, n, n))
    screen = rng.random((n, n, n))
    point_weight = 4.0
    h_inv2 = float((n - 1) ** 2)
    parity = np.indices((n, n, n)).sum(axis=0) & 1
    for color in range(2):
        neighbor_sum, degree = _stencil_sum_degree(expected)
        diagonal = degree * h_inv2 + point_weight * screen * h_inv2
        mask = (parity == color) & (diagonal > 1.0e-30)
        expected[mask] = (
            rhs[mask] + neighbor_sum[mask] * h_inv2
        ) / diagonal[mask]
    lib().mpr_relax_f64(
        addr(solution), addr(rhs), addr(screen), n, point_weight, 1
    )
    assert np.allclose(solution, expected, rtol=0, atol=2e-15)


def test_dual_contour_matches_numpy_source_translation():
    axis = np.linspace(0.0, 1.0, 13)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    field = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - 0.3**2
    actual = mpr.extract_isosurface(field)
    vertices, faces = _reference.dual_isosurface(field, 0.0)
    assert np.allclose(actual.vertices, vertices, rtol=0, atol=2e-16)
    assert np.array_equal(actual.faces, faces)


def test_public_validation_rejects_unsafe_values():
    points = np.array([[0.25, 0.25, 0.25]])
    normals = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(TypeError, match="depth"):
        mpr.solve_screened_poisson(points, normals, depth=2.5)
    with pytest.raises(ValueError, match="finite"):
        mpr.solve_screened_poisson(points, normals, point_weight=np.nan)
    with pytest.raises(ValueError, match="density must be finite"):
        mpr.extract_isosurface(np.zeros((2, 2, 2)), density=np.full((2, 2, 2), np.nan))
    with pytest.raises(ValueError, match="greater than 1"):
        mpr.reconstruct(points, normals, scale=1.0)
