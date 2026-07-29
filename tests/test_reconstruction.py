import numpy as np
import pymeshlab
import pytest

import mojo_poissonrecon as mpr


def sphere_points(count=600, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, count)
    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = np.sqrt(1.0 - z * z)
    points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
    return np.ascontiguousarray(points)


def mesh_volume(vertices, faces):
    return abs(
        np.einsum(
            "ij,ij->i",
            vertices[faces[:, 0]],
            np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
        ).sum()
        / 6.0
    )


def test_full_reconstruction_parity_with_pymeshlab_poissonrecon():
    points = sphere_points()
    actual = mpr.reconstruct(points, points, depth=4, cycles=4)

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=points, v_normals_matrix=points))
    mesh_set.generate_surface_reconstruction_screened_poisson(
        depth=4,
        fulldepth=2,
        cgdepth=0,
        scale=1.1,
        samplespernode=1.5,
        pointweight=4.0,
        iters=8,
        threads=1,
    )
    reference = mesh_set.current_mesh()
    reference_vertices = reference.vertex_matrix()
    reference_faces = reference.face_matrix()

    actual_radius = np.linalg.norm(actual.vertices, axis=1)
    reference_radius = np.linalg.norm(reference_vertices, axis=1)
    assert abs(np.median(actual_radius) - np.median(reference_radius)) < 0.015
    assert abs(actual_radius.std() - reference_radius.std()) < 0.015
    assert mesh_volume(actual.vertices, actual.faces) == pytest.approx(
        mesh_volume(reference_vertices, reference_faces), rel=0.06
    )
    distances = np.linalg.norm(
        actual.vertices[:, None, :] - reference_vertices[None, :, :], axis=2
    )
    diagonal = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    symmetric_chamfer = 0.5 * (
        distances.min(axis=0).mean() + distances.min(axis=1).mean()
    )
    assert symmetric_chamfer / diagonal < 0.025


def test_sphere_is_watertight_symmetric_and_volume_preserving():
    points = sphere_points(1200, seed=2)
    mesh = mpr.reconstruct(points, points, depth=5, cycles=5)
    edges = np.sort(
        np.concatenate(
            [
                mesh.faces[:, [0, 1]],
                mesh.faces[:, [1, 2]],
                mesh.faces[:, [2, 0]],
            ]
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    assert extents.max() / extents.min() < 1.04
    assert mesh_volume(mesh.vertices, mesh.faces) == pytest.approx(4 * np.pi / 3, rel=0.04)
    assert mesh.residual_norm / np.linalg.norm(points) < 0.2


EMPTY = (np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), 0)
SINGLE_TRIANGLE = (
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    np.array([[0, 1, 2]]),
    3,
)
DUPLICATE_VERTICES = (
    np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    ),
    np.array([[0, 1, 2], [3, 1, 2]]),
    4,
)
ZERO_AREA = (
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    np.array([[0, 1, 2]]),
    0,
)
NON_MANIFOLD = (
    np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 0.0, 1.0],
        ]
    ),
    np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]),
    5,
)
UNREFERENCED = (
    np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]]
    ),
    np.array([[0, 1, 2]]),
    3,
)


@pytest.mark.parametrize(
    "vertices,faces,expected",
    [EMPTY, SINGLE_TRIANGLE, DUPLICATE_VERTICES, ZERO_AREA, NON_MANIFOLD, UNREFERENCED],
    ids=[
        "empty-mesh",
        "single-triangle",
        "duplicate-vertices",
        "zero-area-face",
        "non-manifold-edge",
        "unreferenced-vertex",
    ],
)
def test_degenerate_mesh_preprocessing(vertices, faces, expected):
    points, normals = mpr.oriented_points_from_mesh(vertices, faces)
    assert len(points) == expected
    assert points.shape == normals.shape
    assert np.isfinite(normals).all()
    if expected:
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_empty_and_zero_area_mesh_reconstruct_to_empty():
    assert mpr.reconstruct_mesh(EMPTY[0], EMPTY[1]).is_empty
    assert mpr.reconstruct_mesh(ZERO_AREA[0], ZERO_AREA[1]).is_empty


def test_reconstruct_mesh_nonempty_path():
    # A small, consistently oriented octahedron proves the public mesh wrapper
    # reaches reconstruction rather than only covering its empty fast path.
    vertices = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
        dtype=np.int64,
    )
    mesh = mpr.reconstruct_mesh(vertices, faces, depth=3, cycles=2)
    assert not mesh.is_empty
    assert mesh.faces.shape[1] == 3


def test_mesh_normals_are_area_weighted():
    vertices = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int64)
    points, normals = mpr.oriented_points_from_mesh(vertices, faces)
    origin = np.flatnonzero(np.all(points == vertices[0], axis=1))[0]
    # Unnormalized contributions are (0, 0, 4) and (0, 2, 0).
    assert np.allclose(normals[origin], [0.0, 1 / np.sqrt(5), 2 / np.sqrt(5)])
