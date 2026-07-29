"""Screened Poisson reconstruction with Mojo compute kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._lib import addr, f64, i64, lib


@dataclass(frozen=True)
class FlatOctree:
    depth: np.ndarray
    morton: np.ndarray
    parent: np.ndarray
    child_start: np.ndarray
    child_count: np.ndarray
    point_leaf: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.depth.size)


@dataclass(frozen=True)
class GridSolution:
    field: np.ndarray
    density: np.ndarray
    screen: np.ndarray
    rhs: np.ndarray
    iso_value: float
    residual_norm: float
    octree: FlatOctree


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    density: np.ndarray
    iso_value: float = 0.0
    residual_norm: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.vertices.shape[0] == 0


def _points(values) -> np.ndarray:
    result = f64(values)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    return result


def _integer_parameter(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _finite_parameter(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _valid_oriented_rows(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    finite = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    # Scale before squaring so large but finite normals do not overflow.
    scale = np.max(np.abs(normals), axis=1)
    nonzero = scale > 0
    scaled_norm2 = np.zeros(len(normals), dtype=np.float64)
    rows = finite & nonzero
    scaled_norm2[rows] = np.einsum(
        "ij,ij->i", normals[rows] / scale[rows, None], normals[rows] / scale[rows, None]
    )
    return rows & np.isfinite(scaled_norm2) & (scaled_norm2 > 0)


def build_octree(points, depth: int = 6) -> FlatOctree:
    """Build a compact breadth-first Morton octree over unit-cube points."""
    points = _points(points)
    depth = _integer_parameter("depth", depth)
    if not 0 <= depth <= 19:
        raise ValueError("depth must be between 0 and 19")
    if len(points) and (not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1)):
        raise ValueError("octree points must be finite and inside [0, 1]^3")
    if not len(points):
        one = np.array([0], dtype=np.int64)
        zero = np.array([0], dtype=np.int64)
        return FlatOctree(zero, one, np.array([-1], np.int64), np.array([-1], np.int64), zero, np.empty(0, np.int64))

    codes = np.empty((len(points), depth + 1), dtype=np.int64)
    lib().mpr_octree_codes_f64(addr(points), len(points), depth, addr(codes))
    level_codes: list[np.ndarray] = []
    inverse: np.ndarray | None = None
    for d in range(depth + 1):
        unique, inv = np.unique(codes[:, d], return_inverse=True)
        level_codes.append(unique)
        if d == depth:
            inverse = inv.astype(np.int64, copy=False)

    offsets = np.cumsum([0] + [len(values) for values in level_codes], dtype=np.int64)
    morton = np.concatenate(level_codes)
    depths = np.concatenate(
        [np.full(len(values), d, dtype=np.int64) for d, values in enumerate(level_codes)]
    )
    parent = np.full(len(morton), -1, dtype=np.int64)
    child_start = np.full(len(morton), -1, dtype=np.int64)
    child_count = np.zeros(len(morton), dtype=np.int64)
    for d in range(1, depth + 1):
        previous = {int(code): int(offsets[d - 1] + i) for i, code in enumerate(level_codes[d - 1])}
        for local, code in enumerate(level_codes[d]):
            node = int(offsets[d] + local)
            p = previous[int(code) >> 3]
            parent[node] = p
            if child_start[p] < 0:
                child_start[p] = node
            child_count[p] += 1
    assert inverse is not None
    return FlatOctree(depths, morton, parent, child_start, child_count, inverse + offsets[-2])


def _restrict(values: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    coarse_n = (values.shape[0] + 1) // 2
    coarse = (
        np.empty((coarse_n, coarse_n, coarse_n), dtype=np.float64)
        if target is None
        else target
    )
    lib().mpr_restrict_f64(addr(values), addr(coarse), values.shape[0])
    return coarse


def _v_cycle(
    level: int,
    solution: list[np.ndarray],
    rhs: list[np.ndarray],
    screen: list[np.ndarray],
    residual: list[np.ndarray],
    point_weight: float,
    pre_smooth: int,
    post_smooth: int,
) -> None:
    n = solution[level].shape[0]
    if n <= 3:
        lib().mpr_relax_f64(
            addr(solution[level]), addr(rhs[level]), addr(screen[level]), n, point_weight, 40
        )
        return
    lib().mpr_relax_f64(
        addr(solution[level]),
        addr(rhs[level]),
        addr(screen[level]),
        n,
        point_weight,
        pre_smooth,
    )
    lib().mpr_residual_f64(
        addr(solution[level]),
        addr(rhs[level]),
        addr(screen[level]),
        addr(residual[level]),
        n,
        point_weight,
    )
    _restrict(residual[level], rhs[level + 1])
    solution[level + 1].fill(0.0)
    _v_cycle(
        level + 1,
        solution,
        rhs,
        screen,
        residual,
        point_weight,
        pre_smooth,
        post_smooth,
    )
    lib().mpr_prolong_add_f64(addr(solution[level + 1]), addr(solution[level]), n)
    lib().mpr_relax_f64(
        addr(solution[level]),
        addr(rhs[level]),
        addr(screen[level]),
        n,
        point_weight,
        post_smooth,
    )


def solve_screened_poisson(
    points,
    normals,
    *,
    depth: int = 6,
    point_weight: float = 4.0,
    confidence: bool = False,
    cycles: int = 4,
    samples_per_node: float = 1.5,
) -> GridSolution:
    """Build the octree, splat samples, and solve the screened Poisson system."""
    points = _points(points)
    normals = _points(normals)
    if points.shape != normals.shape:
        raise ValueError("points and normals must have the same shape")
    depth = _integer_parameter("depth", depth)
    cycles = _integer_parameter("cycles", cycles)
    point_weight = _finite_parameter("point_weight", point_weight)
    samples_per_node = _finite_parameter("samples_per_node", samples_per_node)
    if not 2 <= depth <= 8:
        raise ValueError("depth must be between 2 and 8")
    if cycles < 1:
        raise ValueError("cycles must be positive")
    if point_weight < 0 or samples_per_node <= 0:
        raise ValueError("point_weight must be non-negative and samples_per_node positive")

    valid = _valid_oriented_rows(points, normals)
    points = points[valid]
    normals = normals[valid]
    if len(points) and (np.any(points < 0) or np.any(points > 1)):
        raise ValueError("solver points must be inside [0, 1]^3")

    tree = build_octree(points, depth)
    n = (1 << depth) + 1
    shape = (n, n, n)
    density = np.zeros(shape, dtype=np.float64)
    vx = np.zeros(shape, dtype=np.float64)
    vy = np.zeros(shape, dtype=np.float64)
    vz = np.zeros(shape, dtype=np.float64)
    screen_fine = np.zeros(shape, dtype=np.float64)
    rhs_fine = np.zeros(shape, dtype=np.float64)
    if len(points):
        accepted = lib().mpr_splat_f64(
            addr(points),
            addr(normals),
            len(points),
            n,
            int(confidence),
            addr(density),
            addr(vx),
            addr(vy),
            addr(vz),
            addr(screen_fine),
            samples_per_node,
        )
        if accepted != len(points):
            raise RuntimeError(
                f"Mojo splat accepted {accepted} of {len(points)} validated samples"
            )
        density /= samples_per_node
        lib().mpr_assemble_f64(
            addr(vx),
            addr(vy),
            addr(vz),
            addr(screen_fine),
            addr(rhs_fine),
            n,
            point_weight,
            0.5,
        )

    screens = [screen_fine]
    while screens[-1].shape[0] > 3:
        screens.append(_restrict(screens[-1]) * 4.0)
    solutions = [np.full_like(screen, 0.5) for screen in screens]
    right_sides = [np.zeros_like(screen) for screen in screens]
    residuals = [np.empty_like(screen) for screen in screens]
    right_sides[0][...] = rhs_fine
    if len(points):
        for _ in range(cycles):
            _v_cycle(
                0,
                solutions,
                right_sides,
                screens,
                residuals,
                point_weight,
                3,
                3,
            )
    else:
        solutions[0].fill(0.0)

    residual_norm = lib().mpr_residual_f64(
        addr(solutions[0]),
        addr(rhs_fine),
        addr(screen_fine),
        addr(residuals[0]),
        n,
        point_weight,
    )
    iso = (
        lib().mpr_iso_value_f64(
            addr(solutions[0]), addr(points), addr(normals), len(points), n, int(confidence)
        )
        if len(points)
        else 0.0
    )
    return GridSolution(
        solutions[0], density, screen_fine, rhs_fine, float(iso), float(residual_norm), tree
    )


def extract_isosurface(
    field,
    *,
    iso_value: float = 0.0,
    density=None,
) -> Mesh:
    """Extract a triangle mesh from a cubic scalar grid on the dual grid."""
    field = f64(field)
    if field.ndim != 3 or len(set(field.shape)) != 1 or field.shape[0] < 2:
        raise ValueError("field must be a cubic array with side length >= 2")
    iso_value = _finite_parameter("iso_value", iso_value)
    if not np.isfinite(field).all():
        raise ValueError("field and iso_value must be finite")
    density = np.zeros_like(field) if density is None else f64(density)
    if density.shape != field.shape:
        raise ValueError("density must have the same shape as field")
    if not np.isfinite(density).all():
        raise ValueError("density must be finite")
    n = field.shape[0]
    dummy_i = np.zeros(1, dtype=np.int64)
    dummy_f = np.zeros(1, dtype=np.float64)
    count = lib().mpr_dual_vertices_f64(
        addr(field), addr(density), n, iso_value, addr(dummy_i), addr(dummy_f), addr(dummy_f), 0
    )
    max_vertices = (n - 1) ** 3
    if count < 0 or count > max_vertices:
        raise RuntimeError(f"invalid dual vertex count returned by Mojo: {count}")
    if count == 0:
        return Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64), np.empty(0), iso_value)
    vertices = np.empty((count, 3), dtype=np.float64)
    vertex_density = np.empty(count, dtype=np.float64)
    cell_map = np.empty((n - 1) ** 3, dtype=np.int64)
    written = lib().mpr_dual_vertices_f64(
        addr(field),
        addr(density),
        n,
        iso_value,
        addr(cell_map),
        addr(vertices),
        addr(vertex_density),
        count,
    )
    if written != count:
        raise RuntimeError("dual vertex count changed between extraction passes")
    face_count = lib().mpr_dual_faces_f64(
        addr(field), n, iso_value, addr(cell_map), addr(dummy_i), 0
    )
    max_faces = 6 * (n - 1) ** 3
    if face_count < 0 or face_count > max_faces:
        raise RuntimeError(f"invalid dual face count returned by Mojo: {face_count}")
    faces = np.empty((face_count, 3), dtype=np.int64)
    if face_count:
        written_faces = lib().mpr_dual_faces_f64(
            addr(field), n, iso_value, addr(cell_map), addr(faces), face_count
        )
        if written_faces != face_count:
            raise RuntimeError("dual face count changed between extraction passes")
    return Mesh(vertices, faces, vertex_density, iso_value)


def _normalize(points: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    width = float(np.max(maximum - minimum)) * scale
    if not np.isfinite(width) or width <= 0:
        width = 1.0
    origin = center - 0.5 * width
    return np.ascontiguousarray((points - origin) / width), origin, width


def reconstruct(
    points,
    normals,
    *,
    depth: int = 6,
    point_weight: float = 4.0,
    confidence: bool = False,
    cycles: int = 4,
    scale: float = 1.1,
    samples_per_node: float = 1.5,
) -> Mesh:
    """Reconstruct a watertight surface from oriented points."""
    points = _points(points)
    normals = _points(normals)
    if points.shape != normals.shape:
        raise ValueError("points and normals must have the same shape")
    scale = _finite_parameter("scale", scale)
    if scale <= 1.0:
        raise ValueError("scale must be greater than 1")
    valid = _valid_oriented_rows(points, normals)
    points = points[valid]
    normals = normals[valid]
    if not len(points):
        return Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64), np.empty(0))
    unit_points, origin, width = _normalize(points, scale)
    solution = solve_screened_poisson(
        unit_points,
        normals,
        depth=depth,
        point_weight=point_weight,
        confidence=confidence,
        cycles=cycles,
        samples_per_node=samples_per_node,
    )
    mesh = extract_isosurface(
        solution.field, iso_value=solution.iso_value, density=solution.density
    )
    vertices = np.ascontiguousarray(mesh.vertices * width + origin)
    return Mesh(vertices, mesh.faces, mesh.density, solution.iso_value, solution.residual_norm)


def oriented_points_from_mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    """Return referenced vertices and area-weighted normals from a triangle mesh."""
    vertices = _points(vertices)
    faces = i64(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (m, 3)")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("face index out of range")
    normals = np.zeros_like(vertices)
    referenced = np.zeros(len(vertices), dtype=bool)
    for face in faces:
        a, b, c = map(int, face)
        cross = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        if np.isfinite(cross).all() and np.dot(cross, cross) > 0:
            normals[a] += cross
            normals[b] += cross
            normals[c] += cross
            referenced[[a, b, c]] = True
    lengths = np.linalg.norm(normals, axis=1)
    keep = referenced & (lengths > 0) & np.isfinite(lengths)
    normals[keep] /= lengths[keep, None]
    return np.ascontiguousarray(vertices[keep]), np.ascontiguousarray(normals[keep])


def reconstruct_mesh(vertices, faces, **kwargs) -> Mesh:
    points, normals = oriented_points_from_mesh(vertices, faces)
    return reconstruct(points, normals, **kwargs)
