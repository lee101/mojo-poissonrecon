"""NumPy translations of upstream kernels used as fine-grained parity oracles."""

from __future__ import annotations

import numpy as np


def octree_codes(points: np.ndarray, depth: int) -> np.ndarray:
    result = np.zeros((len(points), depth + 1), dtype=np.int64)
    for i, point in enumerate(points):
        code = 0
        for d in range(depth):
            scale = 1 << (d + 1)
            xyz = np.clip((point * scale).astype(np.int64), 0, scale - 1)
            child = int(xyz[0] & 1) | int((xyz[1] & 1) << 1) | int((xyz[2] & 1) << 2)
            code = (code << 3) | child
            result[i, d + 1] = code
    return result


def splat(
    points: np.ndarray,
    normals: np.ndarray,
    n: int,
    confidence: bool,
    samples_per_node: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (n, n, n)
    density, vx, vy, vz, screen = (np.zeros(shape) for _ in range(5))
    resolution = n - 1
    for point, normal in zip(points, normals):
        length = np.linalg.norm(normal)
        if length <= 0:
            continue
        sample_weight = length if confidence else 1.0
        normal = normal / length
        grid = point * resolution
        base = np.clip(grid.astype(np.int64), 0, resolution - 1)
        frac = np.clip(grid - base, 0.0, 1.0)
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    weight = (
                        (frac[0] if dx else 1 - frac[0])
                        * (frac[1] if dy else 1 - frac[1])
                        * (frac[2] if dz else 1 - frac[2])
                        * sample_weight
                    )
                    index = (base[2] + dz, base[1] + dy, base[0] + dx)
                    density[index] += 2 * weight
                    screen[index] += weight
    for point, normal in zip(points, normals):
        length = np.linalg.norm(normal)
        if length <= 0:
            continue
        sample_weight = length if confidence else 1.0
        normal = normal / length
        grid = point * resolution
        base = np.clip(grid.astype(np.int64), 0, resolution - 1)
        frac = np.clip(grid - base, 0.0, 1.0)
        local_density = 0.0
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    basis = (
                        (frac[0] if dx else 1 - frac[0])
                        * (frac[1] if dy else 1 - frac[1])
                        * (frac[2] if dz else 1 - frac[2])
                    )
                    local_density += (
                        density[base[2] + dz, base[1] + dy, base[0] + dx] * basis
                    )
        density_factor = max(1.0, local_density / samples_per_node)
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    weight = (
                        (frac[0] if dx else 1 - frac[0])
                        * (frac[1] if dy else 1 - frac[1])
                        * (frac[2] if dz else 1 - frac[2])
                        * sample_weight
                    )
                    index = (base[2] + dz, base[1] + dy, base[0] + dx)
                    vx[index] -= normal[0] * weight * resolution / density_factor
                    vy[index] -= normal[1] * weight * resolution / density_factor
                    vz[index] -= normal[2] * weight * resolution / density_factor
    return density, vx, vy, vz, screen


def assemble(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    screen: np.ndarray,
    point_weight: float,
    target: float = 0.5,
) -> np.ndarray:
    n = vx.shape[0]
    h_inv = n - 1
    result = np.empty_like(vx)
    for z in range(n):
        for y in range(n):
            for x in range(n):
                if x == 0:
                    dx = (vx[z, y, 1] - vx[z, y, 0]) * h_inv
                elif x == n - 1:
                    dx = (vx[z, y, x] - vx[z, y, x - 1]) * h_inv
                else:
                    dx = (vx[z, y, x + 1] - vx[z, y, x - 1]) * 0.5 * h_inv
                if y == 0:
                    dy = (vy[z, 1, x] - vy[z, 0, x]) * h_inv
                elif y == n - 1:
                    dy = (vy[z, y, x] - vy[z, y - 1, x]) * h_inv
                else:
                    dy = (vy[z, y + 1, x] - vy[z, y - 1, x]) * 0.5 * h_inv
                if z == 0:
                    dz = (vz[1, y, x] - vz[0, y, x]) * h_inv
                elif z == n - 1:
                    dz = (vz[z, y, x] - vz[z - 1, y, x]) * h_inv
                else:
                    dz = (vz[z + 1, y, x] - vz[z - 1, y, x]) * 0.5 * h_inv
                screening = point_weight * screen[z, y, x] * h_inv * h_inv
                result[z, y, x] = -(dx + dy + dz) + screening * target
    return result


def restrict(values: np.ndarray) -> np.ndarray:
    fine_n = values.shape[0]
    coarse_n = (fine_n + 1) // 2
    result = np.empty((coarse_n,) * 3)
    for z in range(coarse_n):
        for y in range(coarse_n):
            for x in range(coarse_n):
                total = 0.0
                norm = 0.0
                for oz in range(-1, 2):
                    zz = 2 * z + oz
                    if not 0 <= zz < fine_n:
                        continue
                    wz = 2 if oz == 0 else 1
                    for oy in range(-1, 2):
                        yy = 2 * y + oy
                        if not 0 <= yy < fine_n:
                            continue
                        wy = 2 if oy == 0 else 1
                        for ox in range(-1, 2):
                            xx = 2 * x + ox
                            if not 0 <= xx < fine_n:
                                continue
                            wx = 2 if ox == 0 else 1
                            weight = wx * wy * wz
                            total += values[zz, yy, xx] * weight
                            norm += weight
                result[z, y, x] = total / norm
    return result


def dual_isosurface(
    field: np.ndarray, iso: float
) -> tuple[np.ndarray, np.ndarray]:
    cells = field.shape[0] - 1
    cell_map = np.full((cells, cells, cells), -1, dtype=np.int64)
    vertices: list[list[float]] = []
    edges = (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    corners = np.array(
        [[x, y, z] for z in range(2) for y in range(2) for x in range(2)],
        dtype=np.float64,
    )
    for z in range(cells):
        for y in range(cells):
            for x in range(cells):
                values = np.array(
                    [
                        field[z + int(c[2]), y + int(c[1]), x + int(c[0])]
                        for c in corners
                    ]
                )
                if np.all(values < iso) or np.all(values >= iso):
                    continue
                roots = []
                for a, b in edges:
                    if (values[a] < iso) != (values[b] < iso):
                        t = np.clip((iso - values[a]) / (values[b] - values[a]), 0, 1)
                        roots.append(corners[a] + (corners[b] - corners[a]) * t)
                vertex = (np.array([x, y, z]) + np.mean(roots, axis=0)) / cells
                cell_map[z, y, x] = len(vertices)
                vertices.append(vertex.tolist())

    faces: list[list[int]] = []

    def quad(indices, reverse):
        a, b, c, d = indices
        if min(indices) < 0:
            return
        faces.extend([[a, d, c], [a, c, b]] if reverse else [[a, b, c], [a, c, d]])

    for z in range(1, cells):
        for y in range(1, cells):
            for x in range(cells):
                if (field[z, y, x] < iso) != (field[z, y, x + 1] < iso):
                    quad(
                        [
                            cell_map[z - 1, y - 1, x],
                            cell_map[z - 1, y, x],
                            cell_map[z, y, x],
                            cell_map[z, y - 1, x],
                        ],
                        field[z, y, x] < iso,
                    )
    for z in range(1, cells):
        for y in range(cells):
            for x in range(1, cells):
                if (field[z, y, x] < iso) != (field[z, y + 1, x] < iso):
                    quad(
                        [
                            cell_map[z - 1, y, x - 1],
                            cell_map[z, y, x - 1],
                            cell_map[z, y, x],
                            cell_map[z - 1, y, x],
                        ],
                        field[z, y, x] < iso,
                    )
    for z in range(cells):
        for y in range(1, cells):
            for x in range(1, cells):
                if (field[z, y, x] < iso) != (field[z + 1, y, x] < iso):
                    quad(
                        [
                            cell_map[z, y - 1, x - 1],
                            cell_map[z, y - 1, x],
                            cell_map[z, y, x],
                            cell_map[z, y, x - 1],
                        ],
                        field[z, y, x] < iso,
                    )
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64).reshape(-1, 3)
