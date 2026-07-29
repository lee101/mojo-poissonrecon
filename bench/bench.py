"""Benchmarks against PyMeshLab/PoissonRecon and source-derived NumPy kernels."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from collections import defaultdict

import numpy as np
import pymeshlab

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_poissonrecon as mpr  # noqa: E402
from mojo_poissonrecon import _reference  # noqa: E402
from mojo_poissonrecon._lib import addr, lib  # noqa: E402


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def sphere_points(count, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, count)
    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = np.sqrt(1.0 - z * z)
    return np.ascontiguousarray(
        np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
    )


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf8") as source:
            for line in source:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def physical_cpu_count():
    cores = set()
    physical_id = None
    core_id = None
    try:
        with open("/proc/cpuinfo", encoding="utf8") as source:
            for line in source:
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                elif not line.strip() and physical_id is not None and core_id is not None:
                    cores.add((physical_id, core_id))
                    physical_id = None
                    core_id = None
    except OSError:
        pass
    return len(cores) or max(1, (os.cpu_count() or 1) // 2)


def morton_case():
    points = np.random.default_rng(1).random((50_000, 3))
    actual = np.empty((len(points), 9), dtype=np.int64)
    return (
        lambda: lib().mpr_octree_codes_f64(addr(points), len(points), 8, addr(actual)),
        lambda: _reference.octree_codes(points, 8),
        "NumPy source translation",
    )


def splat_case():
    points = np.random.default_rng(2).random((30_000, 3))
    normals = np.random.default_rng(3).normal(size=(30_000, 3))
    arrays = [np.zeros((33, 33, 33)) for _ in range(5)]

    def mojo():
        for array in arrays:
            array.fill(0)
        lib().mpr_splat_f64(
            addr(points),
            addr(normals),
            len(points),
            33,
            1,
            *(addr(array) for array in arrays),
            1.5,
        )

    return (
        mojo,
        lambda: _reference.splat(points, normals, 33, True),
        "NumPy source translation",
    )


def dual_case():
    axis = np.linspace(0.0, 1.0, 49)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    field = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - 0.3**2
    return (
        lambda: mpr.extract_isosurface(field),
        lambda: _reference.dual_isosurface(field, 0.0),
        "NumPy source translation",
    )


def reconstruction_case(point_count=2_000, depth=5):
    points = sphere_points(point_count, seed=4)
    thread_count = physical_cpu_count()

    def mojo():
        return mpr.reconstruct(points, points, depth=depth, cycles=4)

    def upstream():
        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=points, v_normals_matrix=points))
        mesh_set.generate_surface_reconstruction_screened_poisson(
            depth=depth,
            fulldepth=min(3, depth - 2),
            cgdepth=0,
            scale=1.1,
            samplespernode=1.5,
            pointweight=4.0,
            iters=8,
            threads=thread_count,
        )
        return mesh_set.current_mesh()

    return mojo, upstream, f"PyMeshLab PoissonRecon ({thread_count} threads)"


CASES = [
    ("Morton octree coding, 50k points depth 8", morton_case),
    ("Density/confidence splat, 30k points", splat_case),
    ("Dual contour, 49^3 scalar grid", dual_case),
    ("Full reconstruction, 2k points depth 5", reconstruction_case),
    (
        "Full reconstruction, 20k points depth 7",
        lambda: reconstruction_case(20_000, 7),
    ),
]


def profile_reconstruction():
    points = sphere_points(20_000, seed=4)
    native = lib()
    names = [
        "mpr_octree_codes_f64",
        "mpr_splat_f64",
        "mpr_assemble_f64",
        "mpr_relax_f64",
        "mpr_residual_f64",
        "mpr_restrict_f64",
        "mpr_prolong_add_f64",
        "mpr_iso_value_f64",
        "mpr_dual_vertices_f64",
        "mpr_dual_faces_f64",
    ]
    originals = {name: getattr(native, name) for name in names}
    totals = defaultdict(float)
    calls = defaultdict(int)

    def timed(name):
        function = originals[name]

        def wrapper(*args):
            start = time.perf_counter()
            result = function(*args)
            totals[name] += time.perf_counter() - start
            calls[name] += 1
            return result

        return wrapper

    for name in names:
        setattr(native, name, timed(name))
    try:
        start = time.perf_counter()
        mpr.reconstruct(points, points, depth=7, cycles=4)
        elapsed = time.perf_counter() - start
    finally:
        for name, function in originals.items():
            setattr(native, name, function)

    print(f"Depth-7 reconstruction profile: {elapsed * 1e3:.3f} ms")
    for name in sorted(totals, key=totals.get, reverse=True):
        print(f"{name}: {totals[name] * 1e3:.3f} ms ({calls[name]} calls)")


def main():
    if os.environ.get("MPR_PROFILE"):
        profile_reconstruction()
        return
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}; Python {platform.python_version()}")
    print()
    print("| Kernel | Mojo | Reference | Reference implementation | Speedup |")
    print("|---|---:|---:|---|---:|")
    for name, factory in CASES:
        mojo, reference, reference_name = factory()
        mojo()
        mojo_seconds = best_time(mojo)
        reference_seconds = best_time(reference)
        print(
            f"| {name} | {mojo_seconds * 1e3:.3f} ms | "
            f"{reference_seconds * 1e3:.3f} ms | {reference_name} | "
            f"{reference_seconds / mojo_seconds:.2f}x |"
        )


if __name__ == "__main__":
    main()
