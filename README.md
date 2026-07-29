# mojo-poissonrecon

`mojo-poissonrecon` is a compact Mojo implementation of the compute-bound path in
[Michael Kazhdan's PoissonRecon](https://github.com/mkazhdan/PoissonRecon).
It reconstructs a watertight triangle surface from oriented 3D points using a
screened Poisson solve and returns NumPy arrays through a small Python API.
It is not a drop-in port of the PoissonRecon executable and does not reproduce
the upstream adaptive solver exactly.

This is a derived work of PoissonRecon. The upstream project is MIT licensed;
the core FEM-tree headers additionally retain their 3-clause BSD notice.
This repository is MIT licensed, with upstream attribution in [NOTICE](NOTICE).
The port was traced against PoissonRecon commit
`262b0f539d404057d1f36e1adc07fc9388678899` and the MeshLab-bundled source at
commit `71e7b0b53cd98f520f84cb17b31bac6f1ef947cc`.

## Coverage

Covered and tested:

- breadth-first, structure-of-arrays Morton octree construction through depth 19;
- trilinear density splatting and sample-density output;
- upstream confidence semantics, where normal length becomes sample weight;
- oriented vector-field splatting and screened point constraints;
- a uniform-grid Neumann screened-Poisson assembly and red-black multigrid V-cycles;
- the upstream weighted sample iso-value;
- dual-grid, surface-nets-style triangle extraction with linear edge roots;
- area-weighted conversion from triangle meshes, including degenerate filtering.

The octree organizes samples, but the solve itself uses a uniform scalar grid
at the requested depth. Unlike upstream PoissonRecon, this package does not
provide the PLY command-line tools, file streaming, an adaptive higher-order
FEM basis, boundary-type selection, trimming, envelope constraints, auxiliary
attribute interpolation, nonlinear edge fitting, or polygon output. Input and
output are in-memory NumPy arrays, output faces are triangles, and `depth` is
capped at 8 because uniform-grid memory grows cubically.

Fine-grained tests compare against NumPy translations written directly from the
upstream functions cited in the Mojo source. PyMeshLab does not expose those
internal kernels, so end-to-end behavior is separately checked against its
binding of the real PoissonRecon implementation. Tests also assert
watertightness, symmetry, volume, and handling of empty meshes, a single
triangle, duplicate vertices, zero-area faces, non-manifold edges, and
unreferenced vertices. This parity check is geometric and tolerance-based; it
does not claim identical vertices, faces, or scalar coefficients.

## Install

```bash
pixi install
pixi run build
pixi run test
```

The build creates `dist/libmojo-poissonrecon.so`. Python loads it lazily and
rebuilds it when `src/capi.mojo` is newer.

## Usage

```python
import numpy as np
from mojo_poissonrecon import reconstruct

rng = np.random.default_rng(0)
z = rng.uniform(-1.0, 1.0, 2000)
theta = rng.uniform(0.0, 2.0 * np.pi, 2000)
r = np.sqrt(1.0 - z * z)
points = np.column_stack((r * np.cos(theta), r * np.sin(theta), z))

# Unit-sphere positions are also outward unit normals.
mesh = reconstruct(points, points, depth=5)
print(mesh.vertices.shape, mesh.faces.shape)
print(mesh.density.min(), mesh.density.max())
```

For already normalized points, `solve_screened_poisson` exposes the scalar
field, density grid, right-hand side, residual, iso-value, and flat octree.
`extract_isosurface` can contour any cubic Float64 scalar grid independently.
`reconstruct_mesh(vertices, faces)` derives area-weighted vertex normals before
reconstruction.

## How it works

NumPy owns every allocation. Contiguous Float64 and Int64 arrays cross the C ABI
as integer addresses, and the exported Mojo functions rebuild typed pointers
without copying. The hot grids use row-major `[z, y, x]` storage. Octree nodes
are breadth-first arrays of depth, Morton code, parent, and contiguous child
range rather than C++ pointer graphs.

Python orchestrates the multigrid hierarchy while Mojo performs splatting,
assembly, red-black smoothing, residual evaluation, full-weighting
restriction, trilinear prolongation, iso-value evaluation, and both passes of
dual contour extraction. Caller-owned buffers mean the shared library neither
allocates nor retains memory across calls. Multigrid residual buffers are
allocated once per level and reused across cycles.

The Python boundary validates shapes, exact integer conversions, finiteness,
contiguity, alignment, and non-null addresses before synchronous native calls;
the local NumPy variables keep every buffer alive for the duration of each call.
Residual evaluation uses native-width Float64 SIMD with a scalar remainder.
Large grids use thresholded CPU parallelism for red-black relaxation, residual
evaluation, restriction, and prolongation; smaller grids remain serial to avoid
thread-launch overhead. No GPU path is provided.

## Benchmarks

Run benchmarks only through `pixi run bench`; the task holds a machine-wide
flock. These are the best-of-three results printed by `pixi run bench` on this
machine: Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86_64, Python 3.11.15. The
Mojo parallel runtime and PyMeshLab used the machine's 36 physical cores.

| Kernel | Mojo | Reference | Reference implementation | Speedup |
|---|---:|---:|---|---:|
| Morton octree coding, 50k points depth 8 | 2.052 ms | 5666.938 ms | NumPy source translation | 2761.10x |
| Density/confidence splat, 30k points | 8.615 ms | 3443.053 ms | NumPy source translation | 399.65x |
| Dual contour, 49^3 scalar grid | 10.664 ms | 3042.784 ms | NumPy source translation | 285.34x |
| Full reconstruction, 2k points depth 5 | 32.705 ms | 694.586 ms | PyMeshLab PoissonRecon (36 threads) | 21.24x |
| Full reconstruction, 20k points depth 7 | 745.872 ms | 3255.285 ms | PyMeshLab PoissonRecon (36 threads) | 4.36x |

The first three comparisons isolate kernels that PyMeshLab does not expose.
Their NumPy references favor auditability and parity, not speed. The final two
rows use the comparable public reconstruction pipeline backed by upstream C++.
