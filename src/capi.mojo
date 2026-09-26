"""Flat-buffer screened Poisson reconstruction kernels."""

from std.ffi import external_call
from std.math import iota, sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_GRID_THRESHOLD = 262_144
comptime GRID_Z_CHUNK = 4


# Mojo 1.2.0 removed `std.runtime.asyncrt` entirely, so the old TaskGroup
# fan-out no longer exists. `parallelize` is now a serial chunk loop: the
# grid kernels below are memory-bound (a handful of flops per 8-byte
# element streamed), so threading them costs more than it saves. The call
# sites and the exported ABI are unchanged.
@always_inline
def parallelize[
    origins: OriginSet, //, func: def(Int) capturing[origins] -> None
](num_work_items: Int):
    for task in range(num_work_items):
        func(task)


@always_inline
def fp(address: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=address)


@always_inline
def ip(address: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=address)


@export("mpr_parallel_init")
def mpr_parallel_init() abi("C") -> Int:
    return 0


@always_inline
def grid_index(x: Int, y: Int, z: Int, n: Int) -> Int:
    return (z * n + y) * n + x


@always_inline
def cell_index(x: Int, y: Int, z: Int, cells: Int) -> Int:
    return (z * cells + y) * cells + x


@always_inline
def norm3(x: Float64, y: Float64, z: Float64) -> Float64:
    var scale = max(abs(x), max(abs(y), abs(z)))
    if scale == 0.0:
        return 0.0
    var sx = x / scale
    var sy = y / scale
    var sz = z / scale
    return scale * sqrt(sx * sx + sy * sy + sz * sz)


# PoissonRecon: Src/RegularTree.h RegularTreeNode::ChildIndex
@export("mpr_octree_codes_f64")
def mpr_octree_codes_f64(
    points_address: Int,
    point_count: Int,
    depth: Int,
    codes_address: Int,
) abi("C"):
    var points = fp(points_address)
    var codes = ip(codes_address)
    for i in range(point_count):
        var x = points[3 * i]
        var y = points[3 * i + 1]
        var z = points[3 * i + 2]
        var code: Int = 0
        codes[i * (depth + 1)] = Int64(0)
        for d in range(depth):
            var scale = 1 << (d + 1)
            var ix = Int(x * Float64(scale))
            var iy = Int(y * Float64(scale))
            var iz = Int(z * Float64(scale))
            if ix < 0:
                ix = 0
            if iy < 0:
                iy = 0
            if iz < 0:
                iz = 0
            if ix >= scale:
                ix = scale - 1
            if iy >= scale:
                iy = scale - 1
            if iz >= scale:
                iz = scale - 1
            var child = (1 if (ix & 1) != 0 else 0)
            child |= (2 if (iy & 1) != 0 else 0)
            child |= (4 if (iz & 1) != 0 else 0)
            code = (code << 3) | child
            codes[i * (depth + 1) + d + 1] = Int64(code)


# PoissonRecon: Src/Reconstructors.h Poisson::Solver::Solve sample Process
# PoissonRecon: Src/FEMTree.WeightedSamples.inl FEMTree::_addWeightContribution
@export("mpr_splat_f64")
def mpr_splat_f64(
    points_address: Int,
    normals_address: Int,
    point_count: Int,
    n: Int,
    confidence: Int,
    density_address: Int,
    vx_address: Int,
    vy_address: Int,
    vz_address: Int,
    screen_address: Int,
    samples_per_node: Float64,
) abi("C") -> Int:
    var points = fp(points_address)
    var normals = fp(normals_address)
    var density = fp(density_address)
    var vx = fp(vx_address)
    var vy = fp(vy_address)
    var vz = fp(vz_address)
    var screen = fp(screen_address)
    var resolution = n - 1
    var accepted = 0
    for pidx in range(point_count):
        var nx = normals[3 * pidx]
        var ny = normals[3 * pidx + 1]
        var nz = normals[3 * pidx + 2]
        var length = norm3(nx, ny, nz)
        if length <= 0.0:
            continue
        var sample_weight = length if confidence != 0 else 1.0
        nx /= length
        ny /= length
        nz /= length

        var gx = points[3 * pidx] * Float64(resolution)
        var gy = points[3 * pidx + 1] * Float64(resolution)
        var gz = points[3 * pidx + 2] * Float64(resolution)
        var ix = Int(gx)
        var iy = Int(gy)
        var iz = Int(gz)
        if ix < 0:
            ix = 0
        if iy < 0:
            iy = 0
        if iz < 0:
            iz = 0
        if ix >= resolution:
            ix = resolution - 1
            gx = Float64(resolution)
        if iy >= resolution:
            iy = resolution - 1
            gy = Float64(resolution)
        if iz >= resolution:
            iz = resolution - 1
            gz = Float64(resolution)
        var fx = gx - Float64(ix)
        var fy = gy - Float64(iy)
        var fz = gz - Float64(iz)
        for dz in range(2):
            var wz = fz if dz == 1 else 1.0 - fz
            for dy in range(2):
                var wy = fy if dy == 1 else 1.0 - fy
                for dx in range(2):
                    var wx = fx if dx == 1 else 1.0 - fx
                    var w = wx * wy * wz * sample_weight
                    var idx = grid_index(ix + dx, iy + dy, iz + dz, n)
                    density[idx] += 2.0 * w
                    screen[idx] += w
        accepted += 1

    # PoissonRecon: Src/FEMTree.WeightedSamples.inl FEMTree::_getSampleDepthAndWeight
    for pidx in range(point_count):
        var nx = normals[3 * pidx]
        var ny = normals[3 * pidx + 1]
        var nz = normals[3 * pidx + 2]
        var length = norm3(nx, ny, nz)
        if length <= 0.0:
            continue
        var sample_weight = length if confidence != 0 else 1.0
        nx /= length
        ny /= length
        nz /= length
        var gx = points[3 * pidx] * Float64(resolution)
        var gy = points[3 * pidx + 1] * Float64(resolution)
        var gz = points[3 * pidx + 2] * Float64(resolution)
        var ix = min(max(Int(gx), 0), resolution - 1)
        var iy = min(max(Int(gy), 0), resolution - 1)
        var iz = min(max(Int(gz), 0), resolution - 1)
        var fx = min(max(gx - Float64(ix), 0.0), 1.0)
        var fy = min(max(gy - Float64(iy), 0.0), 1.0)
        var fz = min(max(gz - Float64(iz), 0.0), 1.0)
        var local_density: Float64 = 0.0
        for dz in range(2):
            var wz = fz if dz == 1 else 1.0 - fz
            for dy in range(2):
                var wy = fy if dy == 1 else 1.0 - fy
                for dx in range(2):
                    var wx = fx if dx == 1 else 1.0 - fx
                    local_density += (
                        density[grid_index(ix + dx, iy + dy, iz + dz, n)]
                        * wx
                        * wy
                        * wz
                    )
        var density_factor = max(1.0, local_density / samples_per_node)
        for dz in range(2):
            var wz = fz if dz == 1 else 1.0 - fz
            for dy in range(2):
                var wy = fy if dy == 1 else 1.0 - fy
                for dx in range(2):
                    var wx = fx if dx == 1 else 1.0 - fx
                    var w = wx * wy * wz * sample_weight
                    var idx = grid_index(ix + dx, iy + dy, iz + dz, n)
                    var vector_weight = (
                        w * Float64(resolution) / density_factor
                    )
                    vx[idx] -= nx * vector_weight
                    vy[idx] -= ny * vector_weight
                    vz[idx] -= nz * vector_weight
    return accepted


# PoissonRecon: Src/FEMTree.System.inl FEMTree::_addFEMConstraints
@export("mpr_assemble_f64")
def mpr_assemble_f64(
    vx_address: Int,
    vy_address: Int,
    vz_address: Int,
    screen_address: Int,
    rhs_address: Int,
    n: Int,
    point_weight: Float64,
    target_value: Float64,
) abi("C"):
    var vx = fp(vx_address)
    var vy = fp(vy_address)
    var vz = fp(vz_address)
    var screen = fp(screen_address)
    var rhs = fp(rhs_address)
    var h_inv = Float64(n - 1)
    var h_inv2 = h_inv * h_inv
    for z in range(n):
        for y in range(n):
            for x in range(n):
                var idx = grid_index(x, y, z, n)
                var div: Float64 = 0.0
                if x == 0:
                    div += (vx[grid_index(1, y, z, n)] - vx[idx]) * h_inv
                elif x == n - 1:
                    div += (vx[idx] - vx[grid_index(x - 1, y, z, n)]) * h_inv
                else:
                    div += (
                        vx[grid_index(x + 1, y, z, n)]
                        - vx[grid_index(x - 1, y, z, n)]
                    ) * (0.5 * h_inv)
                if y == 0:
                    div += (vy[grid_index(x, 1, z, n)] - vy[idx]) * h_inv
                elif y == n - 1:
                    div += (vy[idx] - vy[grid_index(x, y - 1, z, n)]) * h_inv
                else:
                    div += (
                        vy[grid_index(x, y + 1, z, n)]
                        - vy[grid_index(x, y - 1, z, n)]
                    ) * (0.5 * h_inv)
                if z == 0:
                    div += (vz[grid_index(x, y, 1, n)] - vz[idx]) * h_inv
                elif z == n - 1:
                    div += (vz[idx] - vz[grid_index(x, y, z - 1, n)]) * h_inv
                else:
                    div += (
                        vz[grid_index(x, y, z + 1, n)]
                        - vz[grid_index(x, y, z - 1, n)]
                    ) * (0.5 * h_inv)
                var screening = point_weight * screen[idx] * h_inv2
                rhs[idx] = -div + screening * target_value


@always_inline
def stencil_sum_and_degree(
    values: F64Ptr, x: Int, y: Int, z: Int, n: Int
) -> Tuple[Float64, Int]:
    var neighbor_sum: Float64 = 0.0
    var degree = 0
    if x > 0:
        neighbor_sum += values[grid_index(x - 1, y, z, n)]
        degree += 1
    if x + 1 < n:
        neighbor_sum += values[grid_index(x + 1, y, z, n)]
        degree += 1
    if y > 0:
        neighbor_sum += values[grid_index(x, y - 1, z, n)]
        degree += 1
    if y + 1 < n:
        neighbor_sum += values[grid_index(x, y + 1, z, n)]
        degree += 1
    if z > 0:
        neighbor_sum += values[grid_index(x, y, z - 1, n)]
        degree += 1
    if z + 1 < n:
        neighbor_sum += values[grid_index(x, y, z + 1, n)]
        degree += 1
    return neighbor_sum, degree


# PoissonRecon: Src/FEMTree.System.inl FEMTree::_solveSystemGS
@export("mpr_relax_f64")
def mpr_relax_f64(
    solution_address: Int,
    rhs_address: Int,
    screen_address: Int,
    n: Int,
    point_weight: Float64,
    iterations: Int,
) abi("C"):
    var h_inv = Float64(n - 1)
    var h_inv2 = h_inv * h_inv
    var task_count = (n + GRID_Z_CHUNK - 1) // GRID_Z_CHUNK
    var active_color: Int

    @parameter
    def work(task: Int):
        comptime W = simdwidthof[DType.float64]()
        var work_solution = fp(solution_address)
        var work_rhs = fp(rhs_address)
        var work_screen = fp(screen_address)
        var start_z = task * GRID_Z_CHUNK
        var end_z = min(start_z + GRID_Z_CHUNK, n)
        for z in range(start_z, end_z):
            for y in range(n):
                var degree = 2
                if y > 0:
                    degree += 1
                if y + 1 < n:
                    degree += 1
                if z > 0:
                    degree += 1
                if z + 1 < n:
                    degree += 1
                if ((y + z) & 1) == active_color:
                    var boundary_idx = grid_index(0, y, z, n)
                    var boundary_pair = stencil_sum_and_degree(
                        work_solution, 0, y, z, n
                    )
                    var boundary_screening = (
                        point_weight * work_screen[boundary_idx] * h_inv2
                    )
                    var boundary_diagonal = (
                        Float64(boundary_pair[1]) * h_inv2
                        + boundary_screening
                    )
                    if boundary_diagonal > 1.0e-30:
                        work_solution[boundary_idx] = (
                            work_rhs[boundary_idx]
                            + boundary_pair[0] * h_inv2
                        ) / boundary_diagonal
                var vector_end = 1 + (n - 2) // W * W
                for x in range(1, vector_end, W):
                    var idx = grid_index(x, y, z, n)
                    var center = work_solution.load[width=W](idx)
                    var neighbor_sum = (
                        work_solution.load[width=W](idx - 1)
                        + work_solution.load[width=W](idx + 1)
                    )
                    if y > 0:
                        neighbor_sum += work_solution.load[width=W](idx - n)
                    if y + 1 < n:
                        neighbor_sum += work_solution.load[width=W](idx + n)
                    if z > 0:
                        neighbor_sum += work_solution.load[width=W](idx - n * n)
                    if z + 1 < n:
                        neighbor_sum += work_solution.load[width=W](idx + n * n)
                    var screening = (
                        point_weight
                        * work_screen.load[width=W](idx)
                        * h_inv2
                    )
                    var diagonal = Float64(degree) * h_inv2 + screening
                    var updated = (
                        work_rhs.load[width=W](idx) + neighbor_sum * h_inv2
                    ) / diagonal
                    var active = (
                        (iota[DType.int, W](x) + y + z) & 1
                    ).eq(active_color)
                    work_solution.store(idx, active.select(updated, center))
                for x in range(vector_end, n):
                    if ((x + y + z) & 1) != active_color:
                        continue
                    var idx = grid_index(x, y, z, n)
                    var pair = stencil_sum_and_degree(
                        work_solution, x, y, z, n
                    )
                    var screening = point_weight * work_screen[idx] * h_inv2
                    var diagonal = Float64(pair[1]) * h_inv2 + screening
                    if diagonal > 1.0e-30:
                        work_solution[idx] = (
                            work_rhs[idx] + pair[0] * h_inv2
                        ) / diagonal

    for _ in range(iterations):
        for color in range(2):
            active_color = color
            if n * n * n >= PARALLEL_GRID_THRESHOLD:
                parallelize[work](task_count)
            else:
                for task in range(task_count):
                    work(task)


@always_inline
def residual_row(
    solution: F64Ptr,
    rhs: F64Ptr,
    screen: F64Ptr,
    residual: F64Ptr,
    x_start: Int,
    x_end: Int,
    y: Int,
    z: Int,
    n: Int,
    h_inv2: Float64,
    point_weight: Float64,
):
    comptime W = simdwidthof[DType.float64]()
    var row_start = (z * n + y) * n
    var degree = 2
    if y > 0:
        degree += 1
    if y + 1 < n:
        degree += 1
    if z > 0:
        degree += 1
    if z + 1 < n:
        degree += 1
    var vector_end = x_start + (x_end - x_start) // W * W
    for x in range(x_start, vector_end, W):
        var idx = row_start + x
        var center = solution.load[width=W](idx)
        var neighbor_sum = (
            solution.load[width=W](idx - 1)
            + solution.load[width=W](idx + 1)
        )
        if y > 0:
            neighbor_sum += solution.load[width=W](idx - n)
        if y + 1 < n:
            neighbor_sum += solution.load[width=W](idx + n)
        if z > 0:
            neighbor_sum += solution.load[width=W](idx - n * n)
        if z + 1 < n:
            neighbor_sum += solution.load[width=W](idx + n * n)
        var screening = (
            point_weight * screen.load[width=W](idx) * h_inv2
        )
        var applied = (
            (Float64(degree) * h_inv2 + screening) * center
            - neighbor_sum * h_inv2
        )
        residual.store(idx, rhs.load[width=W](idx) - applied)
    for x in range(vector_end, x_end):
        var idx = row_start + x
        var pair = stencil_sum_and_degree(solution, x, y, z, n)
        var screening = point_weight * screen[idx] * h_inv2
        var applied = (
            (Float64(pair[1]) * h_inv2 + screening) * solution[idx]
            - pair[0] * h_inv2
        )
        residual[idx] = rhs[idx] - applied


# PoissonRecon: Src/FEMTree.System.inl FEMTree::_getResidual
@export("mpr_residual_f64")
def mpr_residual_f64(
    solution_address: Int,
    rhs_address: Int,
    screen_address: Int,
    residual_address: Int,
    n: Int,
    point_weight: Float64,
) abi("C") -> Float64:
    var solution = fp(solution_address)
    var rhs = fp(rhs_address)
    var screen = fp(screen_address)
    var residual = fp(residual_address)
    var h_inv = Float64(n - 1)
    var h_inv2 = h_inv * h_inv
    var task_count = (n + GRID_Z_CHUNK - 1) // GRID_Z_CHUNK

    @parameter
    def work(task: Int):
        var work_solution = fp(solution_address)
        var work_rhs = fp(rhs_address)
        var work_screen = fp(screen_address)
        var work_residual = fp(residual_address)
        var start_z = task * GRID_Z_CHUNK
        var end_z = min(start_z + GRID_Z_CHUNK, n)
        for z in range(start_z, end_z):
            for y in range(n):
                var left_idx = grid_index(0, y, z, n)
                var pair = stencil_sum_and_degree(
                    work_solution, 0, y, z, n
                )
                var screening = point_weight * work_screen[left_idx] * h_inv2
                var applied = (
                    (Float64(pair[1]) * h_inv2 + screening)
                    * work_solution[left_idx]
                    - pair[0] * h_inv2
                )
                work_residual[left_idx] = work_rhs[left_idx] - applied
                residual_row(
                    work_solution,
                    work_rhs,
                    work_screen,
                    work_residual,
                    1,
                    n - 1,
                    y,
                    z,
                    n,
                    h_inv2,
                    point_weight,
                )
                var right_idx = grid_index(n - 1, y, z, n)
                pair = stencil_sum_and_degree(
                    work_solution, n - 1, y, z, n
                )
                screening = point_weight * work_screen[right_idx] * h_inv2
                applied = (
                    (Float64(pair[1]) * h_inv2 + screening)
                    * work_solution[right_idx]
                    - pair[0] * h_inv2
                )
                work_residual[right_idx] = work_rhs[right_idx] - applied

    if n * n * n >= PARALLEL_GRID_THRESHOLD:
        parallelize[work](task_count)
    else:
        for task in range(task_count):
            work(task)

    comptime W = simdwidthof[DType.float64]()
    var size = n * n * n
    var vector_end = size // W * W
    var square_sums = SIMD[DType.float64, W](0.0)
    for idx in range(0, vector_end, W):
        var values = residual.load[width=W](idx)
        square_sums += values * values
    var square_norm = square_sums.reduce_add()
    for idx in range(vector_end, size):
        square_norm += residual[idx] * residual[idx]
    return sqrt(square_norm)


# PoissonRecon: Src/FEMTree.System.inl FEMTree::_downSample
@export("mpr_restrict_f64")
def mpr_restrict_f64(
    fine_address: Int,
    coarse_address: Int,
    fine_n: Int,
) abi("C"):
    var coarse_n = (fine_n + 1) // 2
    var task_count = (coarse_n + GRID_Z_CHUNK - 1) // GRID_Z_CHUNK

    @parameter
    def work(task: Int):
        comptime W = simdwidthof[DType.float64]()
        var work_fine = fp(fine_address)
        var work_coarse = fp(coarse_address)
        var start_z = task * GRID_Z_CHUNK
        var end_z = min(start_z + GRID_Z_CHUNK, coarse_n)
        for z in range(start_z, end_z):
            for y in range(coarse_n):
                var fz = 2 * z
                var fy = 2 * y
                var boundary_sum: Float64 = 0.0
                var boundary_weight: Float64 = 0.0
                for oz in range(-1, 2):
                    var boundary_z = fz + oz
                    if boundary_z < 0 or boundary_z >= fine_n:
                        continue
                    var boundary_wz: Float64 = 2.0 if oz == 0 else 1.0
                    for oy in range(-1, 2):
                        var boundary_y = fy + oy
                        if boundary_y < 0 or boundary_y >= fine_n:
                            continue
                        var boundary_wy: Float64 = 2.0 if oy == 0 else 1.0
                        for ox in range(0, 2):
                            var boundary_wx: Float64 = 2.0 if ox == 0 else 1.0
                            var boundary_w = (
                                boundary_wx * boundary_wy * boundary_wz
                            )
                            boundary_sum += (
                                work_fine[
                                    grid_index(
                                        ox, boundary_y, boundary_z, fine_n
                                    )
                                ]
                                * boundary_w
                            )
                            boundary_weight += boundary_w
                work_coarse[grid_index(0, y, z, coarse_n)] = (
                    boundary_sum / boundary_weight
                )
                var vector_end = 1 + (coarse_n - 2) // W * W
                for x in range(1, vector_end, W):
                    var weighted_sums = SIMD[DType.float64, W](0.0)
                    var weight_sum: Float64 = 0.0
                    for oz in range(-1, 2):
                        var zz = fz + oz
                        if zz < 0 or zz >= fine_n:
                            continue
                        var wz: Float64 = 2.0 if oz == 0 else 1.0
                        for oy in range(-1, 2):
                            var yy = fy + oy
                            if yy < 0 or yy >= fine_n:
                                continue
                            var wy: Float64 = 2.0 if oy == 0 else 1.0
                            var row = grid_index(2 * x, yy, zz, fine_n)
                            weighted_sums += (
                                (work_fine + row - 1).strided_load[width=W](2)
                                * (wy * wz)
                            )
                            weighted_sums += (
                                (work_fine + row).strided_load[width=W](2)
                                * (2.0 * wy * wz)
                            )
                            weighted_sums += (
                                (work_fine + row + 1).strided_load[width=W](2)
                                * (wy * wz)
                            )
                            weight_sum += 4.0 * wy * wz
                    work_coarse.store(
                        grid_index(x, y, z, coarse_n),
                        weighted_sums / weight_sum,
                    )
                for x in range(vector_end, coarse_n):
                    var weighted_sum: Float64 = 0.0
                    var weight_sum: Float64 = 0.0
                    var fx = 2 * x
                    for oz in range(-1, 2):
                        var zz = fz + oz
                        if zz < 0 or zz >= fine_n:
                            continue
                        var wz: Float64 = 2.0 if oz == 0 else 1.0
                        for oy in range(-1, 2):
                            var yy = fy + oy
                            if yy < 0 or yy >= fine_n:
                                continue
                            var wy: Float64 = 2.0 if oy == 0 else 1.0
                            for ox in range(-1, 2):
                                var xx = fx + ox
                                if xx < 0 or xx >= fine_n:
                                    continue
                                var wx: Float64 = 2.0 if ox == 0 else 1.0
                                var w = wx * wy * wz
                                weighted_sum += (
                                    work_fine[
                                        grid_index(xx, yy, zz, fine_n)
                                    ]
                                    * w
                                )
                                weight_sum += w
                    work_coarse[grid_index(x, y, z, coarse_n)] = (
                        weighted_sum / weight_sum
                    )

    if coarse_n * coarse_n * coarse_n >= PARALLEL_GRID_THRESHOLD:
        parallelize[work](task_count)
    else:
        for task in range(task_count):
            work(task)


# PoissonRecon: Src/FEMTree.System.inl FEMTree::_upSample
@export("mpr_prolong_add_f64")
def mpr_prolong_add_f64(
    coarse_address: Int,
    fine_address: Int,
    fine_n: Int,
) abi("C"):
    var coarse_n = (fine_n + 1) // 2
    var task_count = (fine_n + GRID_Z_CHUNK - 1) // GRID_Z_CHUNK

    @parameter
    def work(task: Int):
        var work_coarse = fp(coarse_address)
        var work_fine = fp(fine_address)
        var start_z = task * GRID_Z_CHUNK
        var end_z = min(start_z + GRID_Z_CHUNK, fine_n)
        for z in range(start_z, end_z):
            var z0 = z // 2
            var z1 = min(z0 + 1, coarse_n - 1)
            var tz: Float64 = 0.5 if (z & 1) != 0 else 0.0
            for y in range(fine_n):
                var y0 = y // 2
                var y1 = min(y0 + 1, coarse_n - 1)
                var ty: Float64 = 0.5 if (y & 1) != 0 else 0.0
                for x in range(fine_n):
                    var x0 = x // 2
                    var x1 = min(x0 + 1, coarse_n - 1)
                    var tx: Float64 = 0.5 if (x & 1) != 0 else 0.0
                    var c000 = work_coarse[
                        grid_index(x0, y0, z0, coarse_n)
                    ]
                    var c100 = work_coarse[
                        grid_index(x1, y0, z0, coarse_n)
                    ]
                    var c010 = work_coarse[
                        grid_index(x0, y1, z0, coarse_n)
                    ]
                    var c110 = work_coarse[
                        grid_index(x1, y1, z0, coarse_n)
                    ]
                    var c001 = work_coarse[
                        grid_index(x0, y0, z1, coarse_n)
                    ]
                    var c101 = work_coarse[
                        grid_index(x1, y0, z1, coarse_n)
                    ]
                    var c011 = work_coarse[
                        grid_index(x0, y1, z1, coarse_n)
                    ]
                    var c111 = work_coarse[
                        grid_index(x1, y1, z1, coarse_n)
                    ]
                    var c00 = c000 * (1.0 - tx) + c100 * tx
                    var c10 = c010 * (1.0 - tx) + c110 * tx
                    var c01 = c001 * (1.0 - tx) + c101 * tx
                    var c11 = c011 * (1.0 - tx) + c111 * tx
                    var c0 = c00 * (1.0 - ty) + c10 * ty
                    var c1 = c01 * (1.0 - ty) + c11 * ty
                    work_fine[grid_index(x, y, z, fine_n)] += (
                        c0 * (1.0 - tz) + c1 * tz
                    )

    if fine_n * fine_n * fine_n >= PARALLEL_GRID_THRESHOLD:
        parallelize[work](task_count)
    else:
        for task in range(task_count):
            work(task)


# PoissonRecon: Src/Reconstructors.h Poisson::Solver::Solve iso-value average
@export("mpr_iso_value_f64")
def mpr_iso_value_f64(
    field_address: Int,
    points_address: Int,
    normals_address: Int,
    point_count: Int,
    n: Int,
    confidence: Int,
) abi("C") -> Float64:
    var field = fp(field_address)
    var points = fp(points_address)
    var normals = fp(normals_address)
    var resolution = n - 1
    var value_sum: Float64 = 0.0
    var weight_sum: Float64 = 0.0
    for pidx in range(point_count):
        var weight = (
            norm3(
                normals[3 * pidx],
                normals[3 * pidx + 1],
                normals[3 * pidx + 2],
            )
            if confidence != 0
            else 1.0
        )
        var gx = points[3 * pidx] * Float64(resolution)
        var gy = points[3 * pidx + 1] * Float64(resolution)
        var gz = points[3 * pidx + 2] * Float64(resolution)
        var ix = min(max(Int(gx), 0), resolution - 1)
        var iy = min(max(Int(gy), 0), resolution - 1)
        var iz = min(max(Int(gz), 0), resolution - 1)
        var fx = min(max(gx - Float64(ix), 0.0), 1.0)
        var fy = min(max(gy - Float64(iy), 0.0), 1.0)
        var fz = min(max(gz - Float64(iz), 0.0), 1.0)
        var value: Float64 = 0.0
        for dz in range(2):
            var wz = fz if dz == 1 else 1.0 - fz
            for dy in range(2):
                var wy = fy if dy == 1 else 1.0 - fy
                for dx in range(2):
                    var wx = fx if dx == 1 else 1.0 - fx
                    value += (
                        field[grid_index(ix + dx, iy + dy, iz + dz, n)]
                        * wx
                        * wy
                        * wz
                    )
        value_sum += value * weight
        weight_sum += weight
    return value_sum / weight_sum if weight_sum > 0.0 else 0.0


@always_inline
def corner_value(
    field: F64Ptr, x: Int, y: Int, z: Int, n: Int, corner: Int
) -> Float64:
    return field[
        grid_index(
            x + (corner & 1),
            y + ((corner >> 1) & 1),
            z + ((corner >> 2) & 1),
            n,
        )
    ]


@always_inline
def add_edge_root(
    v0: Float64,
    v1: Float64,
    iso: Float64,
    x0: Float64,
    y0: Float64,
    z0: Float64,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    sx: Float64,
    sy: Float64,
    sz: Float64,
) -> Tuple[Float64, Float64, Float64, Int]:
    if (v0 < iso) == (v1 < iso):
        return sx, sy, sz, 0
    var t = (iso - v0) / (v1 - v0)
    t = min(max(t, 0.0), 1.0)
    return (
        sx + x0 + (x1 - x0) * t,
        sy + y0 + (y1 - y0) * t,
        sz + z0 + (z1 - z0) * t,
        1,
    )


# PoissonRecon: Src/FEMTree.LevelSet.3D.inl LevelSetExtractor::_SetIsoVertices
# PoissonRecon: Src/MultiGridOctreeData.IsoSurface.inl Octree::_getIsoVertex
@export("mpr_dual_vertices_f64")
def mpr_dual_vertices_f64(
    field_address: Int,
    density_address: Int,
    n: Int,
    iso: Float64,
    cell_map_address: Int,
    vertices_address: Int,
    vertex_density_address: Int,
    capacity: Int,
) abi("C") -> Int:
    var field = fp(field_address)
    var cells = n - 1
    var count = 0
    if capacity <= 0:
        var counts = ip(cell_map_address)
        var task_count = (cells + GRID_Z_CHUNK - 1) // GRID_Z_CHUNK

        @parameter
        def count_work(task: Int):
            comptime W = simdwidthof[DType.float64]()
            var work_field = fp(field_address)
            var work_counts = ip(cell_map_address)
            var local_count: Int64 = 0
            var start_z = task * GRID_Z_CHUNK
            var end_z = min(start_z + GRID_Z_CHUNK, cells)
            for z in range(start_z, end_z):
                for y in range(cells):
                    for vector_x in range(0, cells // W * W, W):
                        var x = vector_x
                        var below = SIMD[DType.int, W](0)
                        var row0 = grid_index(x, y, z, n)
                        var row1 = grid_index(x, y + 1, z, n)
                        var row2 = grid_index(x, y, z + 1, n)
                        var row3 = grid_index(x, y + 1, z + 1, n)
                        below += work_field.load[width=W](row0).lt(iso).select(1, 0)
                        below += work_field.load[width=W](row0 + 1).lt(iso).select(1, 0)
                        below += work_field.load[width=W](row1).lt(iso).select(1, 0)
                        below += work_field.load[width=W](row1 + 1).lt(iso).select(1, 0)
                        below += work_field.load[width=W](row2).lt(iso).select(1, 0)
                        below += work_field.load[width=W](row2 + 1).lt(iso).select(1, 0)
                        below += work_field.load[width=W](row3).lt(iso).select(1, 0)
                        below += work_field.load[width=W](row3 + 1).lt(iso).select(1, 0)
                        var active = below.ne(0) & below.ne(8)
                        local_count += Int64(active.select(1, 0).reduce_add())
                    for tail_x in range(cells // W * W, cells):
                        var x = tail_x
                        var scalar_below = 0
                        for corner in range(8):
                            if corner_value(work_field, x, y, z, n, corner) < iso:
                                scalar_below += 1
                        if scalar_below != 0 and scalar_below != 8:
                            local_count += 1
            work_counts[task] = local_count

        if cells * cells * cells >= PARALLEL_GRID_THRESHOLD:
            parallelize[count_work](task_count)
        else:
            for task in range(task_count):
                count_work(task)
        for task in range(task_count):
            count += Int(counts[task])
        return count

    var density = fp(density_address)
    var cell_map = ip(cell_map_address)
    var vertices = fp(vertices_address)
    var vertex_density = fp(vertex_density_address)
    var mark_task_count = (cells + GRID_Z_CHUNK - 1) // GRID_Z_CHUNK

    @parameter
    def mark_work(task: Int):
        comptime W = simdwidthof[DType.float64]()
        var work_field = fp(field_address)
        var work_cell_map = ip(cell_map_address)
        var start_z = task * GRID_Z_CHUNK
        var end_z = min(start_z + GRID_Z_CHUNK, cells)
        for z in range(start_z, end_z):
            for y in range(cells):
                for x in range(0, cells // W * W, W):
                    var below = SIMD[DType.int, W](0)
                    var row0 = grid_index(x, y, z, n)
                    var row1 = grid_index(x, y + 1, z, n)
                    var row2 = grid_index(x, y, z + 1, n)
                    var row3 = grid_index(x, y + 1, z + 1, n)
                    below += work_field.load[width=W](row0).lt(iso).select(1, 0)
                    below += work_field.load[width=W](row0 + 1).lt(iso).select(1, 0)
                    below += work_field.load[width=W](row1).lt(iso).select(1, 0)
                    below += work_field.load[width=W](row1 + 1).lt(iso).select(1, 0)
                    below += work_field.load[width=W](row2).lt(iso).select(1, 0)
                    below += work_field.load[width=W](row2 + 1).lt(iso).select(1, 0)
                    below += work_field.load[width=W](row3).lt(iso).select(1, 0)
                    below += work_field.load[width=W](row3 + 1).lt(iso).select(1, 0)
                    var active = below.ne(0) & below.ne(8)
                    work_cell_map.store(
                        cell_index(x, y, z, cells),
                        active.select(Int64(0), Int64(-1)),
                    )
                for x in range(cells // W * W, cells):
                    var below = 0
                    for corner in range(8):
                        if corner_value(work_field, x, y, z, n, corner) < iso:
                            below += 1
                    work_cell_map[cell_index(x, y, z, cells)] = (
                        Int64(0) if below != 0 and below != 8 else Int64(-1)
                    )

    if cells * cells * cells >= PARALLEL_GRID_THRESHOLD:
        parallelize[mark_work](mark_task_count)
    else:
        for task in range(mark_task_count):
            mark_work(task)
    for z in range(cells):
        for y in range(cells):
            for x in range(cells):
                if cell_map[cell_index(x, y, z, cells)] < 0:
                    continue
                var values = SIMD[DType.float64, 8](0.0)
                for corner in range(8):
                    values[corner] = corner_value(field, x, y, z, n, corner)
                if count >= capacity:
                    count += 1
                    continue
                var sx: Float64 = 0.0
                var sy: Float64 = 0.0
                var sz: Float64 = 0.0
                var roots = 0
                for axis in range(3):
                    for a in range(2):
                        for b in range(2):
                            var c0: Int
                            var c1: Int
                            var x0: Float64
                            var y0: Float64
                            var z0: Float64
                            var x1: Float64
                            var y1: Float64
                            var z1: Float64
                            if axis == 0:
                                c0 = 2 * a + 4 * b
                                c1 = c0 + 1
                                x0 = 0.0
                                y0 = Float64(a)
                                z0 = Float64(b)
                                x1 = 1.0
                                y1 = y0
                                z1 = z0
                            elif axis == 1:
                                c0 = a + 4 * b
                                c1 = c0 + 2
                                x0 = Float64(a)
                                y0 = 0.0
                                z0 = Float64(b)
                                x1 = x0
                                y1 = 1.0
                                z1 = z0
                            else:
                                c0 = a + 2 * b
                                c1 = c0 + 4
                                x0 = Float64(a)
                                y0 = Float64(b)
                                z0 = 0.0
                                x1 = x0
                                y1 = y0
                                z1 = 1.0
                            var root = add_edge_root(
                                values[c0],
                                values[c1],
                                iso,
                                x0,
                                y0,
                                z0,
                                x1,
                                y1,
                                z1,
                                sx,
                                sy,
                                sz,
                            )
                            sx = root[0]
                            sy = root[1]
                            sz = root[2]
                            roots += root[3]
                var inv_roots = 1.0 / Float64(roots)
                vertices[3 * count] = (Float64(x) + sx * inv_roots) / Float64(cells)
                vertices[3 * count + 1] = (Float64(y) + sy * inv_roots) / Float64(cells)
                vertices[3 * count + 2] = (Float64(z) + sz * inv_roots) / Float64(cells)
                var dsum: Float64 = 0.0
                for corner in range(8):
                    dsum += density[
                        grid_index(
                            x + (corner & 1),
                            y + ((corner >> 1) & 1),
                            z + ((corner >> 2) & 1),
                            n,
                        )
                    ]
                vertex_density[count] = dsum * 0.125
                cell_map[cell_index(x, y, z, cells)] = Int64(count)
                count += 1
    return count


@always_inline
def emit_quad(
    faces: I64Ptr,
    face_count: Int,
    capacity: Int,
    a: Int64,
    b: Int64,
    c: Int64,
    d: Int64,
    reverse: Bool,
) -> Int:
    if a < 0 or b < 0 or c < 0 or d < 0:
        return face_count
    if face_count + 2 <= capacity:
        if reverse:
            faces[3 * face_count] = a
            faces[3 * face_count + 1] = d
            faces[3 * face_count + 2] = c
            faces[3 * face_count + 3] = a
            faces[3 * face_count + 4] = c
            faces[3 * face_count + 5] = b
        else:
            faces[3 * face_count] = a
            faces[3 * face_count + 1] = b
            faces[3 * face_count + 2] = c
            faces[3 * face_count + 3] = a
            faces[3 * face_count + 4] = c
            faces[3 * face_count + 5] = d
    return face_count + 2


# PoissonRecon: Src/FEMTree.LevelSet.3D.inl LevelSetExtractor::_SetIsoSurface
@export("mpr_dual_faces_f64")
def mpr_dual_faces_f64(
    field_address: Int,
    n: Int,
    iso: Float64,
    cell_map_address: Int,
    faces_address: Int,
    capacity: Int,
) abi("C") -> Int:
    var field = fp(field_address)
    var cell_map = ip(cell_map_address)
    var faces = ip(faces_address)
    var cells = n - 1
    var face_count = 0
    for z in range(1, cells):
        for y in range(1, cells):
            for x in range(cells):
                var v0 = field[grid_index(x, y, z, n)]
                var v1 = field[grid_index(x + 1, y, z, n)]
                if (v0 < iso) != (v1 < iso):
                    var a = cell_map[cell_index(x, y - 1, z - 1, cells)]
                    var b = cell_map[cell_index(x, y, z - 1, cells)]
                    var c = cell_map[cell_index(x, y, z, cells)]
                    var d = cell_map[cell_index(x, y - 1, z, cells)]
                    face_count = emit_quad(
                        faces, face_count, capacity, a, b, c, d, v0 < iso
                    )
    for z in range(1, cells):
        for y in range(cells):
            for x in range(1, cells):
                var v0 = field[grid_index(x, y, z, n)]
                var v1 = field[grid_index(x, y + 1, z, n)]
                if (v0 < iso) != (v1 < iso):
                    var a = cell_map[cell_index(x - 1, y, z - 1, cells)]
                    var b = cell_map[cell_index(x - 1, y, z, cells)]
                    var c = cell_map[cell_index(x, y, z, cells)]
                    var d = cell_map[cell_index(x, y, z - 1, cells)]
                    face_count = emit_quad(
                        faces, face_count, capacity, a, b, c, d, v0 < iso
                    )
    for z in range(cells):
        for y in range(1, cells):
            for x in range(1, cells):
                var v0 = field[grid_index(x, y, z, n)]
                var v1 = field[grid_index(x, y, z + 1, n)]
                if (v0 < iso) != (v1 < iso):
                    var a = cell_map[cell_index(x - 1, y - 1, z, cells)]
                    var b = cell_map[cell_index(x, y - 1, z, cells)]
                    var c = cell_map[cell_index(x, y, z, cells)]
                    var d = cell_map[cell_index(x - 1, y, z, cells)]
                    face_count = emit_quad(
                        faces, face_count, capacity, a, b, c, d, v0 < iso
                    )
    return face_count
