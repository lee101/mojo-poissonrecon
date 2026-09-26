"""ctypes loader for the caller-owned-buffer Mojo ABI."""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "capi.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-poissonrecon.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpr_parallel_init": ([], I),
    "mpr_octree_codes_f64": ([I, I, I, I], None),
    "mpr_splat_f64": ([I] * 10 + [F], I),
    "mpr_assemble_f64": ([I, I, I, I, I, I, F, F], None),
    "mpr_relax_f64": ([I, I, I, I, F, I], None),
    "mpr_residual_f64": ([I, I, I, I, I, F], F),
    "mpr_restrict_f64": ([I, I, I], None),
    "mpr_prolong_add_f64": ([I, I, I], None),
    "mpr_iso_value_f64": ([I, I, I, I, I, I], F),
    "mpr_dual_vertices_f64": ([I, I, I, F, I, I, I, I], I),
    "mpr_dual_faces_f64": ([I, I, F, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def _link_args(mojo: str) -> list[str]:
    """Record the AsyncRT runtime that ctypes resolves through this library."""
    lib_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(mojo))), "lib"
    )
    if not os.path.isfile(os.path.join(lib_dir, "libKGENCompilerRTShared.so")):
        raise BuildError("cannot locate the Mojo toolchain lib directory")
    return [
        "-Xlinker", "--no-as-needed",
        "-Xlinker", f"-L{lib_dir}",
        "-Xlinker", "-lKGENCompilerRTShared",
        "-Xlinker", "-lAsyncRTMojoBindings",
        "-Xlinker", "--as-needed",
    ]


def build(force: bool = False) -> str:
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run inside `pixi run`")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB]
        + _link_args(mojo),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
        parallel_device = _loaded.mpr_parallel_init()
        release_device = _loaded.KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice
        release_device.argtypes = [I]
        release_device.restype = None
        atexit.register(release_device, parallel_device)
    return _loaded


def f64(values, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(values)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex values cannot be converted to Float64")
    if not (
        np.issubdtype(source.dtype, np.number)
        or np.issubdtype(source.dtype, np.bool_)
    ):
        raise TypeError("values must be real numbers")
    if np.issubdtype(source.dtype, np.floating) and source.dtype.itemsize > 8:
        raise TypeError("extended-precision floats cannot be narrowed to Float64")
    if np.issubdtype(source.dtype, np.integer):
        limit = 1 << 53
        if source.size and (np.any(source > limit) or np.any(source < -limit)):
            raise ValueError("integer value cannot be represented exactly as Float64")
    if copy:
        return np.array(values, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(values, dtype=np.float64)


def i64(values, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(values)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex values cannot be converted to Int64")
    if np.issubdtype(source.dtype, np.floating):
        if source.size and (
            not np.isfinite(source).all() or np.any(source != np.trunc(source))
        ):
            raise ValueError("Int64 values must be finite integers")
    elif not (
        np.issubdtype(source.dtype, np.integer)
        or np.issubdtype(source.dtype, np.bool_)
    ):
        raise TypeError("values cannot be converted safely to Int64")
    info = np.iinfo(np.int64)
    if source.size and (np.any(source < info.min) or np.any(source > info.max)):
        raise OverflowError("value is outside the Int64 range")
    if copy:
        return np.array(values, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(values, dtype=np.int64)


def addr(values: np.ndarray) -> int:
    if not isinstance(values, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if values.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError("FFI buffers must have native Float64 or Int64 dtype")
    if not values.flags.c_contiguous or not values.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    address = int(values.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return address
