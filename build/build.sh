#!/usr/bin/env bash
set -euo pipefail

mkdir -p dist

# The ctypes layer releases this library's AsyncRT CPU device through
# KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice, so the shared object must record and
# resolve the AsyncRT runtime. No compiled object references those symbols, so
# the linker would drop the DT_NEEDED entries under the default --as-needed;
# force them in and restore the default afterwards.
mojo_prefix="$(dirname "$(dirname "$(readlink -f "$(command -v mojo)")")")"

mojo build --emit shared-lib src/capi.mojo -o dist/libmojo-poissonrecon.so \
  -Xlinker --no-as-needed \
  -Xlinker "-L${mojo_prefix}/lib" \
  -Xlinker -lKGENCompilerRTShared \
  -Xlinker -lAsyncRTMojoBindings \
  -Xlinker --as-needed
