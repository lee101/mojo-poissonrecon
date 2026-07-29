#!/usr/bin/env bash
set -euo pipefail

mkdir -p dist
mojo build --emit shared-lib src/capi.mojo -o dist/libmojo-poissonrecon.so
