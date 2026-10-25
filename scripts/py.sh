#!/bin/bash
# Clean-env python for this repo. Some desktop environments inject a PYTHONPATH
# that shadows venv site-packages; env -i is the only reliable route.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
exec env -i PATH=/usr/bin:/bin:/usr/sbin HOME="${HOME:-/tmp}" \
  "$HERE/.venv/bin/python" "$@"
