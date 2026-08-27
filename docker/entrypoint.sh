#!/bin/bash
# Container entrypoint. First argument selects what to run:
#   batch    batch/start_gemma.sh      (throughput mode)
#   prepare  docker/prepare.sh         (download the model into /app/models)
#   verify   verify.sh [args]
#   <anything else> is exec'd as a command (e.g. bash)
# Before serving, verify.sh --no-server runs and aborts on FAIL (model missing,
# patches not applied, ...); VERIFY=0 skips that.
set -e
cd /app
cmd=${1:-batch}; shift || true
case "$cmd" in
  batch)
    if [ "${VERIFY:-1}" != "0" ]; then
      bash verify.sh --no-server || { echo "entrypoint: verify.sh FAILED — fix the above or set VERIFY=0"; exit 1; }
    fi
    exec bash batch/start_gemma.sh "$@" ;;
  prepare) exec bash docker/prepare.sh "$@" ;;
  verify)  exec bash verify.sh "$@" ;;
  *)       exec "$cmd" "$@" ;;
esac
