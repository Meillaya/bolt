#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <recipe-path>" >&2
  exit 2
fi

python3 ./python/scripts/run_experiment_recipe.py "$1"
