#!/usr/bin/env bash
# GTA "Playing for Data" — download all PNG image + label splits (parts 1–10).
# Source page: https://download.visinf.tu-darmstadt.de/data/from_games/
#
# Usage (from repo root):
#   bash tools/download_gta.sh
#   bash tools/download_gta.sh /path/to/data/gta
#
# Requires: curl

set -euo pipefail

BASE_URL="https://download.visinf.tu-darmstadt.de/data/from_games/data"
OUT_DIR="${1:-data/gta}"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

echo "Downloading GTA zips into: $(pwd)"
echo "Research / educational use only — see dataset terms on the upstream page."
echo

# for prefix in 01 02 03 04 05 06 07 08 09 10; do
for prefix in 09 10; do
  for kind in images labels; do
    name="${prefix}_${kind}.zip"
    url="${BASE_URL}/${name}"
    if [[ -f "${name}" ]]; then
      echo "[${prefix}] skip existing ${name}"
      continue
    fi
    echo "[${prefix}] fetching ${name} ..."
    # -f: fail on HTTP errors; -L: redirects; -C -: resume interrupted downloads into the same file
    curl -fL --retry 3 --retry-delay 5 --continue-at - -o "${name}" "${url}"
  done
done

echo
echo "Done. Unzip *_images.zip → images/ , *_labels.zip → labels/ (see DAFormer/MIC preprocessing), then run:"
echo "  python tools/convert_datasets/gta.py ${OUT_DIR} --nproc 8"
